"""The cell, in Isaac Sim: the same HTTP API as the real cell and mock_cell.py, with physics, rendered cameras and
several objects on the table. Point the unchanged client at it:

    # on the Spark (from the fanuc_er4ia_sim folder, with mock_cell.py and calibration/ next to it)
    "$ISAAC/python.sh" isaac_cell.py --trials 5 --cubes 3 --batteries 2 --caps 1 --seed 1

    # on the Mac, through `ssh -L 9098:127.0.0.1:9098 ...`
    uv run run.py --url http://127.0.0.1:9098 --token mock

Each trial puts the objects at a fresh random layout (seed + trial index), the arm at home, and scores the trial
when the policy has been idle for --idle-s or --trial-s has passed: how many objects ended up in the bin. Scores
and before/after table images go to --out. The TCP actions are solved to joints with the starter's own IK, so the
sim and the real cell share frames: UF0 mm, z_UF0 = z_base - 330, R = J1 + J6.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import queue
import random
import sys
import threading
import time
from http.server import ThreadingHTTPServer
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), os.environ.get("HACK_REPO", str(HERE))]

p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
p.add_argument("--port", type=int, default=9098)
p.add_argument("--trials", type=int, default=5)
p.add_argument("--trial-s", type=float, default=240.0, help="wall-clock limit per trial")
p.add_argument("--idle-s", type=float, default=20.0, help="end a trial when the policy has sent nothing for this long")
p.add_argument("--cubes", type=int, default=3)
p.add_argument("--batteries", type=int, default=2)
p.add_argument("--caps", type=int, default=0)
p.add_argument("--seed", type=int, default=1)
p.add_argument("--bin", default="0.44,0.20", help="bin centre x,y in metres (fixed for the whole launch)")
p.add_argument("--instruction", default="Put the cube into the yellow bin.")
p.add_argument("--out", default="sim_runs")
args = p.parse_args()

COLOURS = ["yellow", "blue", "red"]
AA_R, AA_LEN, AA_MASS = 0.00725, 0.050, 0.023
BIN_OUTER, BIN_POCKET = 0.070, 0.048
BIN_XY = tuple(float(v) for v in args.bin.split(","))
N_CUBES, N_BATT, N_CAPS = args.cubes, args.batteries, args.caps


def make_layout(seed: int) -> list[dict]:
    """Random positions for every object: reachable, apart from each other and from the bin."""
    rng = random.Random(seed)
    items = [("cube", i) for i in range(N_CUBES)] + [("battery", i) for i in range(N_BATT)] + [("cap", i) for i in range(N_CAPS)]
    placed: list[dict] = []
    for kind, i in items:
        for _ in range(10000):
            x, y = rng.uniform(0.25, 0.47), rng.uniform(-0.22, 0.22)
            if not 0.22 < math.hypot(x, y) < 0.50 or math.dist((x, y), BIN_XY) < 0.10:
                continue
            if any(math.dist((x, y), (o["x"], o["y"])) < 0.075 for o in placed):
                continue
            placed.append({"kind": kind, "i": i, "x": x, "y": y, "yaw": rng.uniform(-90, 90)})
            break
        else:
            raise SystemExit("cannot place the objects apart; use fewer objects")
    return placed


def name_of(o: dict) -> str:
    return {"cube": "SortCube", "battery": "SortBattery", "cap": "SortCap"}[o["kind"]] + str(o["i"])


layout0 = make_layout(args.seed)

from isaacsim import SimulationApp  # noqa: E402

app = SimulationApp({"headless": True})

import cv2  # noqa: E402
import omni.usd  # noqa: E402
from isaacsim.core.api import World  # noqa: E402
from isaacsim.core.prims import SingleArticulation, SingleXFormPrim  # noqa: E402
from isaacsim.core.utils.types import ArticulationAction  # noqa: E402
from isaacsim.sensors.camera import Camera  # noqa: E402
from pxr import Gf, UsdGeom, UsdPhysics  # noqa: E402

import cell  # noqa: E402
import fk  # noqa: E402
import ik  # noqa: E402
import mock_cell  # noqa: E402
import scene_config as C  # noqa: E402

Z0 = C.TABLE_TOP_Z + C.SURFACE_DZ          # the work surface in the world, m
UF0_DZ = 330.0                              # z_UF0 = z_base - 330 mm
STEP = 1 / 60

# the objects the starter builds come from these lists
C.SORT_CUBES = [(o["x"], o["y"], COLOURS[o["i"] % 3]) for o in layout0 if o["kind"] == "cube"]
C.SORT_CUBE_YAWS_DEG = [o["yaw"] for o in layout0 if o["kind"] == "cube"]
C.SORT_CAPS = [(o["x"], o["y"]) for o in layout0 if o["kind"] == "cap"]
C.CAP_ORIENTATIONS = ["narrow_end_up"] * N_CAPS
C.SORT_VIALS = []


def add_battery(stage, i: int) -> None:
    """An AA cell lying on the table: a dark cylinder along its local x, origin at its centre."""
    g = UsdGeom.Xform.Define(stage, f"/World/SortBattery{i}")
    body = UsdGeom.Cylinder.Define(stage, f"/World/SortBattery{i}/Collider")
    body.CreateAxisAttr("X")
    body.CreateRadiusAttr(AA_R)
    body.CreateHeightAttr(AA_LEN)
    body.CreateDisplayColorAttr([Gf.Vec3f(0.10, 0.10, 0.11)])
    UsdPhysics.CollisionAPI.Apply(body.GetPrim())
    cell.bind_physics_material(body.GetPrim())
    cell.small_contact_offsets(body.GetPrim())
    cell.make_dynamic(g.GetPrim(), AA_MASS, collide_self=False)


def pose_of(o: dict) -> tuple[np.ndarray, np.ndarray]:
    """World position and (w, x, y, z) for an object's origin: bottom face for cubes and caps, centre for a battery."""
    z = Z0 + (AA_R + 0.0005 if o["kind"] == "battery" else 0.0005)
    h = math.radians(o["yaw"]) / 2
    return np.array([o["x"], o["y"], z]), np.array([math.cos(h), 0.0, 0.0, math.sin(h)])


world = World(stage_units_in_meters=1.0, physics_dt=STEP, rendering_dt=1 / 30)
stage = omni.usd.get_context().get_stage()
cell.build(stage)
for i in range(N_BATT):
    add_battery(stage, i)
    o = next(o for o in layout0 if o["kind"] == "battery" and o["i"] == i)
    pos, quat = pose_of(o)
    UsdGeom.Xformable(stage.GetPrimAtPath(f"/World/SortBattery{i}")).AddTranslateOp().Set(Gf.Vec3d(*pos))
    UsdGeom.Xformable(stage.GetPrimAtPath(f"/World/SortBattery{i}")).AddRotateZOp().Set(o["yaw"])
cell._make_bin(stage, "/World/Bin", xy=BIN_XY, outer=(BIN_OUTER, BIN_OUTER), pocket_size=BIN_POCKET, pocket_r=0.008,
               wall_h=0.050, floor_t=0.012, mat_name="bin_yellow")

cams = {"table": Camera(prim_path=cell.TABLE_CAM_PATH, resolution=(1920, 1080)),
        "wrist": Camera(prim_path=cell.WRIST_CAM_PATH, resolution=(1280, 720))}
cell.enable_ccd(world)
world.reset()
for c in cams.values():
    c.initialize()

arm = SingleArticulation(prim_path=C.ROBOT_PRIM, name="er4ia")
arm.initialize()
dof = list(arm.dof_names)
ARM_IDX = [dof.index(f"J{i}") for i in range(1, 7)]
JAW_IDX = {name: dof.index(name) for name in cell.jaw_targets(C.JAW_OPEN)}
LIMITS = np.deg2rad([C.JOINT_LIMITS_DEG[f"J{i}"] for i in range(1, 7)])
ik.set_tool_axis(C.TOOL_AXIS)
ik.set_spread_axis(C.FINGER_SPREAD_AXIS)
TOOL = tuple(np.array(C.TOOL_AXIS) * C.TOOL_LENGTH)

objects = {name_of(o): SingleXFormPrim("/World/" + name_of(o)) for o in layout0}


class Sim:
    """Everything that touches Isaac. Runs on the main thread; HTTP threads reach it through run_on_main."""

    def __init__(self) -> None:
        self.jobs: queue.Queue = queue.Queue()
        self.q_cmd = np.deg2rad(mock_cell.HOME[:6])
        self.g = 0.0
        self.version = 0
        self.rendered = -1
        self.frames: dict[str, np.ndarray] = {}
        self.last_action = time.time()
        self.ik_failures = 0
        self.layout = layout0
        self.start_poses: dict[str, dict] = {}

    # ---- main-thread plumbing
    def run_on_main(self, fn, timeout: float = 120.0):
        ev, box = threading.Event(), []
        self.jobs.put((fn, ev, box))
        if not ev.wait(timeout):
            raise RuntimeError("the simulator did not answer")
        if isinstance(box[0], Exception):
            raise box[0]
        return box[0]

    def serve_jobs(self) -> bool:
        did = False
        while True:
            try:
                fn, ev, box = self.jobs.get_nowait()
            except queue.Empty:
                return did
            try:
                box.append(fn())
            except Exception as e:  # noqa: BLE001  # handed back to the waiting thread
                box.append(e)
            ev.set()
            did = True

    # ---- the arm
    def command(self, q6, g: float) -> None:
        full = arm.get_joint_positions().copy()
        full[ARM_IDX] = q6
        for joint, value in cell.jaw_targets(C.JAW_OPEN * (1.0 - float(g))).items():
            full[JAW_IDX[joint]] = value
        arm.apply_action(ArticulationAction(joint_positions=full))

    def solve(self, x_mm: float, y_mm: float, z_uf0_mm: float, r_deg: float) -> np.ndarray:
        th = math.radians(r_deg)
        spread = (-math.sin(th), math.cos(th), 0.0)
        target = [x_mm / 1000, y_mm / 1000, (z_uf0_mm + UF0_DZ) / 1000]
        for iters, damping in ((200, 0.05), (600, 0.02)):
            q, ok = ik.solve(self.q_cmd, target, [0.0, 0.0, -1.0], LIMITS, iters=iters, damping=damping, tool_offset=TOOL,
                             spread_dir=spread)
            if ok:
                return q
        self.ik_failures += 1
        print(f"[sim] IK did not converge for {x_mm:.0f}, {y_mm:.0f}, {z_uf0_mm:.0f}, R {r_deg:.0f}", flush=True)
        return q

    def play(self, item) -> None:
        row, dt, kind = item
        if kind == "tcp":
            x, y, z, _w, _p, r, g = row
            q6 = self.solve(x, y, z, r)
        else:
            q6, g = np.deg2rad(row[:6]), row[6]
        self.q_cmd, self.g = q6, float(g)
        for _ in range(max(1, round(dt / STEP))):
            self.command(q6, g)
            world.step(render=False)
        self.version += 1
        self.last_action = time.time()
        self.publish()

    def state(self) -> tuple[list, list]:
        q = arm.get_joint_positions()[ARM_IDX]
        _, tip = fk.tool_pose(q, TOOL)
        d = np.degrees(q)
        joints = [float(v) for v in d] + [self.g]
        tcp = [float(tip[0] * 1000), float(tip[1] * 1000), float(tip[2] * 1000 - UF0_DZ), -180.0, 0.0, float(d[0] + d[5])]
        return joints, tcp

    def publish(self) -> None:
        cell_.q, cell_.tcp = self.state()

    # ---- cameras
    def render(self) -> None:
        if self.rendered == self.version:
            return
        for _ in range(3):
            self.command(self.q_cmd, self.g)
            world.step(render=True)
        for name, cam in cams.items():
            rgba = cam.get_rgba()
            if rgba is None or rgba.size == 0:
                raise RuntimeError(f"the {name} camera returned no image")
            self.frames[name] = np.asarray(rgba)[:, :, :3].astype(np.uint8)
        self.rendered = self.version

    def snapshot(self, name: str) -> np.ndarray:
        def job():
            self.render()
            return self.frames[name]
        return self.run_on_main(job)

    # ---- trials
    def positions(self) -> dict[str, dict]:
        out = {}
        for o in self.layout:
            pos, quat = objects[name_of(o)].get_world_pose()
            z_c = pos[2] + (C.CUBE_EDGE / 2 if o["kind"] == "cube" else C.CAP_HEIGHT / 2 if o["kind"] == "cap" else 0.0)
            out[name_of(o)] = {"kind": o["kind"], "x": float(pos[0]), "y": float(pos[1]), "z_centre": float(z_c)}
        return out

    def reset_trial(self, index: int) -> None:
        self.layout = make_layout(args.seed + index)
        by_name = {name_of(o): o for o in self.layout}
        prims = {n: UsdPhysics.RigidBodyAPI(stage.GetPrimAtPath("/World/" + n)) for n in objects}
        for n, rb in prims.items():
            rb.CreateKinematicEnabledAttr().Set(True)
        q = np.deg2rad(mock_cell.HOME[:6])
        full = arm.get_joint_positions().copy()
        full[ARM_IDX] = q
        for joint, value in cell.jaw_targets(C.JAW_OPEN).items():
            full[JAW_IDX[joint]] = value
        arm.set_joint_positions(full)
        self.q_cmd, self.g = q, 0.0
        for n, o in by_name.items():
            pos, quat = pose_of(o)
            objects[n].set_world_pose(pos, quat)
        for _ in range(20):
            self.command(q, 0.0)
            world.step(render=False)
        for rb in prims.values():
            rb.CreateKinematicEnabledAttr().Set(False)
        for _ in range(60):
            self.command(q, 0.0)
            world.step(render=False)
        self.version += 1
        self.start_poses = self.positions()
        self.last_action = time.time()
        self.publish()

    def score(self, index: int, out_dir: Path) -> dict:
        self.render()
        cv2.imwrite(str(out_dir / f"trial_{index:02d}_end.png"), cv2.cvtColor(self.frames["table"], cv2.COLOR_RGB2BGR))
        end = self.positions()
        half = BIN_POCKET / 2 + 0.004
        rows = {}
        for n, e in end.items():
            s = self.start_poses[n]
            rows[n] = {"kind": e["kind"], "start_mm": [round(s["x"] * 1000), round(s["y"] * 1000)],
                       "end_mm": [round(e["x"] * 1000), round(e["y"] * 1000)],
                       "in_bin": bool(abs(e["x"] - BIN_XY[0]) < half and abs(e["y"] - BIN_XY[1]) < half and e["z_centre"] > Z0 + 0.008),
                       "moved_mm": round(1000 * math.dist((s["x"], s["y"]), (e["x"], e["y"])), 1)}
        result = {"trial": index, "seed": args.seed + index, "bin_mm": [round(BIN_XY[0] * 1000), round(BIN_XY[1] * 1000)],
                  "in_bin": sum(r["in_bin"] for r in rows.values()), "objects": len(rows), "ik_failures": self.ik_failures,
                  "detail": rows}
        (out_dir / f"trial_{index:02d}.json").write_text(json.dumps(result, indent=1))
        verdicts = ", ".join(f"{n} {'in' if r['in_bin'] else 'out'}" for n, r in rows.items())
        print(f"[sim] trial {index}: {result['in_bin']}/{result['objects']} in the bin ({verdicts})", flush=True)
        return result

    def save_start(self, index: int, out_dir: Path) -> None:
        self.render()
        cv2.imwrite(str(out_dir / f"trial_{index:02d}_start.png"), cv2.cvtColor(self.frames["table"], cv2.COLOR_RGB2BGR))


sim = Sim()
out_dir = Path(args.out) / time.strftime("%Y%m%d-%H%M%S")
out_dir.mkdir(parents=True, exist_ok=True)


class IsaacCell(mock_cell.Mock):
    def _play(self) -> None:  # the main thread plays the queue
        return

    def card(self, cam: str) -> np.ndarray:
        return sim.snapshot(cam)

    def _operator(self) -> None:
        results = []
        for t in self.trials:
            with self.lock:
                self.queue.clear()
                self.active, self.state = t, "homing"
            sim.run_on_main(lambda i=t["index"]: sim.reset_trial(i))
            sim.run_on_main(lambda i=t["index"]: sim.save_start(i, out_dir))
            with self.lock:
                self.state, t["status"], t["instruction"] = "idle", "ready", self.instruction
            self.current = t["index"] + 1
            print(f"[sim] {t['trial_id']} ready", flush=True)
            while t["status"] == "ready":
                time.sleep(0.05)
            t0 = time.time()
            sim.last_action = t0
            while time.time() - t0 < self.trial_s:
                if self.stats["actions"] and time.time() - sim.last_action > args.idle_s:
                    break
                time.sleep(0.2)
            with self.lock:
                self.queue.clear()
                t["status"], self.state = "done", "homing"
            results.append(sim.run_on_main(lambda i=t["index"]: sim.score(i, out_dir)))
            self.rollouts.append({"actions": self.stats["actions"], "rejected": self.stats["rejected"], "latency": []})
            self.stats.clear()
            time.sleep(1.0)
        total = sum(r["in_bin"] for r in results), sum(r["objects"] for r in results)
        (out_dir / "summary.json").write_text(json.dumps({"in_bin": total[0], "objects": total[1], "trials": results}, indent=1))
        print(f"[sim] all trials done: {total[0]}/{total[1]} objects in the bin; results in {out_dir}", flush=True)
        with self.lock:
            self.active, self.state = None, "idle"


mock_cell.NATIVE = {"table": (1920, 1080), "side": (1920, 1080), "wrist": (1280, 720)}
# warm the shaders up: the first rendered frames are not final
for _ in range(90):
    world.step(render=True)
cell_ = IsaacCell(args.trials, args.trial_s, "mock", args.instruction, quiet=True)
server = ThreadingHTTPServer(("127.0.0.1", args.port), mock_cell.handler(cell_))
threading.Thread(target=server.serve_forever, daemon=True).start()
print(f"[sim] cell on http://127.0.0.1:{args.port}, token 'mock', {args.trials} trials, "
      f"{N_CUBES} cubes + {N_BATT} batteries + {N_CAPS} caps, bin at {BIN_XY}; results in {out_dir}", flush=True)

while app.is_running():
    did = sim.serve_jobs()
    item = None
    if cell_.lock.acquire(blocking=False):
        try:
            if cell_.queue and cell_.state == "running":
                item = cell_.queue.popleft()
                cell_.stats["actions"] += 1
                cell_.step += 1
        finally:
            cell_.lock.release()
    if item is not None:
        sim.play(item)
    elif not did:
        time.sleep(0.003)
    if cell_.active is None and cell_.state == "idle" and len(cell_.rollouts) >= args.trials:
        break
app.close()
