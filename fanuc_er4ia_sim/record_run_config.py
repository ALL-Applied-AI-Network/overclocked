"""Record effective scene settings before starting expensive pipeline stages."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import scene_config as C
from calibration_mismatch import configuration


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True, type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    camera_control = configuration()
    run = args.run.resolve()
    hashes = {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted([*root.glob("*.py"), *root.glob("*.sh")])
    }
    record = {
        "schema_version": 1,
        "experiment_class": "development",
        "condition": "+".join(
            (["layout_seed"] if C.CUBE_LAYOUT_SEED != 20261009 else [])
            + (["cube_yaw_" + C.CUBE_YAW_CONTROL["mode"]]
               if C.CUBE_YAW_CONTROL["mode"] != "original" else [])
            + (["camera_mismatch"] if camera_control['translation_mm'] or camera_control['rotation_deg'] else [])
        ) or "nominal",
        "run_id": run.name,
        "run_directory": str(run),
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_sha256": hashes,
        "effective": {
            "camera_mismatch": camera_control,
            "cube_layout_seed": C.CUBE_LAYOUT_SEED,
            "cube_yaws_deg": list(C.SORT_CUBE_YAWS_DEG),
            "cube_yaw_control": dict(C.CUBE_YAW_CONTROL),
            "layout": [
                {"name": f"SortCube{i}", "x_m": x, "y_m": y, "colour": colour}
                for i, (x, y, colour) in enumerate(C.SORT_CUBES)
            ],
        },
        "layout_scope": "Two yellow cubes remain fixed; four other cubes are seeded.",
        "layout_and_yaw_usage": "Reproducibility/evaluation only; not grasp targets.",
        "validation_status": "Host configuration recorded; simulator validation pending.",
    }
    with (run / "run_config.json").open("x") as output:
        json.dump(record, output, indent=2)
        output.write("\n")
    print(f"[configuration] saved {run / 'run_config.json'}")


if __name__ == "__main__":
    main()
