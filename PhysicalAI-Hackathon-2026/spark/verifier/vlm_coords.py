import base64, json, re, glob, math, urllib.request, cv2, numpy as np
import cube_vision as V, policy as P
def ask(png, prompt):
    body=json.dumps({"image_b64":base64.b64encode(png).decode(),"prompt":prompt,"max_new_tokens":60}).encode()
    r=urllib.request.urlopen(urllib.request.Request("http://127.0.0.1:8765/ask",body,{"Content-Type":"application/json"}),timeout=60); return json.loads(r.read())
frames=sorted(glob.glob("debug/20261030-22*_look0_raw.png"))+sorted(glob.glob("debug/20261030-21*_look0_raw.png"))
errs=[]; shown=0
for f in frames:
    img=cv2.imread(f); rgb=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    if rgb.shape[:2]!=(1080,1920): continue
    for c in V.detect_cubes(rgb,P.COLOURS,None):
        if c["iou"]<0.6: continue
        u,v=c["px"]; col=c["colour"]
        prompt=(f"This image is 1920 pixels wide and 1080 pixels tall. Find the small {col} cube on the white table. "
                f"Reply with JSON only: {{\"x\": <pixel column of the cube centre>, \"y\": <pixel row of the cube centre>}}")
        ok,png=cv2.imencode(".png",img); out=ask(png.tobytes(),prompt); raw=out.get("raw","")
        m=re.findall(r"-?\d+\.?\d*",raw)
        j=out.get("json") or {}
        def num(v):
            while isinstance(v,(list,tuple)): v=v[0]
            return float(v)
        if shown<2: print("raw reply:",raw[:100].replace("\n"," "))
        if "x" in j and "y" in j: x,y=num(j["x"]),num(j["y"])
        elif len(m)>=2: x,y=float(m[0]),float(m[1])
        else: print("no coords:",raw[:60]); continue
        x,y=x/1000*1920,y/1000*1080   # the model answers on a 0-1000 scale
        e=math.hypot(x-u,y-v); errs.append(e); mm=[ (c['xy_mm']) ]
        if shown<8: print(f"{col:6s} reference px ({u:.0f},{v:.0f})  model says ({x:.0f},{y:.0f})  error {e:.0f} px"); shown+=1
    if len(errs)>=24: break
e=np.array(errs); mt=e[e<50]; print(f"right cube found: {len(mt)}/{len(e)}; among those: median {np.median(mt):.1f} px, max {mt.max():.0f} px; wrong cube (another of the same colour): {(e>=50).sum()}/{len(e)}")
print(f"\n(1 px is about 1 mm on the table at this camera)\n{len(e)} cubes: median error {np.median(e):.0f} px, 90th percentile {np.percentile(e,90):.0f} px, within 5 px: {(e<=5).sum()}/{len(e)}, within 2 px: {(e<=2).sum()}/{len(e)}")
