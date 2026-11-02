import base64, json, time, urllib.request, cv2
labels=json.load(open("verifier_data/side_labels.json"))
PROMPT=('This is a close-up of a robot gripper from the side. Two metal fingers hang down from the tool. Is there a small object (a cube or a battery) '
        'held between the very tips of the fingers, lifted off the table? Reply with JSON only: {"held": true} or {"held": false}.')
def ask(png):
    body=json.dumps({"image_b64":base64.b64encode(png).decode(),"prompt":PROMPT,"max_new_tokens":40}).encode()
    req=urllib.request.Request("http://127.0.0.1:8765/ask",body,{"Content-Type":"application/json"})
    with urllib.request.urlopen(req,timeout=60) as r: return json.loads(r.read())
rows=[]
for k,v in labels.items():
    if v["label"] not in ("held","empty"): continue
    img=cv2.imread(v["crop"]); img=cv2.resize(img,(480,480),interpolation=cv2.INTER_CUBIC); ok,png=cv2.imencode(".png",img)
    t=time.time()
    try: out=ask(png.tobytes()); raw=out.get("raw",""); j=out.get("json") or {}; pred=j.get("held")
    except Exception as e: raw=repr(e)[:80]; pred=None
    rows.append((k,v["label"],pred,round(time.time()-t,1),raw[:60].replace("\n"," ")))
    print(rows[-1],flush=True)
tp=sum(1 for _,l,p,_,_ in rows if l=="held" and p is True); fn=sum(1 for _,l,p,_,_ in rows if l=="held" and p is False)
fp=sum(1 for _,l,p,_,_ in rows if l=="empty" and p is True); tn=sum(1 for _,l,p,_,_ in rows if l=="empty" and p is False); na=sum(1 for *_,p,_,_ in rows if p is None)
print(f"held correct {tp}, held missed {fn}, EMPTY called held (false pass) {fp}, empty correct {tn}, no answer {na}; mean latency {sum(r[3] for r in rows)/len(rows):.1f}s")
