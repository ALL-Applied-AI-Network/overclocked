"""Export the C-RADIO backbone (summary vector) to ONNX, build a TensorRT FP16 engine on the GB10, compare latency with torch."""
import os, sys, time, numpy as np, torch
from transformers import AutoModel
import tensorrt as trt
dev = "cuda"
m = AutoModel.from_pretrained("nvidia/C-RADIOv2-B", trust_remote_code=True).eval().to(dev)
class Summary(torch.nn.Module):
    def __init__(self, m): super().__init__(); self.m = m
    def forward(self, x): return self.m(x)[0]
net = Summary(m).eval()
x = torch.rand(1, 3, 224, 224, device=dev)
with torch.no_grad(): ref = net(x).float().cpu().numpy()
def bench(fn, n=100):
    for _ in range(10): fn()
    torch.cuda.synchronize(); t = time.time()
    for _ in range(n): fn()
    torch.cuda.synchronize(); return (time.time() - t) / n * 1000
with torch.no_grad():
    t32 = bench(lambda: net(x))
    with torch.autocast("cuda", dtype=torch.float16): t16 = bench(lambda: net(x))
print("torch fp32 %.2f ms, torch fp16 autocast %.2f ms" % (t32, t16), flush=True)
with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16): ref16 = net(x).float().cpu().numpy()
onnx_path = "cradio_summary_fp16.onnx"
try:
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16): torch.onnx.export(net, (x,), onnx_path, input_names=["image"], output_names=["summary"], opset_version=17, dynamo=False)
except Exception as e:
    print("legacy export failed:", repr(e)[:300], flush=True)
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16): torch.onnx.export(net, (x,), onnx_path, input_names=["image"], output_names=["summary"], opset_version=18)
print("onnx written", os.path.getsize(onnx_path) // 1e6, "MB", flush=True)
logger = trt.Logger(trt.Logger.WARNING)
builder = trt.Builder(logger)
network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))
parser = trt.OnnxParser(network, logger)
ok = parser.parse_from_file(onnx_path)
if not ok:
    for i in range(parser.num_errors): print("onnx parse error:", parser.get_error(i));
    sys.exit(1)
cfg = builder.create_builder_config()
t = time.time(); blob = builder.build_serialized_network(network, cfg); print("engine built in %.0fs, %.0f MB" % (time.time() - t, blob.nbytes / 1e6), flush=True)
rt = trt.Runtime(logger); eng = rt.deserialize_cuda_engine(blob); ctx = eng.create_execution_context()
tmap = {trt.DataType.FLOAT: torch.float32, trt.DataType.HALF: torch.float16}
names = [eng.get_tensor_name(i) for i in range(eng.num_io_tensors)]
bufs = {}
for n in names:
    dt = tmap[eng.get_tensor_dtype(n)]; shp = tuple(eng.get_tensor_shape(n))
    if eng.get_tensor_mode(n) == trt.TensorIOMode.INPUT: bufs[n] = x.to(dt).contiguous()
    else: bufs[n] = torch.empty(shp, device=dev, dtype=dt)
    ctx.set_tensor_address(n, bufs[n].data_ptr()); print("tensor", n, dt, shp, flush=True)
out = [bufs[n] for n in names if eng.get_tensor_mode(n) == trt.TensorIOMode.OUTPUT][0]
stream = torch.cuda.Stream()
def run():
    ctx.execute_async_v3(torch.cuda.current_stream().cuda_stream)
ttrt = bench(run)
run(); torch.cuda.synchronize(); trt_out = out.float().cpu().numpy()
cos = float((trt_out * ref16).sum() / (np.linalg.norm(trt_out) * np.linalg.norm(ref16)))
print("TensorRT fp16 %.2f ms; cosine similarity of the summary vector with torch fp32: %.5f" % (ttrt, cos), flush=True)
open("results_trt.json", "w").write(__import__("json").dumps({"torch_fp32_ms": t32, "torch_fp16_ms": t16, "trt_fp16_ms": ttrt, "cosine": cos}))
print("DONE", flush=True)
