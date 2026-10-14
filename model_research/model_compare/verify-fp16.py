"""Verify and benchmark the FP16 TensorRT engine against the FP32 engine."""

import time

import numpy as np
import torch
from polygraphy.backend.trt import engine_from_bytes

ENGINE_FP16 = "output/rfdetr-medium_fp16.trt"
ENGINE_FP32 = "output/rfdetr-medium_fp32.trt"
WARMUP = 10
ITERS = 50


def run(engine, name: str, x_half: torch.Tensor, x_float: torch.Tensor) -> dict:
    ctx = engine.create_execution_context()
    stream = torch.cuda.Stream().cuda_stream

    in_name = next(engine.get_tensor_name(i) for i in range(engine.num_io_tensors)
                   if engine.get_tensor_mode(engine.get_tensor_name(i)).name == "INPUT")
    in_half = "HALF" in str(engine.get_tensor_dtype(in_name))
    x = (x_half if in_half else x_float).contiguous()
    ctx.set_tensor_address(in_name, x.data_ptr())

    outputs = []
    for i in range(engine.num_io_tensors):
        n = engine.get_tensor_name(i)
        if engine.get_tensor_mode(n).name == "OUTPUT":
            o = torch.empty(tuple(engine.get_tensor_shape(n)), dtype=torch.half if in_half else torch.float32).cuda()
            ctx.set_tensor_address(n, o.data_ptr())
            outputs.append((n, o))

    for _ in range(WARMUP):
        ctx.execute_async_v3(stream)
    torch.cuda.synchronize()

    start = time.perf_counter()
    for _ in range(ITERS):
        ctx.execute_async_v3(stream)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    print(f"[{name}] latency={elapsed / ITERS * 1000:.2f} ms")
    return {n: o.detach().cpu().numpy() for n, o in outputs}


def main() -> None:
    with open(ENGINE_FP16, "rb") as f:
        eng16 = engine_from_bytes(f.read())
    with open(ENGINE_FP32, "rb") as f:
        eng32 = engine_from_bytes(f.read())

    x_float = torch.randn((1, 3, 576, 576), dtype=torch.float32).cuda()
    x_half = x_float.half()

    out16 = run(eng16, "fp16", x_half, x_float)
    out32 = run(eng32, "fp32", x_half, x_float)

    print("\nSanity check (fp16 vs fp32):")
    for (name16, a), (name32, b) in zip(out16.items(), out32.items()):
        a, b = a.astype(np.float32), b.astype(np.float32)
        denom = np.maximum(np.abs(b), 1e-3)
        rel = np.abs(a - b) / denom
        print(f"  {name16}: shape={a.shape} max_rel_err={rel.max():.4f} mean_rel_err={rel.mean():.6f}")


if __name__ == "__main__":
    main()