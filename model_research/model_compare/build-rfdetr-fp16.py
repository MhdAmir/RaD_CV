"""Build a strongly-typed TensorRT engine from the FP16 ONNX model.

TensorRT 11.x removed the FP16/INT8 builder flags (weakly-typed precision flags are
gone; engines are STRONGLY_TYPED and precision comes from the network dtypes), so
the FP16 route is: convert the ONNX to float16 first (already done:
output/rfdetr-medium_fp16.onnx), then build the engine with no precision flags.
"""

from polygraphy.backend.trt import (
    CreateConfig,
    engine_from_network,
    network_from_onnx_path,
    save_engine,
)

ONNX_FP16 = r"C:\cctv-insight\research_amir\out\rfdetr-base.onnx"
ENGINE_OUT = r"C:\cctv-insight\research_amir\out\rfdetr-base_fp16.trt"


def main() -> None:
    engine = engine_from_network(
        network_from_onnx_path(ONNX_FP16),
        config=CreateConfig(),
    )
    save_engine(engine, path=ENGINE_OUT)
    print(f"Saved engine: {ENGINE_OUT}")


if __name__ == "__main__":
    main()