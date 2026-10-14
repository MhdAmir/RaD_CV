import tensorrt as trt

ENGINE_PATH = r"C:\cctv-insight\research_amir\model_research\model_compare\output\rfdetr-medium_fp16.trt"

logger = trt.Logger(trt.Logger.INFO)

with open(ENGINE_PATH, "rb") as f:
    engine_data = f.read()

runtime = trt.Runtime(logger)
engine = runtime.deserialize_cuda_engine(engine_data)

print("=== TensorRT Engine ===")

for i in range(engine.num_io_tensors):
    name = engine.get_tensor_name(i)
    mode = engine.get_tensor_mode(name)
    shape = engine.get_tensor_shape(name)
    dtype = engine.get_tensor_dtype(name)

    print(f"\nName  : {name}")
    print(f"Mode  : {mode}")
    print(f"Shape : {shape}")
    print(f"Dtype : {dtype}")

    if mode == trt.TensorIOMode.INPUT:
        print("Input tensor")

        # Untuk dynamic shape, dimension biasanya bernilai -1
        if -1 in shape:
            print(">>> Dynamic shape detected")
        else:
            print(">>> Fixed shape detected")