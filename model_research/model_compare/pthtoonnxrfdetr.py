from rfdetr import RFDETRBase

model = RFDETRBase(pretrain_weights=r"C:\Users\istd.magang17\Downloads\checkpoint_best_ema.pth")

model.export(format="onnx",output_dir="out/", fp16=True)