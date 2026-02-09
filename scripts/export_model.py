import torch
from torchvision.models import mobilenet_v3_large

def build_model(num_classes: int):
    model = mobilenet_v3_large(weights=None)
    in_features = model.classifier[-1].in_features
    model.classifier[-1] = torch.nn.Linear(in_features, num_classes)
    return model

def load_checkpoint(model, ckpt_path: str):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    state = ckpt.get("state_dict", ckpt)

    cleaned = {}
    for k, v in state.items():
        if k.startswith("net.model."):
            cleaned[k[len("net.model."):]] = v
        elif k.startswith("model."):
            cleaned[k[len("model."):]] = v
        elif k.startswith("net."):
            cleaned[k[len("net."):]] = v

    missing, unexpected = model.load_state_dict(cleaned, strict=True)
    print("Missing keys:", missing)
    print("Unexpected keys:", unexpected)

def export_onnx(model, onnx_path: str):
    model.eval()
    dummy = torch.randn(1, 3, 480, 480, dtype=torch.float32)
    torch.onnx.export(
        model,
        dummy,
        onnx_path,
        opset_version=17,
        do_constant_folding=True,
        input_names=["images"],
        output_names=["logits"],
        dynamic_axes=None,
        dynamo=False
    )
    print("Exported:", onnx_path)

if __name__ == "__main__":
    CKPT = "/home/lukasb/Documents/NoisyLabelDefectDetection/logs/train/SimpleDetection/full_cleaned/2026-02-09_09-09-26_ce/run_1/checkpoints/epoch_051-val_f1_0.8560.ckpt"
    ONNX = "simple_model.onnx"
    NUM_CLASSES = 3  # <-- set this

    model = build_model(NUM_CLASSES)
    load_checkpoint(model, CKPT)
    export_onnx(model, ONNX)
