import torch
from torchvision.models import mobilenet_v3_large, resnet50

def build_mobile_net(num_classes: int):
    model = mobilenet_v3_large(weights=None)
    in_features = model.classifier[-1].in_features
    model.classifier[-1] = torch.nn.Linear(in_features, num_classes)
    return model

def build_resnet(num_classes: int):
    model = resnet50(weights=None)
    in_features = model.fc.in_features
    model.fc = torch.nn.Linear(in_features, num_classes)
    return model

def load_checkpoint(model, ckpt_path: str):
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
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
    CKPT = "/home/lukasb/Documents/NoisyLabelDefectDetection/logs/train/SurfaceDefectDetection/clean/2026-02-20_08-15-54_ce_resnet/run_1/checkpoints/epoch_153-val_f1_0.7870.ckpt"
    ONNX = "resnet.onnx"
    NUM_CLASSES = 9  # <-- set this

    model = build_resnet(NUM_CLASSES)
    load_checkpoint(model, CKPT)
    export_onnx(model, ONNX)
