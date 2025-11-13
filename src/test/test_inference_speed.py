import time
import torch
from torchvision.transforms import Compose, Resize, ToTensor, Normalize

import torchvision.models as models

def benchmark_model(model, input_tensor, model_name, num_iterations=100):
  """Benchmark inference speed of a model."""
  model.eval()
  
  # Warmup
  with torch.no_grad():
    for _ in range(10):
      _ = model(input_tensor)
  
  # Measure
  start_time = time.time()
  with torch.no_grad():
    for _ in range(num_iterations):
      _ = model(input_tensor)
  end_time = time.time()
  
  avg_time = (end_time - start_time) / num_iterations * 1000  # ms
  print(f"{model_name}: {avg_time:.2f} ms per inference")
  return avg_time

def main():
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  device = torch.device("mps" if torch.mps.is_available() else "cpu")
  print(f"Using device: {device}\n")
  
  # Create dummy input
  input_tensor = torch.randn(1, 3, 480, 640).to(device)
  
  # Load models
  efficientnet_b0 = models.efficientnet_b0(pretrained=True).to(device)
  efficientnet_v2_s = models.efficientnet_v2_s(pretrained=True).to(device)
  efficientnet_v2_l = models.efficientnet_v2_l(pretrained=True).to(device)
  mobilenet_v2 = models.mobilenet_v2(pretrained=True).to(device)
  mobilenet_v3_s = models.mobilenet_v3_small(pretrained=True).to(device)
  mobilenet_v3_l = models.mobilenet_v3_large(pretrained=True).to(device)
  
  # Benchmark
  print("Benchmarking inference speed (100 iterations):\n")
  benchmark_model(efficientnet_b0, input_tensor, "EfficientNet B0")
  benchmark_model(efficientnet_v2_s, input_tensor, "EfficientNet V2 s")
  benchmark_model(efficientnet_v2_l, input_tensor, "EfficientNet V2 l")
  benchmark_model(mobilenet_v2, input_tensor, "MobileNet V2")
  benchmark_model(mobilenet_v3_s, input_tensor, "MobileNet V3 s")
  benchmark_model(mobilenet_v3_l, input_tensor, "MobileNet V3 l")

if __name__ == "__main__":
  main()