import torch

if torch.cuda.is_available():
    print("GPU is available. Using GPU for inference.")
else:
    print("GPU is not available. Using CPU for inference.")