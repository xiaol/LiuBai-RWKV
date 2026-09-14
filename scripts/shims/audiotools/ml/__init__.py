import torch.nn as nn
class BaseModel(nn.Module):
    INTERN = []
    EXTERN = []
class Accelerator:  # referenced at import time by dac.utils; never used for encoding
    def __init__(self, *a, **k):
        raise RuntimeError("audiotools stub: Accelerator not available")
