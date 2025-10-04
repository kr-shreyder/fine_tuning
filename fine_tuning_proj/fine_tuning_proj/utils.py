import torch
import numpy as np
import random
import os

def set_seed(seed: int = 42):
    """Фиксирует все генераторы случайных чисел (Python, NumPy, PyTorch)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        # Эти настройки гарантируют детерминизм (воспроизводимость) на GPU [cite: 14]
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False