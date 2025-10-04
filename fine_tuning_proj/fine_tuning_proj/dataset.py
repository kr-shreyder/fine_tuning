import os
import torchvision.transforms as T
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from .config import DataConfig

def get_transforms(cfg: DataConfig):
    """Определяет трансформации с аугментацией для train и без для val/test."""
    # Среднее и стандартное отклонение для нормализации ImageNet
    NORMALIZE_MEAN = [0.485, 0.456, 0.406]
    NORMALIZE_STD = [0.229, 0.224, 0.225]

    train_transforms = T.Compose([
        T.RandomResizedCrop(cfg.IMG_SIZE), 
        T.RandomHorizontalFlip(),         
        T.ToTensor(),
        T.Normalize(NORMALIZE_MEAN, NORMALIZE_STD)
    ])

    val_test_transforms = T.Compose([
        T.Resize(256),               
        T.CenterCrop(cfg.IMG_SIZE),  
        T.ToTensor(),
        T.Normalize(NORMALIZE_MEAN, NORMALIZE_STD)
    ])
    
    return train_transforms, val_test_transforms

def get_dataloaders(cfg: DataConfig, train_transforms, val_test_transforms, batch_size: int):
    """Создает DataLoader для train, val и test."""
    
    train_data = ImageFolder(os.path.join(cfg.DATA_DIR_RAW, 'train'), transform=train_transforms)
    val_data = ImageFolder(os.path.join(cfg.DATA_DIR_RAW, 'val'), transform=val_test_transforms)
    test_data = ImageFolder(os.path.join(cfg.DATA_DIR_RAW, 'test'), transform=val_test_transforms)
    
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=2)

    return train_loader, val_loader, test_loader