import os
import torchvision.transforms as T
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from .config import DataConfig # Импорт вашей новой конфигурации

def get_transforms(cfg: DataConfig):
    """Определяет трансформации с аугментацией для train и без для val/test."""
    # Среднее и стандартное отклонение для нормализации ImageNet [cite: 20]
    NORMALIZE_MEAN = [0.485, 0.456, 0.406]
    NORMALIZE_STD = [0.229, 0.224, 0.225]

    # 1. Трансформации для обучения (с аугментацией) [cite: 24]
    train_transforms = T.Compose([
        T.RandomResizedCrop(cfg.IMG_SIZE), # Аугментация
        T.RandomHorizontalFlip(),          # Аугментация
        T.ToTensor(),
        T.Normalize(NORMALIZE_MEAN, NORMALIZE_STD)
    ])

    # 2. Трансформации для валидации/теста (без аугментации) [cite: 32]
    # Используются те же шаги, что и для инференса, чтобы избежать отклонений
    val_test_transforms = T.Compose([
        T.Resize(256),                     # Изменение размера
        T.CenterCrop(cfg.IMG_SIZE),        # Кадрирование до 224x224
        T.ToTensor(),
        T.Normalize(NORMALIZE_MEAN, NORMALIZE_STD)
    ])
    
    return train_transforms, val_test_transforms

def get_dataloaders(cfg: DataConfig, train_transforms, val_test_transforms, batch_size: int):
    """Создает DataLoader для train, val и test."""
    
    # ImageFolder автоматически считывает классы из имен папок в data/raw/train, val, test
    train_data = ImageFolder(os.path.join(cfg.DATA_DIR_RAW, 'train'), transform=train_transforms)
    val_data = ImageFolder(os.path.join(cfg.DATA_DIR_RAW, 'val'), transform=val_test_transforms)
    test_data = ImageFolder(os.path.join(cfg.DATA_DIR_RAW, 'test'), transform=val_test_transforms)
    
    # Создание DataLoader'ов
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=2)

    return train_loader, val_loader, test_loader