from pathlib import Path
from dataclasses import dataclass, field
from dotenv import load_dotenv
from loguru import logger

# Load environment variables from .env file if it exists
load_dotenv()

# --- PATHS ---
PROJ_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJ_ROOT / "data"

# --- DATACLASSES ---

@dataclass
class GlobalConfig:
    """Общие настройки и сиды."""
    SEED: int = 42                                     # Фиксация генераторов случайных чисел [cite: 8, 14]
    
@dataclass
class DataConfig:
    """Конфигурация данных и путей."""
    # Путь к папке raw, где лежат train/val/test
    DATA_DIR_RAW: str = str(DATA_DIR / "raw")
    IMG_SIZE: int = 224                                # Стандартный размер входа для ImageNet-моделей [cite: 20]
    NUM_CLASSES: int = 3
    
@dataclass
class ModelConfig:
    """Конфигурация моделей и экспорта ONNX."""
    MODEL_1_NAME: str = 'resnet18'                     # Модель 1: CNN [cite: 23]
    MODEL_2_NAME: str = 'vit_base_patch16_224'         # Модель 2: Transformer [cite: 23]
    PRETRAINED: bool = True
    ONNX_PATH: str = str(PROJ_ROOT / "app" / "model.onnx") # Путь для экспорта в папку 'app' [cite: 29]

@dataclass
class TrainConfig:
    """Гиперпараметры для обучения."""
    EPOCHS: int = 15
    BATCH_SIZE: int = 32
    LEARNING_RATE: float = 1e-4
    OPTIMIZER: str = 'AdamW'
    FREEZE_STRATEGY_DESC: str = 'Двухэтапное дообучение: Head, затем разморозка последних слоев'