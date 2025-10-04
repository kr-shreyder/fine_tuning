import torch
import torch.nn as nn
import timm
from pathlib import Path
from fine_tuning_proj.config import GlobalConfig, DataConfig, ModelConfig, PROJ_ROOT # Импорт PROJ_ROOT!
from fine_tuning_proj.utils import set_seed

set_seed(GlobalConfig.SEED)
DEVICE = 'cpu'
data_cfg = DataConfig()
model_cfg = ModelConfig()

FINAL_MODEL_NAME = model_cfg.MODEL_2_NAME # 'vit_base_patch16_224'
WEIGHTS_PATH = PROJ_ROOT / "models" / "best_vit_weights.pth" 

def load_and_adapt_model(model_name, num_classes, weights_path):
    """Инициализирует, адаптирует и загружает веса ViT."""
    model = timm.create_model(model_name, pretrained=False)
    
    # Адаптация Head для ViT
    num_ftrs = model.head.in_features
    model.head = nn.Linear(num_ftrs, num_classes)
        
    # Загрузка обученных весов
    model.load_state_dict(torch.load(weights_path, map_location=DEVICE))
    
    print(f"Веса из {weights_path.name} успешно загружены.")
    
    model.to(DEVICE).eval()
    return model

def export_to_onnx(model, onnx_path):
    # Фиктивный входной тензор для экспорта (BATCH_SIZE=1)
    dummy_input = torch.randn(1, 3, data_cfg.IMG_SIZE, data_cfg.IMG_SIZE, device=DEVICE)
    
    onnx_file_path = Path(onnx_path)
    onnx_file_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Экспорт в ONNX
    torch.onnx.export(
        model, 
        dummy_input, 
        onnx_file_path,
        export_params=True, 
        opset_version=14, 
        do_constant_folding=True, 
        input_names=['input'], 
        output_names=['output'],
        dynamic_axes={'input': {0: 'batch_size'}}
    )
    print(f"Модель {FINAL_MODEL_NAME} успешно экспортирована в ONNX: {onnx_file_path}")

def main():
    model = load_and_adapt_model(FINAL_MODEL_NAME, data_cfg.NUM_CLASSES, WEIGHTS_PATH)
    
    export_to_onnx(model, model_cfg.ONNX_PATH)

if __name__ == "__main__":
    main()