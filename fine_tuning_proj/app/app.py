import onnxruntime
import numpy as np
from PIL import Image
# ИСПРАВЛЕНИЕ: Добавлен импорт Path
from pathlib import Path
from torchvision.transforms import transforms as T
import gradio as gr

# --- КОНФИГУРАЦИЯ (Должна совпадать с config.py и dataset.py) ---
CLASS_NAMES = ["Lily", "Orchid", "Peony"] 
IMG_SIZE = 224
ONNX_MODEL_PATH = "model.onnx" 

# 1. Предварительная Обработка (КРИТИЧНО! ДОЛЖНА СОВПАДАТЬ С ОБУЧЕНИЕМ)
NORMALIZE_MEAN = [0.485, 0.456, 0.406]
NORMALIZE_STD = [0.229, 0.224, 0.225]

preprocess = T.Compose([
    T.Resize(256),      
    T.CenterCrop(IMG_SIZE),
    T.ToTensor(),       
    T.Normalize(NORMALIZE_MEAN, NORMALIZE_STD)
])

# 2. Загрузка ONNX Runtime Сессии
try:
    # Используем Path(__file__).parent для получения пути к папке 'app'
    model_full_path = Path(__file__).parent / ONNX_MODEL_PATH
    sess = onnxruntime.InferenceSession(str(model_full_path))
    print(f"✅ ONNX модель успешно загружена из {ONNX_MODEL_PATH}")
# ИСПРАВЛЕНИЕ: Использование общего исключения, чтобы избежать AttributeError
except Exception as e:
    print(f"❌ Ошибка загрузки ONNX Runtime: {e}")
    sess = None

# 3. Функция Прогнозирования
def predict(input_image: Image.Image):
    """Принимает PIL Image, обрабатывает и выполняет инференс с помощью ONNX."""
    if sess is None:
        return {"Error": "ONNX Runtime не инициализирован. Проверьте модель."}

    # 1. Пре-процессинг
    input_tensor = preprocess(input_image).unsqueeze(0).numpy()
    
    # 2. Инференс
    input_name = sess.get_inputs()[0].name
    output_name = sess.get_outputs()[0].name
    
    onnx_output = sess.run([output_name], {input_name: input_tensor})
    
    # 3. Пост-процессинг: Softmax и форматирование
    logits = onnx_output[0]
    # Используем numpy.softmax вручную
    exp_logits = np.exp(logits)
    probabilities = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
    
    # Форматирование для Gradio
    confidences = {CLASS_NAMES[i]: float(probabilities[0, i]) for i in range(len(CLASS_NAMES))}
    
    return confidences

# 4. Запуск интерфейса Gradio
if sess is not None:
    gr.Interface(
        fn=predict,
        inputs=gr.Image(type="pil", label="Загрузите изображение для классификации"),
        outputs=gr.Label(num_top_classes=3),
        title="ViT Классификатор Цветов (ONNX Runtime)",
        description="Используется лучшая модель (ViT), экспортированная для быстрого инференса на CPU.",
    ).launch(share=True) 