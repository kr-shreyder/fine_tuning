import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

def plot_learning_curves(history, model_name):
    """Строит графики потерь и точности по эпохам."""
    
    # Создаем фигуры для Loss и Accuracy
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # График Loss
    axes[0].plot(history['train_loss'], label='Train Loss')
    axes[0].plot(history['val_loss'], label='Validation Loss')
    axes[0].set_title(f'{model_name} - Loss по эпохам')
    axes[0].set_xlabel('Эпоха')
    axes[0].set_ylabel('Loss')
    axes[0].legend()

    # График Accuracy
    axes[1].plot(history['train_acc'], label='Train Accuracy')
    axes[1].plot(history['val_acc'], label='Validation Accuracy')
    axes[1].set_title(f'{model_name} - Accuracy по эпохам')
    axes[1].set_xlabel('Эпоха')
    axes[1].set_ylabel('Accuracy')
    axes[1].legend()

    plt.tight_layout()
    plt.show()

# Функция для построения матрицы ошибок
def plot_confusion_matrix(y_true, y_pred, class_names, model_name):
    """Строит матрицу ошибок."""
    
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    
    fig, ax = plt.subplots(figsize=(8, 8))
    disp.plot(cmap=plt.cm.Blues, ax=ax)
    plt.title(f'{model_name} - Матрица Ошибок')
    plt.show()