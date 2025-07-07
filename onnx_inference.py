import cv2
import argparse
from dataset import CHARS
from PIL import Image
from time import time
import numpy as np
import onnxruntime as ort
from utils import decode_function, BeamDecoder



def batch_transform(rgb_batch, mean, std):
    """
    Обрабатывает батч изображений: resize, transpose, normalize.

    Параметры:
        rgb_batch (np.ndarray): Входной батч изображений в формате (B, H, W, C)
        mean (list or np.ndarray): Средние значения для каждого канала (C,)
        std (list or np.ndarray): Стандартные отклонения для каждого канала (C,)

    Возвращает:
        np.ndarray: Нормализованный батч в формате (B, C, H, W)
    """
    B, H, W, C = rgb_batch.shape

    # Конвертируем в float32 и делим на 255, если это uint8
    if rgb_batch.dtype == np.uint8:
        rgb_batch = rgb_batch.astype(np.float32) / 255.0

    # Resize (24, 94) — высота 24, ширина 94
    resized_batch = np.zeros((B, 24, 94, C), dtype=np.float32)
    for i in range(B):
        resized_batch[i] = cv2.resize(rgb_batch[i], (94, 24))  # (W, H)

    # Перевод в формат (B, C, H, W)
    transposed = np.transpose(resized_batch, (0, 3, 1, 2))  # (B, C, H, W)

    # Нормализация по каналам
    for i in range(transposed.shape[1]):  # для каждого канала
        transposed[:, i, :, :] = (transposed[:, i, :, :] - mean[i]) / std[i]

    contiguous_transposed = np.ascontiguousarray(transposed)

    return contiguous_transposed.astype(np.float32)


def onnx_benchmark(model_path: str, image_path: str, runs: int) -> float:
    """
    Возвращает среднее время инференса в мс
    """
    sess_options = ort.SessionOptions()

    session = ort.InferenceSession(
        model_path, providers=["CPUExecutionProvider"], sess_options=sess_options
    )
    input_name = session.get_inputs()[0].name

    batch_size = 8

    mean = np.array([0.496, 0.502, 0.504], dtype=np.float32)
    std = np.array([0.254, 0.2552, 0.2508], dtype=np.float32)

    images = np.stack([image.copy() for i in range(batch_size)])
    times = []

    for _ in range(runs):
        start = time()
        data = batch_transform(images, mean, std)

        predictions = session.run(None, {input_name: data})
        labels, prob, pred_labels = decode_function(predictions[0], CHARS, BeamDecoder)
        times.append(time() - start)
        print(labels)

    return sum(times) * 1000 / len(times) / runs


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, help="path to image")
    parser.add_argument("--model", type=str, help="path to onnx model")
    parser.add_argument("--runs", type=int, help="number of model runs", default=10)

    args = parser.parse_args()

    model = args.model
    image = Image.open(args.image)
    runs = args.runs

    time_per_image = onnx_benchmark(model, image, runs)

    print(f"per image: {time_per_image:.3f} ms")