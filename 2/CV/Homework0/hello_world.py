import cv2
import numpy as np

def predict(img: np.ndarray, model_path: str) -> np.ndarray:
    m = np.load(model_path)
    return (img - m[0])/m[1]

def train(img: np.ndarray, save_model_path: str) -> None:
    np.save(save_model_path, np.array([img.mean(), img.std()]))