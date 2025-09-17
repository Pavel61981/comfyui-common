import torch
import torch.nn.functional as F
import numpy as np
import mediapipe as mp
import os
import requests
from typing import Tuple, Optional, List

# Пути к моделям (автоматическая загрузка при первом запуске)
MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")
FACE_MODEL_PATH = os.path.join(MODELS_DIR, "face_landmarker.task")
POSE_MODEL_PATH = os.path.join(MODELS_DIR, "pose_landmarker.task")

# Ссылки для загрузки моделей
FACE_MODEL_URL = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
POSE_MODEL_URL = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker/float16/1/pose_landmarker.task"

# Глобальный кэш моделей
_MODELS = {"face": None, "pose": None}


def download_model(url: str, path: str):
    """Скачивает модель, если её нет"""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if not os.path.exists(path):
        print(f"[ImageRegionExtractorQwen] Downloading model: {url.split('/')[-1]}...")
        response = requests.get(url, stream=True)
        with open(path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        print(f"[ImageRegionExtractorQwen] Model saved to {path}")


def get_face_mesh_model():
    if _MODELS["face"] is None:
        download_model(FACE_MODEL_URL, FACE_MODEL_PATH)
        base_options = mp.tasks.BaseOptions(model_asset_path=FACE_MODEL_PATH)
        options = mp.tasks.vision.FaceLandmarkerOptions(
            base_options=base_options,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=False,
            num_faces=5,
        )
        _MODELS["face"] = mp.tasks.vision.FaceLandmarker.create_from_options(options)
    return _MODELS["face"]


def get_pose_model():
    if _MODELS["pose"] is None:
        download_model(POSE_MODEL_URL, POSE_MODEL_PATH)
        base_options = mp.tasks.BaseOptions(model_asset_path=POSE_MODEL_PATH)
        options = mp.tasks.vision.PoseLandmarkerOptions(
            base_options=base_options,
            output_segmentation_masks=False,
            min_pose_detection_confidence=0.5,
            min_pose_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        _MODELS["pose"] = mp.tasks.vision.PoseLandmarker.create_from_options(options)
    return _MODELS["pose"]


def tensor_to_bgr(tensor: torch.Tensor) -> np.ndarray:
    """Преобразует тензор ComfyUI [H, W, C] в BGR numpy [H, W, 3] для MediaPipe"""
    # ComfyUI использует RGB → конвертируем в BGR
    if tensor.dim() == 3 and tensor.shape[-1] == 3:
        rgb = (tensor.cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
        return rgb[..., ::-1]  # RGB → BGR
    raise ValueError("Invalid image format")


def create_face_mask(image_shape: Tuple[int, int], face_landmarks) -> np.ndarray:
    """Создаёт маску лица через контур (без триангуляции)"""
    H, W = image_shape
    mask = np.zeros((H, W), dtype=np.uint8)

    # Основные точки для контура лица (нижняя часть)
    FACE_CONTOUR_POINTS = [
        10,
        338,
        297,
        332,
        284,
        251,
        389,
        356,
        454,
        323,
        361,
        288,
        397,
        365,
        379,
        378,
        400,
        152,
        148,
        176,
        149,
        150,
        136,
        172,
        116,
        123,
        147,
        156,
        172,
    ]

    points = []
    for idx in FACE_CONTOUR_POINTS:
        landmark = face_landmarks[idx]
        x = int(landmark.x * W)
        y = int(landmark.y * H)
        points.append([x, y])

    points = np.array(points, dtype=np.int32)
    cv2.fillPoly(mask, [points], 255)
    return mask


def create_body_mask(
    image_shape: Tuple[int, int], pose_landmarks, min_confidence: float = 0.5
) -> np.ndarray:
    """Создаёт маску тела через выпуклую оболочку точек выше порога уверенности"""
    H, W = image_shape
    mask = np.zeros((H, W), dtype=np.uint8)

    # Точки, определяющие контур тела (без рук/ног)
    BODY_POINTS = [
        11,
        12,
        23,
        24,
        26,
        25,
        28,
        27,
        30,
        29,
        32,
        31,  # Плечи, бёдра, колени, лодыжки
    ]

    points = []
    for idx in BODY_POINTS:
        landmark = pose_landmarks[idx]
        if landmark.score > min_confidelity:
            x = int(landmark.x * W)
            y = int(landmark.y * H)
            points.append([x, y])

    if len(points) >= 3:
        hull = cv2.convexHull(np.array(points))
        cv2.fillPoly(mask, [hull], 255)

    return mask


def gaussian_blur_mask(mask: torch.Tensor, radius: int) -> torch.Tensor:
    """Размытие маски гауссовым фильтром"""
    if radius <= 0:
        return mask

    # Создаём ядро
    kernel_size = radius * 2 + 1
    kernel = torch.arange(kernel_size, dtype=torch.float32) - radius
    kernel = torch.exp(-0.5 * (kernel / radius) ** 2)
    kernel = kernel / kernel.sum()
    kernel = kernel.view(1, 1, -1)

    mask = mask.unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]

    # Горизонтальное размытие
    mask = F.conv2d(mask, kernel.unsqueeze(-1), padding=(radius, 0))
    # Вертикальное размытие
    mask = F.conv2d(mask, kernel.unsqueeze(-2), padding=(0, radius))

    return mask.squeeze(0).squeeze(0)


def blend_red_mask(
    image: torch.Tensor, mask: torch.Tensor, alpha: float = 0.5
) -> torch.Tensor:
    """Накладывает полупрозрачную красную маску на изображение"""
    H, W, C = image.shape
    red_overlay = torch.zeros_like(image)
    red_overlay[:, :, 0] = 1.0  # красный канал

    mask = mask.unsqueeze(-1)  # [H, W, 1]
    blended = image * (1 - mask * alpha) + red_overlay * (mask * alpha)
    return blended


class ImageRegionExtractorQwen:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "target_region": (
                    ["face", "body", "body_without_face"],
                    {"default": "face"},
                ),
            },
            "optional": {
                "confidence_threshold": (
                    "FLOAT",
                    {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.05},
                ),
                "mask_blur_radius": (
                    "INT",
                    {"default": 5, "min": 0, "max": 50, "step": 1},
                ),
            },
        }

    RETURN_TYPES = ("IMAGE", "MASK", "IMAGE", "BOOLEAN")
    RETURN_NAMES = ("IMAGE", "MASK", "DEBUG_IMAGE", "FOUND")
    FUNCTION = "extract_region"
    CATEGORY = "image/masking"

    def extract_region(
        self,
        image: torch.Tensor,
        target_region: str,
        confidence_threshold: float = 0.5,
        mask_blur_radius: int = 5,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, bool]:
        # Поддержка батчей
        if len(image.shape) == 4:
            original_image = image[0]
        else:
            original_image = image

        # Убедимся, что [H, W, C]
        if original_image.shape[0] == 3:
            original_image = original_image.permute(1, 2, 0)

        H, W, C = original_image.shape
        bgr_image = tensor_to_bgr(original_image)

        # Создаем пустую маску
        mask = torch.zeros((H, W), dtype=torch.float32)
        found = False

        try:
            if target_region == "face":
                face_mesh = get_face_mesh_model()
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=bgr_image)
                detection_result = face_mesh.detect(mp_image)

                if detection_result.face_landmarks:
                    # Берём ПЕРВОЕ лицо
                    face_landmarks = detection_result.face_landmarks[0]
                    face_mask_np = create_face_mask((H, W), face_landmarks)
                    mask = torch.from_numpy(face_mask_np.astype(np.float32) / 255.0)
                    found = True

            elif target_region == "body":
                pose_model = get_pose_model()
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=bgr_image)
                detection_result = pose_model.detect(mp_image)

                if detection_result.pose_landmarks:
                    # Берём ПЕРВУЮ позу
                    pose_landmarks = detection_result.pose_landmarks[0]
                    body_mask_np = create_body_mask(
                        (H, W), pose_landmarks, min_confidence=confidence_threshold
                    )
                    mask = torch.from_numpy(body_mask_np.astype(np.float32) / 255.0)
                    found = True

            elif target_region == "body_without_face":
                # Получаем маску тела
                body_mask = torch.zeros((H, W), dtype=torch.float32)
                pose_model = get_pose_model()
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=bgr_image)
                detection_result = pose_model.detect(mp_image)

                if detection_result.pose_landmarks:
                    pose_landmarks = detection_result.pose_landmarks[0]
                    body_mask_np = create_body_mask(
                        (H, W), pose_landmarks, min_confidence=confidence_threshold
                    )
                    body_mask = torch.from_numpy(
                        body_mask_np.astype(np.float32) / 255.0
                    )

                # Получаем маску лица
                face_mask = torch.zeros((H, W), dtype=torch.float32)
                face_mesh = get_face_mesh_model()
                detection_result = face_mesh.detect(mp_image)

                if detection_result.face_landmarks:
                    face_landmarks = detection_result.face_landmarks[0]
                    face_mask_np = create_face_mask((H, W), face_landmarks)
                    face_mask = torch.from_numpy(
                        face_mask_np.astype(np.float32) / 255.0
                    )

                # Вычитаем лицо из тела
                mask = torch.clamp(body_mask - face_mask, 0, 1)
                found = body_mask.max() > 0.1

            else:
                raise ValueError(f"Unsupported region: {target_region}")

        except Exception as e:
            print(f"[RegionExtractor] ERROR: {str(e)}")
            mask = torch.zeros((H, W), dtype=torch.float32)
            found = False

        # Применяем размытие
        if mask_blur_radius > 0 and found:
            mask = gaussian_blur_mask(mask, mask_blur_radius)

        # Формируем отладочное изображение
        debug_image = blend_red_mask(original_image, mask, alpha=0.5)

        # Возвращаем в формате ComfyUI
        return (
            original_image.unsqueeze(0) if len(image.shape) == 4 else original_image,
            mask.unsqueeze(0) if len(image.shape) == 4 else mask,
            debug_image.unsqueeze(0) if len(image.shape) == 4 else debug_image,
            found,
        )


# Регистрация ноды
NODE_CLASS_MAPPINGS = {"ImageRegionExtractorQwen": ImageRegionExtractorQwen}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ImageRegionExtractorQwen": "Image Region Extractor (Qwen)"
}
