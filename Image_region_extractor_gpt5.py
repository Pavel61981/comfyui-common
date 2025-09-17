import torch
import numpy as np
import cv2
import mediapipe as mp
import warnings
from typing import Optional

# ==================================
# Константы для MediaPipe
# ==================================
# Индексы точек для получения контура овала лица
# (для MediaPipe FaceMesh; проверено для текущей версии)
FACE_OVAL_INDICES = [
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
    377,
    152,
    148,
    176,
    149,
    150,
    136,
    172,
    58,
    132,
    93,
    234,
    127,
    162,
    21,
    54,
    103,
    67,
    109,
]


class ImageRegionExtractorGpt5:
    """
    Кастомная нода для ComfyUI, которая экстрагирует маску указанного
    региона (лицо, тело) из изображения, используя MediaPipe для
    эффективной и точной детекции.
    """

    mp_face_mesh = mp.solutions.face_mesh
    mp_selfie_segmentation = mp.solutions.selfie_segmentation

    # Кеш моделей для повышения производительности
    _cached_face_mesh = None
    _cached_selfie_segmentation = None

    @classmethod
    def INPUT_TYPES(s):
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
                    {"default": 0.5, "min": 0.1, "max": 1.0, "step": 0.05},
                ),
                "mask_blur_radius": (
                    "INT",
                    {"default": 5, "min": 0, "max": 50, "step": 1},
                ),
                "reuse_models": ("BOOLEAN", {"default": True}),
                "return_masked_image": ("BOOLEAN", {"default": False}),
            },
        }

    RETURN_TYPES = ("IMAGE", "MASK", "IMAGE", "BOOLEAN")
    RETURN_NAMES = ("IMAGE", "MASK", "DEBUG_IMAGE", "FOUND")
    FUNCTION = "extract"
    CATEGORY = "Masquerade"

    # ------------------------------
    # Вспомогательные функции
    # ------------------------------

    def tensor_to_np(self, tensor: torch.Tensor) -> np.ndarray:
        """Конвертирует тензор [B,H,W,C] (0..1 float) в numpy [H,W,C] (uint8)."""
        if not isinstance(tensor, torch.Tensor):
            raise TypeError("image must be a torch.Tensor")
        arr = tensor.detach().cpu().numpy()
        if arr.ndim == 4:
            arr = arr[0]
        elif arr.ndim != 3:
            raise ValueError(f"Unexpected tensor shape: {arr.shape}")
        if arr.dtype == np.uint8:
            return arr
        arr = np.clip(arr, 0.0, 1.0)
        return (arr * 255.0).astype(np.uint8)

    def np_to_tensor(self, array: np.ndarray) -> torch.Tensor:
        """Конвертирует numpy [H,W,C] (uint8) в тензор [1,H,W,C] (0..1 float32)."""
        a = array.astype(np.float32) / 255.0
        return torch.from_numpy(a).unsqueeze(0)

    # ------------------------------
    # Маска лица
    # ------------------------------
    def _get_face_mask(
        self, image_np: np.ndarray, threshold: float, reuse_models: bool = True
    ) -> Optional[np.ndarray]:
        h, w, _ = image_np.shape
        mask = np.zeros((h, w), dtype=np.uint8)
        found = False

        try:
            if reuse_models and self._cached_face_mesh is None:
                self._cached_face_mesh = self.mp_face_mesh.FaceMesh(
                    static_image_mode=True,
                    max_num_faces=10,
                    min_detection_confidence=threshold,
                )

            face_mesh = (
                self._cached_face_mesh
                if reuse_models
                else self.mp_face_mesh.FaceMesh(
                    static_image_mode=True,
                    max_num_faces=10,
                    min_detection_confidence=threshold,
                )
            )

            results = face_mesh.process(image_np)
            if not results or not getattr(results, "multi_face_landmarks", None):
                return None

            for face_landmarks in results.multi_face_landmarks:
                points = []
                for index in FACE_OVAL_INDICES:
                    if index >= len(face_landmarks.landmark):
                        continue
                    lm = face_landmarks.landmark[index]
                    points.append([int(lm.x * w), int(lm.y * h)])

                if len(points) >= 3:
                    convex_hull = cv2.convexHull(np.array(points))
                    cv2.fillConvexPoly(mask, convex_hull, 255)
                    found = True

            return mask if found else None

        except Exception as e:
            warnings.warn(f"Face mesh failed: {e}")
            return None
        finally:
            if not reuse_models and "face_mesh" in locals():
                face_mesh.close()

    # ------------------------------
    # Маска тела
    # ------------------------------
    def _get_body_mask(
        self, image_np: np.ndarray, threshold: float, reuse_models: bool = True
    ) -> Optional[np.ndarray]:
        try:
            if reuse_models and self._cached_selfie_segmentation is None:
                self._cached_selfie_segmentation = (
                    self.mp_selfie_segmentation.SelfieSegmentation(model_selection=0)
                )

            seg = (
                self._cached_selfie_segmentation
                if reuse_models
                else self.mp_selfie_segmentation.SelfieSegmentation(model_selection=0)
            )
            results = seg.process(image_np)
            if not results or getattr(results, "segmentation_mask", None) is None:
                return None

            mask = (results.segmentation_mask > threshold).astype(np.uint8) * 255
            return mask if np.sum(mask) > 10 else None
        except Exception as e:
            warnings.warn(f"Selfie segmentation failed: {e}")
            return None
        finally:
            if not reuse_models and "seg" in locals():
                seg.close()

    # ------------------------------
    # Основная функция
    # ------------------------------
    def extract(
        self,
        image: torch.Tensor,
        target_region: str,
        confidence_threshold: float,
        mask_blur_radius: int,
        reuse_models: bool = True,
        return_masked_image: bool = False,
    ):
        image_np = self.tensor_to_np(image)
        h, w, _ = image_np.shape
        raw_mask = None

        if target_region == "face":
            raw_mask = self._get_face_mask(image_np, confidence_threshold, reuse_models)
        elif target_region == "body":
            raw_mask = self._get_body_mask(image_np, confidence_threshold, reuse_models)
        elif target_region == "body_without_face":
            body_mask = self._get_body_mask(
                image_np, confidence_threshold, reuse_models
            )
            face_mask = self._get_face_mask(
                image_np, confidence_threshold, reuse_models
            )
            if body_mask is not None:
                raw_mask = np.clip(
                    body_mask.astype(np.int16)
                    - (face_mask.astype(np.int16) if face_mask is not None else 0),
                    0,
                    255,
                ).astype(np.uint8)
        else:
            raise ValueError(
                f"Неподдерживаемое значение target_region: {target_region}"
            )

        found = raw_mask is not None and np.any(raw_mask)
        final_mask_np = raw_mask if found else np.zeros((h, w), dtype=np.uint8)

        if mask_blur_radius > 0:
            radius = mask_blur_radius * 2 + 1
            max_k = min(radius, min(h // 2 * 2 + 1, w // 2 * 2 + 1))
            final_mask_np = cv2.GaussianBlur(final_mask_np, (max_k, max_k), 0)

        # Debug overlay
        debug_image_np = image_np.copy()
        red_overlay = np.zeros_like(debug_image_np)
        red_overlay[final_mask_np > 0] = [255, 0, 0]
        debug_image_np = cv2.addWeighted(red_overlay, 0.5, debug_image_np, 0.5, 0)

        # Маска в формате [1,H,W,1] float32
        out_mask = final_mask_np.astype(np.float32) / 255.0
        out_mask = out_mask[np.newaxis, ..., np.newaxis]
        output_mask_tensor = torch.from_numpy(out_mask)

        # Выбор, что вернуть как IMAGE
        if return_masked_image:
            masked_np = image_np.copy()
            masked_np[final_mask_np == 0] = 0
            out_image_tensor = self.np_to_tensor(masked_np)
        else:
            out_image_tensor = image

        debug_image_tensor = self.np_to_tensor(debug_image_np)

        return (out_image_tensor, output_mask_tensor, debug_image_tensor, bool(found))

    # Опционально — закрыть кешированные модели
    def close_cached_models(self):
        if self._cached_face_mesh is not None:
            try:
                self._cached_face_mesh.close()
            except Exception:
                pass
            self._cached_face_mesh = None
        if self._cached_selfie_segmentation is not None:
            try:
                self._cached_selfie_segmentation.close()
            except Exception:
                pass
            self._cached_selfie_segmentation = None


# ==================================
# Регистрация ноды в ComfyUI
# ==================================
NODE_CLASS_MAPPINGS = {"ImageRegionExtractorGpt5": ImageRegionExtractorGpt5}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ImageRegionExtractorGpt5": "Image Region Extractor (Gpt5)"
}
