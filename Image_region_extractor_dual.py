import torch
import numpy as np
import cv2
import mediapipe as mp
import warnings
from typing import Optional

# ==================================
# Константы для MediaPipe
# ==================================
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

# Основные landmarks для отладки: глаза, брови, нос, губы, овал
FACIAL_LANDMARKS_DEBUG_GROUPS = {
    "face_oval": [
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
    ],
    "left_eye": [
        33,
        7,
        163,
        144,
        145,
        153,
        154,
        155,
        133,
        173,
        157,
        158,
        159,
        160,
        161,
        246,
    ],
    "right_eye": [
        362,
        382,
        381,
        380,
        374,
        373,
        390,
        249,
        263,
        466,
        388,
        387,
        386,
        385,
        384,
        398,
    ],
    "left_eyebrow": [70, 63, 105, 66, 107, 55, 65, 52, 53, 46],
    "right_eyebrow": [336, 296, 334, 293, 300, 276, 283, 282, 295, 285],
    "nose": [1, 2, 98, 327, 4, 5, 195, 197, 6, 168, 8, 193, 194],
    "lips": [
        61,
        185,
        40,
        39,
        37,
        0,
        267,
        269,
        270,
        409,
        291,
        375,
        321,
        405,
        314,
        17,
        84,
        181,
        91,
        146,
        61,  # нижняя + замкнутая
    ],
}


class ImageRegionExtractorDual:
    """
    Нода ComfyUI — возвращает:
      - IMAGE (исходное изображение)
      - MASK (маска лица)  -> shape [1,H,W], float32 0..1
      - MASK (маска тела без лица) -> shape [1,H,W], float32 0..1
      - IMAGE (debug image) (если enable_debug=False, возвращается исходное изображение)
      - BOOLEAN (found_face)
      - BOOLEAN (found_body_without_face)

    Поведение детекции лиц:
      1) Сначала пробуем только FaceMesh — если он вернул хотя бы одну точную маску,
         возвращаем **только** её (приоритет точного овала).
      2) Если FaceMesh не дал результата, то тогда применяется fallback:
         FaceDetection → bbox → ROI → FaceMesh (гибридный подход)
      3) Если и это не сработало — создаём маску на основе bbox (эллипс + размытие)

    В debug-режиме: если включено — отображаются КЛЮЧЕВЫЕ точки лица (глаза, нос, губы, брови, овал),
    даже если маска не построена — для отладки сложных случаев.
    """

    mp_face_mesh = mp.solutions.face_mesh
    mp_selfie_segmentation = mp.solutions.selfie_segmentation
    mp_face_detection = mp.solutions.face_detection

    # Простой кеш моделей для переиспользования
    _cached_face_mesh = None
    _cached_selfie_segmentation = None
    _cached_face_detection = None

    FACIAL_LANDMARKS_DEBUG_GROUPS = (
        FACIAL_LANDMARKS_DEBUG_GROUPS  # ← доступ внутри класса
    )

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
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
                "enable_debug": ("BOOLEAN", {"default": True}),
                "face_detection_fallback": ("BOOLEAN", {"default": True}),
                "face_detection_confidence": (
                    "FLOAT",
                    {"default": 0.35, "min": 0.1, "max": 1.0, "step": 0.05},
                ),
                "debug_show_all_landmarks": (
                    "BOOLEAN",
                    {"default": True},
                ),  # ← добавлено
            },
        }

    RETURN_TYPES = ("IMAGE", "MASK", "MASK", "IMAGE", "BOOLEAN", "BOOLEAN")
    RETURN_NAMES = (
        "IMAGE",
        "FACE_MASK",
        "BODY_NO_FACE_MASK",
        "DEBUG_IMAGE",
        "FOUND_FACE",
        "FOUND_BODY_NO_FACE",
    )
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

    def _postprocess_mask(self, mask: np.ndarray, kernel_size: int = 5) -> np.ndarray:
        """Заполнение дыр и небольшое расширение маски."""
        if mask is None or not np.any(mask):
            return mask
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (kernel_size, kernel_size)
        )
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)  # Заполнить дыры
        mask = cv2.morphologyEx(mask, cv2.MORPH_DILATE, kernel)  # Чуть расширить
        return mask

    # ------------------------------
    # FaceMesh -> точечная маска (если possible)
    # ------------------------------
    def _face_mesh_mask_from_mesh(
        self, face_landmarks, w: int, h: int
    ) -> Optional[np.ndarray]:
        """
        Построить маску из face_landmarks.
        - Если точек 3 и более -> создается выпуклая оболочка (convex hull).
        - Если точек 1 или 2 -> создаются круги/линии для представления найденной части.
        """
        mask = np.zeros((h, w), dtype=np.uint8)
        points = []
        for index in FACE_OVAL_INDICES:
            if index < len(face_landmarks.landmark):
                lm = face_landmarks.landmark[index]
                if not (np.isfinite(lm.x) and np.isfinite(lm.y)):
                    continue
                px = int(lm.x * w)
                py = int(lm.y * h)
                if 0 <= px < w and 0 <= py < h:
                    points.append([px, py])

        if len(points) >= 3:
            hull = cv2.convexHull(np.array(points))
            cv2.fillConvexPoly(mask, hull, 255)
            return mask
        elif len(points) == 2:
            p1 = tuple(points[0])
            p2 = tuple(points[1])
            distance = np.linalg.norm(np.array(p1) - np.array(p2))
            thickness = max(10, int(distance * 0.5))
            cv2.line(mask, p1, p2, color=255, thickness=thickness, lineType=cv2.LINE_AA)
            return mask
        elif len(points) == 1:
            p1 = tuple(points[0])
            radius = int(w * 0.05)
            cv2.circle(mask, p1, radius, color=255, thickness=-1)
            return mask

        return None

    def _get_face_mask_from_mesh_only(
        self, image_np: np.ndarray, threshold: float, reuse_models: bool = True
    ) -> Optional[np.ndarray]:
        """Запускает только FaceMesh на изображении, возвращает объединённую маску или None."""
        h, w, _ = image_np.shape
        try:
            if reuse_models and self._cached_face_mesh is None:
                self._cached_face_mesh = self.mp_face_mesh.FaceMesh(
                    static_image_mode=True,
                    max_num_faces=10,
                    refine_landmarks=True,  # ← улучшенные landmarks
                    min_detection_confidence=threshold,
                )

            face_mesh = (
                self._cached_face_mesh
                if reuse_models
                else self.mp_face_mesh.FaceMesh(
                    static_image_mode=True,
                    max_num_faces=10,
                    refine_landmarks=True,
                    min_detection_confidence=threshold,
                )
            )

            results = face_mesh.process(image_np)
            if results and getattr(results, "multi_face_landmarks", None):
                union_mask_mesh = np.zeros((h, w), dtype=np.uint8)
                found_mesh = False
                for face_landmarks in results.multi_face_landmarks:
                    m = self._face_mesh_mask_from_mesh(face_landmarks, w, h)
                    if m is not None:
                        union_mask_mesh = np.maximum(union_mask_mesh, m)
                        found_mesh = True
                if found_mesh:
                    return union_mask_mesh
        except Exception as e:
            warnings.warn(f"Face mesh step failed: {e}")
        finally:
            if not reuse_models and "face_mesh" in locals():
                try:
                    face_mesh.close()
                except Exception:
                    pass
        return None

    # ------------------------------
    # FaceDetection → ROI → FaceMesh (гибридный подход)
    # ------------------------------
    def _get_face_detection_bbox_only(
        self, image_np: np.ndarray, confidence: float, reuse_models: bool = True
    ) -> Optional[np.ndarray]:
        """Возвращает объединённую bbox-маску из FaceDetection (без эллипса, просто прямоугольники)."""
        h, w, _ = image_np.shape
        try:
            if reuse_models and self._cached_face_detection is None:
                self._cached_face_detection = self.mp_face_detection.FaceDetection(
                    min_detection_confidence=confidence
                )

            detector = (
                self._cached_face_detection
                if reuse_models
                else self.mp_face_detection.FaceDetection(
                    min_detection_confidence=confidence
                )
            )

            det_results = detector.process(image_np)
            if det_results and getattr(det_results, "detections", None):
                union_mask = np.zeros((h, w), dtype=np.uint8)
                for detection in det_results.detections:
                    loc = detection.location_data
                    if not loc or not getattr(loc, "relative_bounding_box", None):
                        continue
                    bbox = loc.relative_bounding_box
                    xmin = int(bbox.xmin * w)
                    ymin = int(bbox.ymin * h)
                    bw = int(bbox.width * w)
                    bh = int(bbox.height * h)
                    x1 = max(0, xmin)
                    y1 = max(0, ymin)
                    x2 = min(w, xmin + bw)
                    y2 = min(h, ymin + bh)
                    if x2 > x1 and y2 > y1:
                        union_mask[y1:y2, x1:x2] = 255
                if np.any(union_mask):
                    return union_mask
        except Exception as e:
            warnings.warn(f"Face detection bbox extraction failed: {e}")
        finally:
            if not reuse_models and "detector" in locals():
                try:
                    detector.close()
                except Exception:
                    pass
        return None

    def _hybrid_detection_fallback(
        self,
        image_np: np.ndarray,
        threshold: float,
        face_detection_confidence: float,
        reuse_models: bool = True,
    ) -> Optional[np.ndarray]:
        """
        Гибридный подход:
          1. Получить bbox из FaceDetection.
          2. Для каждого bbox — вырезать ROI (с запасом 25%).
          3. Запустить FaceMesh на ROI.
          4. Собрать маску обратно в полный размер.
        """
        h, w, _ = image_np.shape
        bbox_mask = self._get_face_detection_bbox_only(
            image_np, face_detection_confidence, reuse_models
        )
        if bbox_mask is None:
            return None

        coords = cv2.findNonZero(bbox_mask)
        if coords is None:
            return None

        # Получаем общий bounding box всех лиц (можно и по отдельности, но так проще)
        x, y, bw, bh = cv2.boundingRect(coords)

        # Расширяем ROI на 25% с каждой стороны
        margin_x = int(bw * 0.25)
        margin_y = int(bh * 0.25)
        x1 = max(0, x - margin_x)
        y1 = max(0, y - margin_y)
        x2 = min(w, x + bw + margin_x)
        y2 = min(h, y + bh + margin_y)

        if x2 <= x1 or y2 <= y1:
            return None

        roi = image_np[y1:y2, x1:x2]
        roi_mask = self._get_face_mask_from_mesh_only(roi, threshold, reuse_models)

        if roi_mask is not None:
            full_mask = np.zeros((h, w), dtype=np.uint8)
            full_mask[y1:y2, x1:x2] = roi_mask
            return full_mask

        return None

    # ------------------------------
    # FaceDetection → эллипс маска (fallback последней инстанции)
    # ------------------------------
    def _mask_from_detection_bbox(
        self, bbox, w: int, h: int, expand: float = 1.3
    ) -> np.ndarray:
        """
        Создаёт ЭЛЛИПС внутри расширенного bbox (не прямоугольник!).
        expand: расширение bbox для захвата волос/ушей/подбородка.
        После рисования эллипса — применяется размытие и морфология для мягкого расширения.
        """
        # Расширяем bbox для лучшего покрытия
        cx = (bbox.xmin + bbox.width / 2.0) * w
        cy = (bbox.ymin + bbox.height / 2.0) * h
        half_w = (bbox.width * w / 2.0) * expand
        half_h = (bbox.height * h / 2.0) * expand

        # Ограничиваем размеры, чтобы не выйти за границы
        half_w = max(1.0, min(half_w, w / 2.0))
        half_h = max(1.0, min(half_h, h / 2.0))

        mask = np.zeros((h, w), dtype=np.uint8)
        center = (int(round(cx)), int(round(cy)))
        axes = (int(round(half_w)), int(round(half_h)))

        # Рисуем ЭЛЛИПС (не прямоугольник!)
        cv2.ellipse(
            mask,
            center,
            axes,
            angle=0,
            startAngle=0,
            endAngle=360,
            color=255,
            thickness=-1,
        )

        # Мягкое расширение: размытие + морфология
        k = max(5, int(min(half_w, half_h) * 0.2) // 2 * 2 + 1)
        mask = cv2.GaussianBlur(mask, (k, k), 0)
        kernel_size = max(3, int(k * 0.5) // 2 * 2 + 1)
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (kernel_size, kernel_size)
        )
        mask = cv2.dilate(mask, kernel)

        return mask

    def _get_face_mask_fallback_old(
        self,
        image_np: np.ndarray,
        face_detection_confidence: float,
        reuse_models: bool = True,
    ) -> Optional[np.ndarray]:
        """Последний fallback — маска на основе bbox с размытием."""
        h, w, _ = image_np.shape
        try:
            if reuse_models and self._cached_face_detection is None:
                self._cached_face_detection = self.mp_face_detection.FaceDetection(
                    min_detection_confidence=face_detection_confidence
                )

            detector = (
                self._cached_face_detection
                if reuse_models
                else self.mp_face_detection.FaceDetection(
                    min_detection_confidence=face_detection_confidence
                )
            )

            det_results = detector.process(image_np)
            if det_results and getattr(det_results, "detections", None):
                union_mask_det = np.zeros((h, w), dtype=np.uint8)
                for detection in det_results.detections:
                    loc = detection.location_data
                    if not loc or not getattr(loc, "relative_bounding_box", None):
                        continue
                    bbox = loc.relative_bounding_box
                    m = self._mask_from_detection_bbox(bbox, w, h, expand=1.3)
                    union_mask_det = np.maximum(union_mask_det, m)
                if np.any(union_mask_det):
                    return union_mask_det
        except Exception as e:
            warnings.warn(f"Face detection fallback failed: {e}")
        finally:
            if not reuse_models and "detector" in locals():
                try:
                    detector.close()
                except Exception:
                    pass
        return None

    # ------------------------------
    # Маска лица: приоритет FaceMesh → гибридный подход → bbox fallback
    # ------------------------------
    def _get_face_mask(
        self,
        image_np: np.ndarray,
        threshold: float,
        reuse_models: bool = True,
        fallback: bool = True,
        face_detection_confidence: float = 0.35,
    ) -> Optional[np.ndarray]:
        """
        Сначала пробуем только FaceMesh и возвращаем её результат, если он есть.
        Если FaceMesh вернул None => (и fallback=True) используем FaceDetection → ROI → FaceMesh.
        Если и это не сработало — fallback на bbox-маску (эллипс).
        """
        # --- Попытка FaceMesh (точный овал) ---
        face_mask = self._get_face_mask_from_mesh_only(
            image_np, threshold, reuse_models
        )
        if face_mask is not None:
            return self._postprocess_mask(face_mask, kernel_size=5)

        if not fallback:
            return None

        # --- Гибридный подход: FaceDetection → ROI → FaceMesh ---
        hybrid_mask = self._hybrid_detection_fallback(
            image_np, threshold, face_detection_confidence, reuse_models
        )
        if hybrid_mask is not None:
            return self._postprocess_mask(hybrid_mask, kernel_size=5)

        # --- Последний fallback: эллипс + размытие ---
        fallback_mask = self._get_face_mask_fallback_old(
            image_np, face_detection_confidence, reuse_models
        )
        if fallback_mask is not None:
            return self._postprocess_mask(fallback_mask, kernel_size=7)

        return None

    # ------------------------------
    # Маска тела (как раньше)
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
                try:
                    seg.close()
                except Exception:
                    pass

    # ------------------------------
    # Визуализация ключевых точек лица (глаза, нос, губы, брови, овал)
    # ------------------------------
    def _draw_key_face_landmarks(
        self, image_np: np.ndarray, face_landmarks_list
    ) -> np.ndarray:
        """
        Рисует только КЛЮЧЕВЫЕ точки лица: глаза, брови, нос, губы, овал.
        Рисует разными цветами для удобства.
        """
        if not face_landmarks_list:
            return image_np

        h, w = image_np.shape[:2]
        output = image_np.copy()

        # Цвета для групп
        COLORS = {
            "face_oval": (255, 255, 255),  # белый
            "left_eye": (0, 255, 0),  # зелёный
            "right_eye": (0, 255, 0),  # зелёный
            "left_eyebrow": (255, 0, 255),  # пурпурный
            "right_eyebrow": (255, 0, 255),  # пурпурный
            "nose": (0, 165, 255),  # оранжевый
            "lips": (0, 0, 255),  # красный
        }

        for face_landmarks in face_landmarks_list:
            for group_name, indices in self.FACIAL_LANDMARKS_DEBUG_GROUPS.items():
                color = COLORS.get(group_name, (255, 255, 255))
                for idx in indices:
                    if idx >= len(face_landmarks.landmark):
                        continue
                    lm = face_landmarks.landmark[idx]
                    if not (np.isfinite(lm.x) and np.isfinite(lm.y)):
                        continue
                    px = int(lm.x * w)
                    py = int(lm.y * h)
                    if 0 <= px < w and 0 <= py < h:
                        cv2.circle(
                            output, (px, py), radius=1, color=color, thickness=-1
                        )

        return output

    # ------------------------------
    # Основной метод
    # ------------------------------
    def extract(
        self,
        image: torch.Tensor,
        confidence_threshold: float = 0.5,
        mask_blur_radius: int = 5,
        reuse_models: bool = True,
        enable_debug: bool = True,
        face_detection_fallback: bool = True,
        face_detection_confidence: float = 0.35,
        debug_show_all_landmarks: bool = True,  # ← добавлено
    ):
        image_np = self.tensor_to_np(image)
        h, w, _ = image_np.shape

        face_mask = self._get_face_mask(
            image_np,
            confidence_threshold,
            reuse_models,
            fallback=face_detection_fallback,
            face_detection_confidence=face_detection_confidence,
        )
        body_mask = self._get_body_mask(image_np, confidence_threshold, reuse_models)

        if body_mask is not None:
            if face_mask is not None:
                body_no_face = np.clip(
                    body_mask.astype(np.int16) - face_mask.astype(np.int16), 0, 255
                ).astype(np.uint8)
            else:
                body_no_face = body_mask.copy()
        else:
            body_no_face = None

        found_face = face_mask is not None and np.any(face_mask)
        found_body_no_face = body_no_face is not None and np.any(body_no_face)

        face_mask_final = face_mask if found_face else np.zeros((h, w), dtype=np.uint8)
        body_no_face_final = (
            body_no_face if found_body_no_face else np.zeros((h, w), dtype=np.uint8)
        )

        # Размытие масок (по желанию)
        if mask_blur_radius > 0:
            ksize = mask_blur_radius * 2 + 1
            face_mask_final = cv2.GaussianBlur(face_mask_final, (ksize, ksize), 0)
            body_no_face_final = cv2.GaussianBlur(body_no_face_final, (ksize, ksize), 0)

        # Отладочное изображение
        if enable_debug:
            debug_image_np = image_np.copy()

            # --- Визуализация ключевых точек лица, если включено ---
            if debug_show_all_landmarks:
                try:
                    if reuse_models and self._cached_face_mesh is None:
                        self._cached_face_mesh = self.mp_face_mesh.FaceMesh(
                            static_image_mode=True,
                            max_num_faces=10,
                            refine_landmarks=True,
                            min_detection_confidence=confidence_threshold,
                        )

                    face_mesh_vis = (
                        self._cached_face_mesh
                        if reuse_models
                        else self.mp_face_mesh.FaceMesh(
                            static_image_mode=True,
                            max_num_faces=10,
                            refine_landmarks=True,
                            min_detection_confidence=confidence_threshold,
                        )
                    )

                    results_vis = face_mesh_vis.process(image_np)
                    if results_vis and results_vis.multi_face_landmarks:
                        # ← ВАЖНО: рисуем landmarks даже если face_mask is None!
                        debug_image_np = self._draw_key_face_landmarks(
                            debug_image_np, results_vis.multi_face_landmarks
                        )

                    if not reuse_models and "face_mesh_vis" in locals():
                        face_mesh_vis.close()

                except Exception as e:
                    warnings.warn(f"Failed to draw face landmarks for debug: {e}")

            # --- Отрисовка масок (лицо + тело) ---
            overlay = np.zeros_like(debug_image_np)
            overlay[face_mask_final > 0] = [255, 0, 0]  # красный для лица
            overlay[body_no_face_final > 0] = [0, 0, 255]  # синий для тела
            debug_image_np = cv2.addWeighted(overlay, 0.5, debug_image_np, 0.5, 0)

            debug_image_tensor = self.np_to_tensor(debug_image_np)
        else:
            debug_image_tensor = image

        # Приведение маски к форме [1,H,W] float32 0..1
        face_mask_out = (face_mask_final.astype(np.float32) / 255.0)[np.newaxis, ...]
        body_no_face_out = (body_no_face_final.astype(np.float32) / 255.0)[
            np.newaxis, ...
        ]

        face_mask_tensor = torch.from_numpy(face_mask_out)
        body_no_face_tensor = torch.from_numpy(body_no_face_out)

        return (
            image,
            face_mask_tensor,
            body_no_face_tensor,
            debug_image_tensor,
            bool(found_face),
            bool(found_body_no_face),
        )

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
        if self._cached_face_detection is not None:
            try:
                self._cached_face_detection.close()
            except Exception:
                pass
            self._cached_face_detection = None


# ==================================
# Регистрация ноды в ComfyUI
# ==================================
NODE_CLASS_MAPPINGS = {"ImageRegionExtractorDual": ImageRegionExtractorDual}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ImageRegionExtractorDual": "Image Region Extractor (Dual)"
}
