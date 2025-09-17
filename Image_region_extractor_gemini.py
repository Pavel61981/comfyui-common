import torch
import numpy as np
import cv2
import mediapipe as mp

# ==================================
# Константы для MediaPipe
# ==================================
# Индексы точек для получения контура овала лица
# Можно найти в документации MediaPipe или open-source проектах
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


class ImageRegionExtractorGemini:
    """
    Кастомная нода для ComfyUI, которая экстрагирует маску указанного
    региона (лицо, тело) из изображения, используя MediaPipe для
    эффективной и точной детекции.
    """

    # Инициализация моделей MediaPipe один раз для эффективности
    # Это позволяет не загружать модели при каждом запуске ноды
    mp_face_mesh = mp.solutions.face_mesh
    mp_selfie_segmentation = mp.solutions.selfie_segmentation

    # Мы будем создавать экземпляры внутри функции, чтобы обеспечить потокобезопасность
    # и корректное управление ресурсами с помощью 'with'

    @classmethod
    def INPUT_TYPES(s):
        """Определение входных параметров для интерфейса ComfyUI."""
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
            },
        }

    RETURN_TYPES = ("IMAGE", "MASK", "IMAGE", "BOOLEAN")
    RETURN_NAMES = ("IMAGE", "MASK", "DEBUG_IMAGE", "FOUND")
    FUNCTION = "extract"
    CATEGORY = "Masquerade"  # Или любая другая категория на ваш выбор

    def tensor_to_np(self, tensor: torch.Tensor) -> np.ndarray:
        """Конвертирует тензор [B, H, W, C] (0-1, float) в NumPy массив [H, W, C] (0-255, uint8)."""
        # Убираем batch-измерение, конвертируем в NumPy, масштабируем до 0-255 и меняем тип
        return (tensor[0].numpy() * 255).astype(np.uint8)

    def np_to_tensor(self, array: np.ndarray) -> torch.Tensor:
        """Конвертирует NumPy массив [H, W, C] (0-255, uint8) в тензор [1, H, W, C] (0-1, float)."""
        # Масштабируем до 0-1, конвертируем в тензор и добавляем batch-измерение
        return torch.from_numpy(array / 255.0).float().unsqueeze(0)

    def _get_face_mask(
        self, image_np: np.ndarray, threshold: float
    ) -> np.ndarray | None:
        """Создает маску лица с помощью Face Mesh."""
        h, w, _ = image_np.shape
        mask = np.zeros((h, w), dtype=np.uint8)
        found_at_least_one = False

        with self.mp_face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=10,  # Ищем до 10 лиц
            min_detection_confidence=threshold,
        ) as face_mesh:

            # MediaPipe ожидает RGB, ComfyUI предоставляет RGB
            results = face_mesh.process(image_np)

            if not results.multi_face_landmarks:
                return None

            for face_landmarks in results.multi_face_landmarks:
                found_at_least_one = True
                points = []
                for index in FACE_OVAL_INDICES:
                    lm = face_landmarks.landmark[index]
                    # Конвертируем нормализованные координаты в пиксельные
                    points.append([int(lm.x * w), int(lm.y * h)])

                # Создаем выпуклую оболочку для гладкого контура
                convex_hull = cv2.convexHull(np.array(points))
                # Заполняем область внутри контура на маске
                cv2.fillConvexPoly(mask, convex_hull, 255)

        return mask if found_at_least_one else None

    def _get_body_mask(
        self, image_np: np.ndarray, threshold: float
    ) -> np.ndarray | None:
        """Создает маску тела с помощью Selfie Segmentation."""
        with self.mp_selfie_segmentation.SelfieSegmentation(
            model_selection=0
        ) as selfie_segmentation:  # model=0 для общего случая

            results = selfie_segmentation.process(image_np)

            # Создаем бинарную маску на основе порога уверенности
            mask = (results.segmentation_mask > threshold).astype(np.uint8) * 255

            # Проверяем, найдено ли что-то существенное
            if np.sum(mask) > 10:  # Простое условие, чтобы отсечь шум
                return mask
            else:
                return None

    def extract(
        self,
        image: torch.Tensor,
        target_region: str,
        confidence_threshold: float,
        mask_blur_radius: int,
    ):
        """Основная логика работы ноды."""

        # 1. Конвертация и подготовка
        image_np = self.tensor_to_np(image)
        h, w, _ = image_np.shape
        raw_mask = None

        # 2. Детекция региона и создание "сырой" маски
        if target_region == "face":
            raw_mask = self._get_face_mask(image_np, confidence_threshold)

        elif target_region == "body":
            raw_mask = self._get_body_mask(image_np, confidence_threshold)

        elif target_region == "body_without_face":
            body_mask = self._get_body_mask(image_np, confidence_threshold)
            face_mask = self._get_face_mask(image_np, confidence_threshold)

            if body_mask is not None:
                if face_mask is not None:
                    # Вычитаем маску лица из маски тела
                    # np.clip гарантирует, что значения останутся в пределах 0-255
                    raw_mask = np.clip(
                        body_mask.astype(np.int16) - face_mask.astype(np.int16), 0, 255
                    ).astype(np.uint8)
                else:
                    raw_mask = body_mask  # Если лицо не найдено, возвращаем маску тела

        else:
            raise ValueError(
                f"Неподдерживаемое значение target_region: {target_region}"
            )

        # 3. Обработка результата
        found = raw_mask is not None and np.any(raw_mask)

        if not found:
            # Если ничего не найдено, создаем пустую (черную) маску
            final_mask_np = np.zeros((h, w), dtype=np.uint8)
        else:
            final_mask_np = raw_mask

        # 4. Размытие маски
        if mask_blur_radius > 0:
            # Радиус должен быть нечетным
            radius = mask_blur_radius * 2 + 1
            final_mask_np = cv2.GaussianBlur(final_mask_np, (radius, radius), 0)

        # 5. Создание отладочного изображения
        debug_image_np = image_np.copy()
        # Создаем красный слой с альфа-каналом, соответствующим маске
        red_overlay = np.zeros_like(debug_image_np)
        # Устанавливаем красный цвет там, где маска не черная
        red_overlay[final_mask_np > 0] = [255, 0, 0]

        # Накладываем красный слой с прозрачностью 0.5
        alpha = 0.5
        debug_image_np = cv2.addWeighted(
            red_overlay, alpha, debug_image_np, 1 - alpha, 0
        )

        # 6. Конвертация результатов обратно в тензоры
        # Маска должна быть [H, W] и нормализована до 0-1
        output_mask = torch.from_numpy(final_mask_np / 255.0).float()

        # Отладочное изображение конвертируем обратно в формат ComfyUI
        debug_image_tensor = self.np_to_tensor(debug_image_np)

        # Возвращаем кортеж в соответствии с RETURN_TYPES
        return (image, output_mask, debug_image_tensor, found)


# ==================================
# Регистрация ноды в ComfyUI
# ==================================
NODE_CLASS_MAPPINGS = {"ImageRegionExtractorGemini": ImageRegionExtractorGemini}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ImageRegionExtractorGemini": "Image Region Extractor (Gemini)"
}
