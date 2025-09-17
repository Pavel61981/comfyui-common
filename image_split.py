import torch
import numpy as np
from PIL import Image

class ImageSplit:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
            },
            "optional": {
                "sensitivity": ("FLOAT", {"default": 0.1, "min": 0.01, "max": 1.0, "step": 0.01}),
                "min_width_ratio": ("FLOAT", {"default": 0.1, "min": 0.05, "max": 0.45, "step": 0.01}),
            }
        }

    RETURN_TYPES = ("IMAGE", "IMAGE")
    RETURN_NAMES = ("left_image", "right_image")
    FUNCTION = "split"
    CATEGORY = "image"

    def split(self, image, sensitivity=0.1, min_width_ratio=0.1):
        """
        Разделяет изображение на две части по горизонтали.
        Ищет место с максимальным градиентом по вертикали (границу между изображениями).
        Если не найдено — делит по центру.
        """
        batch_size, height, width, channels = image.shape
        results_left = []
        results_right = []

        for i in range(batch_size):
            img = image[i]  # H, W, C

            # Конвертируем в numpy для удобства
            img_np = img.cpu().numpy()

            # Минимальная ширина каждой части (в пикселях)
            min_width_px = int(width * min_width_ratio)

            # Вычисляем разницу между соседними столбцами (градиент по X)
            # Форма: (H, W-1)
            diff = np.abs(img_np[:, 1:, :] - img_np[:, :-1, :])  # разница между соседними столбцами
            mean_diff_per_col = np.mean(diff, axis=(0, 2))       # усредняем по высоте и каналам → (W-1,)

            # Ищем кандидатов на границу — где градиент выше sensitivity * max
            threshold = sensitivity * np.max(mean_diff_per_col)
            candidate_indices = np.where(mean_diff_per_col > threshold)[0]

            best_split = width // 2  # fallback — центр
            best_score = -1

            # Проверяем каждого кандидата — насколько "чист" разрез по всей высоте
            for x in candidate_indices:
                # Оценим "резкость" перехода в этом столбце по всей высоте
                col_diff = diff[:, x, :]  # H, C
                score = np.mean(col_diff)  # чем выше — тем резче переход
                if score > best_score:
                    # Проверим, что слева и справа достаточно места
                    if x + 1 >= min_width_px and (width - (x + 1)) >= min_width_px:
                        best_score = score
                        best_split = x + 1  # потому что diff на 1 короче — это граница между x и x+1

            # Если не нашли подходящую границу — делим по центру
            if best_score == -1:
                best_split = width // 2

            # Делим изображение
            left_part = img[:, :best_split, :]
            right_part = img[:, best_split:, :]

            results_left.append(left_part)
            results_right.append(right_part)

        # Собираем обратно в батч
        left_batch = torch.stack(results_left, dim=0)
        right_batch = torch.stack(results_right, dim=0)

        return (left_batch, right_batch)


# Регистрация ноды
NODE_CLASS_MAPPINGS = {
    "ImageSplit": ImageSplit
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ImageSplit": "Image split"
}
