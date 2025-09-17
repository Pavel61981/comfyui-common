import torch
import numpy as np
from scipy.ndimage import gaussian_filter


class MaskBlur:
    """
    Эта нода применяет размытие по Гауссу (Gaussian Blur) к входной маске.
    Она позволяет смягчить края маски, создавая более плавные переходы.
    Вход: маска и радиус размытия.
    Выход: размытая маска.
    """

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "mask": ("MASK",),
                "blur_radius": (
                    "FLOAT",
                    {
                        "default": 10.0,
                        "min": 0.0,
                        "max": 1000.0,  # Увеличен максимальный радиус для большей гибкости
                        "step": 0.1,
                        "display": "number",
                    },
                ),
            },
        }

    RETURN_TYPES = ("MASK",)
    FUNCTION = "blur"

    CATEGORY = "mask"

    def blur(self, mask, blur_radius):
        if blur_radius == 0:
            return (mask,)  # Если размытие не требуется, возвращаем исходную маску

        # Убедимся, что маска имеет правильную размерность (батч, высота, ширина)
        if mask.dim() == 2:
            mask = mask.unsqueeze(0)

        # Клонируем выходной тензор, чтобы избежать изменения исходных данных
        blurred_mask = mask.clone()

        # Обрабатываем каждую маску в батче
        for i in range(mask.shape[0]):
            mask_np = mask[i].cpu().numpy()
            # Применяем фильтр Гаусса с помощью SciPy
            blurred_mask_np = gaussian_filter(mask_np, sigma=blur_radius)
            # Конвертируем обратно в тензор PyTorch и размещаем на том же устройстве, что и исходная маска
            blurred_mask[i] = torch.from_numpy(blurred_mask_np).to(mask.device)

        return (blurred_mask,)


# --- Регистрация ноды в ComfyUI ---
NODE_CLASS_MAPPINGS = {"MaskBlur": MaskBlur}

NODE_DISPLAY_NAME_MAPPINGS = {"MaskBlur": "Mask Blur (RU)"}
