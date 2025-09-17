# scale_to_nearest_side_32.py

import torch
import torch.nn.functional as F
from PIL import Image
import numpy as np

class ImageScaleToNearestSide32:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),  # [B, H, W, C]
                "target_size": ("INT", {
                    "default": 1344,
                    "min": 32,
                    "max": 4096,
                    "step": 32,
                    "display": "slider"
                }),
            },
        }

    RETURN_TYPES = ("IMAGE", "INT", "INT", "IMAGE", "INT", "INT")
    RETURN_NAMES = ("original_image", "original_width", "original_height", "new_image", "new_width", "new_height")
    FUNCTION = "scale_to_nearest_side_32"
    CATEGORY = "Image Processing"

    def scale_to_nearest_side_32(self, image, target_size):
        """
        Масштабирует изображение так, чтобы наименьшая сторона = target_size (кратно 32),
        а большая сторона обрезается до ближайшего меньшего кратного 32.
        """
        # Сохраняем оригинальное изображение и размеры
        original_image = image
        B, H, W, C = image.shape
        original_width = W
        original_height = H

        # Определяем, какая сторона меньше
        if H < W:
            # Высота — меньшая сторона → масштабируем по высоте
            scale = target_size / H
            new_H_raw = target_size
            new_W_raw = int(round(W * scale))
        elif W < H:
            # Ширина — меньшая сторона → масштабируем по ширине
            scale = target_size / W
            new_W_raw = target_size
            new_H_raw = int(round(H * scale))
        else:
            # Квадрат
            scale = target_size / H
            new_H_raw = target_size
            new_W_raw = target_size

        # Определяем метод интерполяции
        mode = 'bicubic' if scale > 1.0 else 'area'

        # Конвертируем в [B, C, H, W] для torch.nn.functional.interpolate
        image_perm = image.permute(0, 3, 1, 2)  # [B, H, W, C] -> [B, C, H, W]

        # Масштабируем
        if mode == 'bicubic':
            resized = F.interpolate(image_perm, size=(new_H_raw, new_W_raw), mode='bicubic', align_corners=False)
        else:  # 'area'
            resized = F.interpolate(image_perm, size=(new_H_raw, new_W_raw), mode='area')

        # Возвращаем в [B, H, W, C]
        resized = resized.permute(0, 2, 3, 1)

        # Приводим большую сторону к кратности 32 с обрезкой по центру
        if H < W:
            # Была высота меньше → ширина — большая сторона
            new_W = (new_W_raw // 32) * 32
            new_H = new_H_raw  # = target_size
            # Обрезка по ширине
            left = (new_W_raw - new_W) // 2
            right = left + new_W
            top = 0
            bottom = new_H
        elif W < H:
            # Была ширина меньше → высота — большая сторона
            new_H = (new_H_raw // 32) * 32
            new_W = new_W_raw  # = target_size
            # Обрезка по высоте
            top = (new_H_raw - new_H) // 2
            bottom = top + new_H
            left = 0
            right = new_W
        else:
            # Квадрат → обе стороны = target_size, кратны 32 — обрезки нет
            new_W = new_W_raw
            new_H = new_H_raw
            left = 0
            right = new_W
            top = 0
            bottom = new_H

        # Обрезаем
        cropped = resized[:, top:bottom, left:right, :]

        return (
            original_image,
            original_width,
            original_height,
            cropped,
            new_W,
            new_H
        )

# Регистрация ноды
NODE_CLASS_MAPPINGS = {
    "ImageScaleToNearestSide32": ImageScaleToNearestSide32
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ImageScaleToNearestSide32": "Image Scale to Nearest Side (32-step)"
}
