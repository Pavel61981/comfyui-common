import torch
import torch.nn.functional as F
import logging

logger = logging.getLogger(__name__)

class ImagesSizeAligner:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image_a": ("IMAGE",),
                "image_b": ("IMAGE",),
            },
            "optional": {
                "name": ("STRING", {"default": "", "multiline": False}),
            }
        }

    RETURN_TYPES = ("IMAGE", "IMAGE", "INT", "INT")
    RETURN_NAMES = ("image_a_resized", "image_b_resized", "target_width", "target_height")
    FUNCTION = "align"
    CATEGORY = "image/processing"

    def align(self, image_a, image_b, name=""):
        """
        Приводит два изображения к одинаковому размеру.
        Большее масштабируется с сохранением пропорций и обрезается по центру до размеров меньшего.
        Меньшее возвращается без изменений.
        """
        # Получаем размеры
        b1, h1, w1, c1 = image_a.shape
        b2, h2, w2, c2 = image_b.shape

        # Проверка совпадения каналов
        if c1 != c2:
            raise ValueError(f"Количество каналов не совпадает: {c1} vs {c2}")

        # Логируем предупреждение, если размеры не совпадают
        if (h1, w1) != (h2, w2):
            name_part = f" '{name}'" if name else ""
            logger.warning(
                f"ImagesSizeAligner{name_part}: Размеры не совпадают — "
                f"image_a: {w1}x{h1}, image_b: {w2}x{h2}. Будет выполнено масштабирование и обрезка."
            )

        # Определяем целевое изображение — с меньшей площадью
        area1 = h1 * w1
        area2 = h2 * w2

        if area1 <= area2:
            target_h, target_w = h1, w1
            # image_a остаётся без изменений
            img_a_out = image_a
            # image_b нужно масштабировать и обрезать
            img_b_out = self.resize_and_crop(image_b, target_h, target_w)
        else:
            target_h, target_w = h2, w2
            # image_b остаётся без изменений
            img_b_out = image_b
            # image_a нужно масштабировать и обрезать
            img_a_out = self.resize_and_crop(image_a, target_h, target_w)

        return (img_a_out, img_b_out, target_w, target_h)

    def resize_and_crop(self, image, target_h, target_w):
        """
        Масштабирует изображение с сохранением пропорций, затем обрезает по центру до target_h x target_w.
        Вход: тензор [B, H, W, C]
        Выход: тензор [B, target_h, target_w, C]
        """
        b, h, w, c = image.shape

        # Переводим в формат [B, C, H, W] для F.interpolate
        image = image.permute(0, 3, 1, 2)  # [B, C, H, W]

        # Определяем, какую сторону масштабировать, чтобы покрыть целевой размер
        scale_h = target_h / h
        scale_w = target_w / w

        # Выбираем минимальный scale, чтобы изображение ПОКРЫЛО целевой размер (crop-to-fit)
        scale = max(scale_h, scale_w)

        new_h = int(h * scale)
        new_w = int(w * scale)

        # Масштабируем
        resized = F.interpolate(image, size=(new_h, new_w), mode='bilinear', align_corners=False)

        # Обрезаем по центру
        start_h = (new_h - target_h) // 2
        start_w = (new_w - target_w) // 2

        cropped = resized[:, :, start_h:start_h + target_h, start_w:start_w + target_w]

        # Возвращаем в формат [B, H, W, C]
        cropped = cropped.permute(0, 2, 3, 1)

        return cropped

# Регистрация ноды в ComfyUI
NODE_CLASS_MAPPINGS = {
    "ImagesSizeAligner": ImagesSizeAligner
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ImagesSizeAligner": "📏 Images Size Aligner"
}
