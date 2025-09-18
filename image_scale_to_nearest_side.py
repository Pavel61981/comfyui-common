import torch
import torch.nn.functional as F

class ImageScaleToNearestSide:
    """
    Нода "scale_to_nearest_side" — масштабирует изображение так, чтобы
    меньшая сторона стала равна target_size, при этом соотношение сторон
    сохраняется максимально точно. Никакой обрезки и кратности 32 нет.

    Вход: IMAGE тензор формы [B, H, W, C], значения 0..1.
    Выход: оригинал + размеры, новая картинка + новые размеры.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),  # [B, H, W, C], float 0..1
                "target_size": (
                    "INT",
                    {
                        "default": 1344,
                        "min": 32,
                        "max": 4096,
                        "step": 32,
                        "display": "slider",
                    },
                ),
            },
        }

    RETURN_TYPES = ("IMAGE", "INT", "INT", "IMAGE", "INT", "INT")
    RETURN_NAMES = (
        "original_image",
        "original_width",
        "original_height",
        "new_image",
        "new_width",
        "new_height",
    )
    FUNCTION = "scale_to_nearest_side"
    CATEGORY = "Image Processing"
    OUTPUT_NODE = False

    def _validate_inputs(self, image: torch.Tensor, target_size: int):
        if not isinstance(image, torch.Tensor):
            raise RuntimeError("[scale_to_nearest_side] 'image' должен быть тензором torch.Tensor")

        if image.ndim != 4:
            raise RuntimeError(
                f"[scale_to_nearest_side] Ожидалась форма [B,H,W,C], получено ndim={image.ndim}"
            )

        B, H, W, C = image.shape
        if B <= 0 or H <= 0 or W <= 0 or C <= 0:
            raise RuntimeError(
                f"[scale_to_nearest_side] Некорректная форма тензора: {image.shape}"
            )

        if C != 3 and C != 4 and C != 1:
            # Разрешим 1/3/4 каналов, как это встречается в ComfyUI
            raise RuntimeError(
                f"[scale_to_nearest_side] Ожидалось 1, 3 или 4 канала, получено C={C}"
            )

        if not isinstance(target_size, int):
            raise RuntimeError("[scale_to_nearest_side] 'target_size' должен быть целым числом")

        if target_size < 32 or target_size > 4096:
            raise RuntimeError(
                f"[scale_to_nearest_side] 'target_size' вне допустимого диапазона [32, 4096]: {target_size}"
            )

    def _compute_new_hw(self, H: int, W: int, target_size: int):
        if H < W:
            scale = target_size / float(H)
            new_H = target_size
            new_W = int(round(W * scale))
        elif W < H:
            scale = target_size / float(W)
            new_W = target_size
            new_H = int(round(H * scale))
        else:
            # квадрат
            scale = target_size / float(H)
            new_H = target_size
            new_W = target_size

        # гарантируем минимум 1 пиксель
        new_H = max(1, int(new_H))
        new_W = max(1, int(new_W))
        return new_H, new_W, scale

    def scale_to_nearest_side(self, image: torch.Tensor, target_size: int):
        """
        Масштабирует изображение так, чтобы меньшая сторона стала равна target_size.
        Соотношение сторон сохраняется. Никакой обрезки и кратности 32.
        Интерполяция: bicubic при увеличении, area при уменьшении/равенстве.
        """
        try:
            self._validate_inputs(image, target_size)

            original_image = image
            B, H, W, C = image.shape
            original_width = int(W)
            original_height = int(H)

            new_H, new_W, scale = self._compute_new_hw(H, W, target_size)

            # Определяем режим интерполяции
            mode = "bicubic" if scale > 1.0 else "area"

            # Переставляем оси под interpolate: [B,H,W,C] -> [B,C,H,W]
            image_chw = image.permute(0, 3, 1, 2).contiguous()

            if mode == "bicubic":
                resized_chw = F.interpolate(
                    image_chw, size=(new_H, new_W), mode="bicubic", align_corners=False
                )
            else:
                resized_chw = F.interpolate(
                    image_chw, size=(new_H, new_W), mode="area"
                )

            # Возвращаем к [B,H,W,C]
            resized = resized_chw.permute(0, 2, 3, 1).contiguous()

            return (
                original_image,
                original_width,
                original_height,
                resized,
                int(new_W),
                int(new_H),
            )

        except Exception as e:
            msg = f"[scale_to_nearest_side] Ошибка: {str(e)}"
            print(msg)
            raise RuntimeError(msg) from e


# Регистрация ноды в ComfyUI
NODE_CLASS_MAPPINGS = {"ImageScaleToNearestSide": ImageScaleToNearestSide}
NODE_DISPLAY_NAME_MAPPINGS = {"ImageScaleToNearestSide": "Image Scale to Nearest Side"}
