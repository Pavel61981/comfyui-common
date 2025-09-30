# filename: image_select.py

import torch


class ImageSelect:
    """
    Выбирает одно изображение из батча BODY_MASKS по индексу.
    Индекс циклический (index % B), возвращает IMAGE с батчем размера 1.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "body_masks": ("IMAGE",),
                "index": ("INT", {"default": 0, "min": -1024, "max": 1024, "step": 1}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("IMAGE",)
    FUNCTION = "execute"
    CATEGORY = "Masquerade/Mask"
    OUTPUT_NODE = False

    def execute(self, body_masks: torch.Tensor, index: int = 0):
        if not isinstance(body_masks, torch.Tensor):
            raise RuntimeError("[ImageSelect] Ожидается тензор IMAGE")
        if body_masks.ndim != 4:
            raise RuntimeError(
                f"[ImageSelect] Ожидается тензор формы (B,H,W,C), получено: {tuple(body_masks.shape)}"
            )

        b = body_masks.shape[0]
        if b <= 0:
            raise RuntimeError("[ImageSelect] Пустой батч IMAGE")

        i = int(index) % b
        out = body_masks[i : i + 1]  # сохранить размер батча = 1
        return (out,)


# Регистрация в ComfyUI
NODE_CLASS_MAPPINGS = {"ImageSelect": ImageSelect}
NODE_DISPLAY_NAME_MAPPINGS = {"ImageSelect": "Image Select By Index"}
