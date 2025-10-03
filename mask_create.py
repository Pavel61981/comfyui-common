# file: mask_create.py
import torch


class MaskCreate:
    """
    Нода "mask_create" — создаёт полностью заполненную маску (все значения = 1.0)
    заданного размера Height x Width.

    Входы:
      - Width  (INT): ширина маски в пикселях.
      - Height (INT): высота маски в пикселях.

    Выход:
      - Mask (MASK): тензор float32 формы [H, W] со значениями 1.0 (белая маска).
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "Width": ("INT", {"default": 512, "min": 1, "max": 8192, "step": 8}),
                "Height": ("INT", {"default": 512, "min": 1, "max": 8192, "step": 8}),
            }
        }

    RETURN_TYPES = ("MASK",)
    RETURN_NAMES = ("Mask",)
    FUNCTION = "execute"
    CATEGORY = "utils/mask"
    OUTPUT_NODE = False

    def execute(self, Width: int, Height: int):
        """
        Возвращает кортеж из одного элемента — маски (MASK) размером [Height, Width],
        полностью заполненной единицами.
        """
        try:
            if not isinstance(Width, int) or not isinstance(Height, int):
                raise ValueError("Width и Height должны быть целыми числами (INT).")

            if Width <= 0 or Height <= 0:
                raise ValueError("Width и Height должны быть > 0.")

            if Width > 8192 or Height > 8192:
                raise ValueError("Width и Height не должны превышать 8192.")

            mask = torch.ones((Height, Width), dtype=torch.float32, device="cpu")
            return (mask,)

        except Exception as e:
            msg = f"[mask_filled] Ошибка: {str(e)}"
            print(msg)
            raise RuntimeError(msg) from e


# Регистрация ноды в ComfyUI
NODE_CLASS_MAPPINGS = {"MaskCreate": MaskCreate}
NODE_DISPLAY_NAME_MAPPINGS = {"MaskCreate": "🧩 Mask Create"}
