# filename: any_to_types.py
import math
from typing import Any

try:
    import torch
except Exception as e:
    raise RuntimeError("[any_to_type] torch is required by ComfyUI runtime") from e


def _type_name(x: Any) -> str:
    """Удобное имя типа для сообщений об ошибках."""
    if isinstance(x, torch.Tensor):
        return f"Tensor(shape={tuple(x.shape)})"
    return type(x).__name__


def _is_image(x: Any) -> bool:
    """Проверка формата IMAGE: (B,H,W,C) с C==3, float."""
    if not isinstance(x, torch.Tensor):
        return False
    if x.dim() != 4:
        return False
    b, h, w, c = x.shape
    return c == 3 and x.dtype in (
        torch.float32,
        torch.float64,
        torch.float16,
        torch.bfloat16,
    )


def _is_mask(x: Any) -> bool:
    """Проверка, что тензор можно считать MASK в одном из допустимых форматов."""
    if not isinstance(x, torch.Tensor):
        return False
    if x.dim() == 3:
        # (B,H,W)
        return True
    if x.dim() == 4:
        # (B,1,H,W) или (B,H,W,1)
        b, c_or_h, h_or_w, w_or_1 = x.shape
        return (c_or_h == 1) or (w_or_1 == 1)
    return False


def _to_mask_tensor(x: torch.Tensor) -> torch.Tensor:
    """
    Привести произвольную маску к формату (B,1,H,W), float32, значения [0,1] без масштабирования.
    Допускает вход (B,H,W) или (B,1,H,W) или (B,H,W,1).
    """
    if x.dim() == 3:
        # (B,H,W) -> (B,1,H,W)
        x = x.unsqueeze(1)
    elif x.dim() == 4:
        # либо (B,1,H,W), либо (B,H,W,1)
        if x.shape[1] == 1:
            # уже (B,1,H,W)
            pass
        elif x.shape[-1] == 1:
            # (B,H,W,1) -> (B,1,H,W)
            x = x.permute(0, 3, 1, 2)
        else:
            raise RuntimeError(
                f"[any_to_type] cannot normalize MASK: unexpected shape {tuple(x.shape)}"
            )
    else:
        raise RuntimeError(
            f"[any_to_type] cannot normalize MASK: unexpected rank {x.dim()}"
        )
    if x.dtype != torch.float32:
        x = x.float()
    x = torch.clamp(x, 0.0, 1.0)
    return x


def _image_to_mask(img: torch.Tensor, binarize: bool, thr: float) -> torch.Tensor:
    """
    IMAGE(B,H,W,3) -> MASK(B,1,H,W), значения [0,1], опциональная бинаризация.
    """
    if not _is_image(img):
        raise RuntimeError(
            f"[any_to_type] expected IMAGE tensor (B,H,W,3), got {tuple(img.shape)}"
        )
    # Среднее по каналам -> (B,H,W)
    mask = img.mean(dim=-1)
    # -> (B,1,H,W)
    mask = mask.unsqueeze(1)
    mask = torch.clamp(mask, 0.0, 1.0)
    if binarize:
        mask = (mask >= float(thr)).float()
    return mask


def _mask_to_image(mask: torch.Tensor) -> torch.Tensor:
    """
    MASK(B,1,H,W)/(B,H,W)/(B,H,W,1) -> IMAGE(B,H,W,3) с повторением каналов, [0,1].
    """
    mask = _to_mask_tensor(mask)  # (B,1,H,W)
    # -> (B,H,W,1)
    img1 = mask.permute(0, 2, 3, 1)
    # повторить канал до 3
    img3 = img1.repeat(1, 1, 1, 3)
    img3 = torch.clamp(img3, 0.0, 1.0)
    return img3


def _format_mask_b1hw_to(
    mask_b1hw: torch.Tensor, fmt: str, strict: bool
) -> torch.Tensor:
    """
    Преобразовать каноническую маску (B,1,H,W) к одному из форматов: B1HW, BHW, HW.
    - HW допустим только при B==1; при Strict=False берём первый элемент батча.
    """
    if mask_b1hw.dim() != 4 or mask_b1hw.shape[1] != 1:
        raise RuntimeError(
            f"[any_to_type] expected (B,1,H,W) to format, got {tuple(mask_b1hw.shape)}"
        )

    b, _, h, w = mask_b1hw.shape

    if fmt == "B1HW":
        return mask_b1hw
    elif fmt == "BHW":
        return mask_b1hw.squeeze(1)  # (B,H,W)
    elif fmt == "HW":
        if b == 1:
            return mask_b1hw[0, 0, :, :]  # (H,W)
        if strict:
            raise RuntimeError(
                f"[any_to_type] Output_Format=HW requires batch size 1, got B={b}. "
                f"Use B1HW/BHW or set Strict=False to take the first item."
            )
        print(
            f"[any_to_type] Warning: HW requested with B={b}; returning first batch item (H,W)."
        )
        return mask_b1hw[0, 0, :, :]
    else:
        raise RuntimeError(f"[any_to_type] unknown Output_Format: {fmt}")


def _round_to_int(x: float, mode: str) -> int:
    if mode == "round":
        return int(round(x))
    elif mode == "floor":
        return math.floor(x)
    elif mode == "ceil":
        return math.ceil(x)
    elif mode == "truncate":
        return int(x)  # усечение к 0
    else:
        raise RuntimeError(f"[any_to_type] unknown rounding mode: {mode}")


def _parse_float_from_string(s: str, strict: bool) -> float:
    s_clean = s.strip()
    if not strict:
        if "." not in s_clean and s_clean.count(",") == 1:
            s_clean = s_clean.replace(",", ".", 1)
    return float(s_clean)


def _parse_int_from_string(s: str, rounding: str, strict: bool) -> int:
    s_clean = s.strip()
    try:
        return int(s_clean, 10)
    except Exception:
        # не целое, пробуем как float
        f = _parse_float_from_string(s_clean, strict)
        return _round_to_int(f, rounding)


_TRUE_LITERALS = {"true", "1", "yes", "on", "да", "вкл"}
_FALSE_LITERALS = {"false", "0", "no", "off", "нет", "выкл", ""}


class ImageFromAny:
    """
    Нода "ANY To IMAGE".
    Разрешены входы: IMAGE (pass-through), MASK (MASK->IMAGE).
    Иначе — RuntimeError.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "Value": ("*",),
                "Strict": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("Image",)
    FUNCTION = "execute"
    CATEGORY = "utils/cast"
    OUTPUT_NODE = False

    def execute(self, Value, Strict):
        try:
            if _is_image(Value):
                return (Value,)
            if _is_mask(Value):
                return (_mask_to_image(Value),)
            raise RuntimeError(
                f"[any_to_type] cannot cast {_type_name(Value)} to IMAGE "
                f"(only IMAGE or MASK are supported)"
            )
        except RuntimeError as e:
            if Strict:
                raise
            # Даже при Strict=False в IMAGE мы не «синтезируем» изображение — это запрещено ТЗ.
            raise


class MaskFromAny:
    """
    Нода "ANY To MASK".
    Разрешены входы: MASK (нормализация формы), IMAGE (IMAGE->MASK).
    Output_Format управляет формой выхода: B1HW (B,1,H,W), BHW (B,H,W), HW (H,W; только B==1).
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "Value": ("*",),
                "Binarize_From_Image": ("BOOLEAN", {"default": False}),
                "Mask_Threshold": (
                    "FLOAT",
                    {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01},
                ),
                "Output_Format": (["B1HW", "BHW", "HW"], {"default": "B1HW"}),
                "Strict": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("MASK",)
    RETURN_NAMES = ("Mask",)
    FUNCTION = "execute"
    CATEGORY = "utils/cast"
    OUTPUT_NODE = False

    def execute(
        self, Value, Binarize_From_Image, Mask_Threshold, Output_Format, Strict
    ):
        try:
            if _is_mask(Value):
                mask_b1hw = _to_mask_tensor(Value)  # (B,1,H,W)
                out = _format_mask_b1hw_to(mask_b1hw, Output_Format, Strict)
                return (out,)
            if _is_image(Value):
                mask_b1hw = _image_to_mask(
                    Value, Binarize_From_Image, Mask_Threshold
                )  # (B,1,H,W)
                out = _format_mask_b1hw_to(mask_b1hw, Output_Format, Strict)
                return (out,)
            raise RuntimeError(
                f"[any_to_type] cannot cast {_type_name(Value)} to MASK "
                f"(only IMAGE or MASK are supported)"
            )
        except RuntimeError:
            raise
        except Exception as e:
            raise RuntimeError(
                f"[any_to_type] unexpected error in MASK cast: {e}"
            ) from e


class IntFromAny:
    """
    Нода "ANY To INT".
    Поддерживает: INT (как есть), FLOAT (с округлением), BOOLEAN (True->1/False->0),
    STRING (int или float->округление при необходимости).
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "Value": ("*",),
                "Rounding_Mode": (
                    ["round", "floor", "ceil", "truncate"],
                    {"default": "round"},
                ),
                "Strict": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("INT",)
    RETURN_NAMES = ("Int",)
    FUNCTION = "execute"
    CATEGORY = "utils/cast"
    OUTPUT_NODE = False

    def execute(self, Value, Rounding_Mode, Strict):
        try:
            if isinstance(Value, bool):
                return (1 if Value else 0,)
            if isinstance(Value, int) and not isinstance(Value, bool):
                return (Value,)
            if isinstance(Value, float):
                if math.isnan(Value) or math.isinf(Value):
                    # NaN при Strict=True — ошибка; при Strict=False -> 0 (truncate).
                    if math.isnan(Value):
                        if Strict:
                            raise RuntimeError("[any_to_type] cannot cast NaN to INT")
                        return (0,)
                    # inf/-inf -> условная отсечка (сателлиты для предсказуемости)
                    lim = 2**31 - 1
                    return (-lim - 1 if Value < 0 else lim,)
                return (_round_to_int(Value, Rounding_Mode),)
            if isinstance(Value, str):
                try:
                    result = _parse_int_from_string(Value, Rounding_Mode, Strict)
                    return (result,)
                except Exception:
                    raise RuntimeError(f'[any_to_type] cannot parse "{Value}" as INT')
            # Остальные типы недопустимы (включая IMAGE/MASK)
            raise RuntimeError(f"[any_to_type] cannot cast {_type_name(Value)} to INT")
        except RuntimeError:
            raise
        except Exception as e:
            raise RuntimeError(
                f"[any_to_type] unexpected error in INT cast: {e}"
            ) from e


class FloatFromAny:
    """
    Нода "ANY To FLOAT".
    Поддерживает: FLOAT (как есть), INT (float()), BOOLEAN (1.0/0.0), STRING (float()) с мягкой запятой при Strict=False.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "Value": ("*",),
                "Strict": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("FLOAT",)
    RETURN_NAMES = ("Float",)
    FUNCTION = "execute"
    CATEGORY = "utils/cast"
    OUTPUT_NODE = False

    def execute(self, Value, Strict):
        try:
            if isinstance(Value, bool):
                return (1.0 if Value else 0.0,)
            if isinstance(Value, (int, float)) and not isinstance(Value, bool):
                return (float(Value),)
            if isinstance(Value, str):
                try:
                    f = _parse_float_from_string(Value, Strict)
                    return (f,)
                except Exception:
                    raise RuntimeError(f'[any_to_type] cannot parse "{Value}" as FLOAT')
            raise RuntimeError(
                f"[any_to_type] cannot cast {_type_name(Value)} to FLOAT"
            )
        except RuntimeError:
            raise
        except Exception as e:
            raise RuntimeError(
                f"[any_to_type] unexpected error in FLOAT cast: {e}"
            ) from e


class BooleanFromAny:
    """
    Нода "ANY To BOOLEAN".
    Поддерживает: BOOLEAN (как есть), INT/FLOAT (value != 0; NaN: ошибка при Strict=True, False при Strict=False),
    STRING (по словарю истинности/ложности).
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "Value": ("*",),
                "Strict": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("BOOLEAN",)
    RETURN_NAMES = ("Bool",)
    FUNCTION = "execute"
    CATEGORY = "utils/cast"
    OUTPUT_NODE = False

    def execute(self, Value, Strict):
        try:
            if isinstance(Value, bool):
                return (Value,)
            if isinstance(Value, (int, float)) and not isinstance(Value, bool):
                if isinstance(Value, float) and math.isnan(Value):
                    if Strict:
                        raise RuntimeError("[any_to_type] cannot cast NaN to BOOLEAN")
                    return (False,)
                return (Value != 0,)
            if isinstance(Value, str):
                s = Value.strip().lower()
                if s in _TRUE_LITERALS:
                    return (True,)
                if s in _FALSE_LITERALS:
                    return (False,)
                if Strict:
                    raise RuntimeError(
                        f'[any_to_type] cannot parse "{Value}" as BOOLEAN'
                    )
                return (False,)
            raise RuntimeError(
                f"[any_to_type] cannot cast {_type_name(Value)} to BOOLEAN"
            )
        except RuntimeError:
            raise
        except Exception as e:
            raise RuntimeError(
                f"[any_to_type] unexpected error in BOOLEAN cast: {e}"
            ) from e


class StringFromAny:
    """
    Нода "ANY To STRING".
    Любой тип преобразуется в строковое представление.
    IMAGE/MASK выводятся в формате: "IMAGE[B,H,W,3]" и "MASK[B,1,H,W]".
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "Value": ("*",),
                "Strict": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("Text",)
    FUNCTION = "execute"
    CATEGORY = "utils/cast"
    OUTPUT_NODE = False

    def execute(self, Value, Strict):
        try:
            if isinstance(Value, str):
                return (Value,)
            if isinstance(Value, (int, float, bool)):
                return (str(Value),)
            if _is_image(Value):
                b, h, w, c = Value.shape
                return (f"IMAGE[{b},{h},{w},{c}]",)
            if _is_mask(Value):
                m = _to_mask_tensor(Value)
                b, c, h, w = m.shape  # c==1
                return (f"MASK[{b},{c},{h},{w}]",)
            # Любой другой объект — стандартное str()
            return (str(Value),)
        except Exception as e:
            raise RuntimeError(
                f"[any_to_type] unexpected error in STRING cast: {e}"
            ) from e


# Регистрация нод
NODE_CLASS_MAPPINGS = {
    "ImageFromAny": ImageFromAny,
    "MaskFromAny": MaskFromAny,
    "IntFromAny": IntFromAny,
    "FloatFromAny": FloatFromAny,
    "BooleanFromAny": BooleanFromAny,
    "StringFromAny": StringFromAny,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ImageFromAny": "🧩 ANY To IMAGE",
    "MaskFromAny": "🧩 ANY To MASK",
    "IntFromAny": "🧩 ANY To INT",
    "FloatFromAny": "🧩 ANY To FLOAT",
    "BooleanFromAny": "🧩 ANY To BOOLEAN",
    "StringFromAny": "🧩 ANY To STRING",
}
