# filename: mask_get_by_index.py
import re
from typing import Any

try:
    import torch
except Exception as e:
    raise RuntimeError("[mask_get_by_index] ComfyUI requires torch runtime") from e


class MaskGetByIndex:
    """
    🧩 Mask Get By Index
    ====================
    Выбирает **одну** маску из батча по **целочисленному** индексу.

    Входы:
      • Masks_or_Images — принимает MASK или IMAGE:
          - MASK: (B,1,H,W) или (B,H,W) или (B,H,W,1) — нормализуется к (B,1,H,W)
          - IMAGE: (B,H,W,3) — конвертируется в маску усреднением по каналам → (B,1,H,W)
      • Index (INT) — индекс элемента батча. Отрицательные индексы поддерживаются (−1 — последний).

    Параметры:
      • Binarize_From_Image (bool, по умолчанию False) — бинаризация при входе IMAGE
      • Mask_Threshold (float в [0..1], по умолчанию 0.5) — порог бинаризации
      • Output_Format ("B1HW" | "BHW" | "HW", по умолчанию "B1HW") — форма выхода
      • Index_Adjust (string, по умолчанию "") — корректировка индекса:
            "i"     — без изменений,
            "i+K"   — прибавить K,
            "i-K"   — вычесть K,
        пробелы допускаются: "i + 2", "i - 1".
      • Strict (bool, по умолчанию True):
            - при выходе индекса за пределы батча — ошибка; при Strict=False — clamping с предупреждением
            - при некорректном Index_Adjust — ошибка; при Strict=False — игнор c предупреждением

    Формат результата:
      • "B1HW" → (1,1,H,W)
      • "BHW"  → (1,H,W)
      • "HW"   → (H,W)

    Примечания:
      • Значения маски клипуются в [0,1], dtype — float32.
      • Внутренний базовый формат — (B,1,H,W); выбор по индексу уменьшает батч до 1.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "Masks_or_Images": ("*",),
                "Index": ("INT", {"default": 0, "step": 1}),
            },
            "optional": {
                "Binarize_From_Image": ("BOOLEAN", {"default": False}),
                "Mask_Threshold": (
                    "FLOAT",
                    {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01},
                ),
                "Output_Format": (["B1HW", "BHW", "HW"], {"default": "B1HW"}),
                "Index_Adjust": ("STRING", {"default": ""}),
                "Strict": ("BOOLEAN", {"default": True}),
            },
        }

    RETURN_TYPES = ("MASK",)
    RETURN_NAMES = ("Mask",)
    FUNCTION = "execute"
    CATEGORY = "utils/mask"
    OUTPUT_NODE = False

    # ---------- helpers ----------

    @staticmethod
    def _type_name(x: Any) -> str:
        if isinstance(x, torch.Tensor):
            return f"Tensor(shape={tuple(x.shape)})"
        return type(x).__name__

    @staticmethod
    def _is_image(x: Any) -> bool:
        if not isinstance(x, torch.Tensor) or x.dim() != 4:
            return False
        return x.shape[-1] == 3 and x.dtype in (
            torch.float32,
            torch.float64,
            torch.float16,
            torch.bfloat16,
        )

    @staticmethod
    def _is_mask(x: Any) -> bool:
        if not isinstance(x, torch.Tensor):
            return False
        if x.dim() == 3:
            return True  # (B,H,W)
        if x.dim() == 4:
            return x.shape[1] == 1 or x.shape[-1] == 1  # (B,1,H,W) или (B,H,W,1)
        return False

    @staticmethod
    def _to_mask_tensor(x: torch.Tensor) -> torch.Tensor:
        """Нормализовать маску к (B,1,H,W), float32, clamp[0,1]."""
        if x.dim() == 3:
            x = x.unsqueeze(1)  # (B,H,W) -> (B,1,H,W)
        elif x.dim() == 4:
            if x.shape[1] == 1:
                pass  # (B,1,H,W)
            elif x.shape[-1] == 1:
                x = x.permute(0, 3, 1, 2)  # (B,H,W,1) -> (B,1,H,W)
            else:
                raise RuntimeError(
                    f"[mask_get_by_index] cannot normalize MASK with shape {tuple(x.shape)}"
                )
        else:
            raise RuntimeError(f"[mask_get_by_index] unexpected mask rank: {x.dim()}")
        if x.dtype != torch.float32:
            x = x.float()
        return torch.clamp(x, 0.0, 1.0)

    @staticmethod
    def _image_to_mask(img: torch.Tensor, binarize: bool, thr: float) -> torch.Tensor:
        """IMAGE(B,H,W,3) -> MASK(B,1,H,W), clamp[0,1], опциональная бинаризация."""
        if not MaskGetByIndex._is_image(img):
            raise RuntimeError(
                f"[mask_get_by_index] expected IMAGE (B,H,W,3), got {MaskGetByIndex._type_name(img)}"
            )
        mask = img.mean(dim=-1)  # (B,H,W)
        mask = mask.unsqueeze(1)  # (B,1,H,W)
        mask = torch.clamp(mask, 0.0, 1.0)
        if binarize:
            mask = (mask >= float(thr)).float()
        return mask

    @staticmethod
    def _apply_index_adjust(i: int, adjust: str, strict: bool) -> int:
        """
        Применить корректировку индекса строкой:
          - "i"     → без изменений
          - "i+K"   → i + K
          - "i-K"   → i - K
        Пробелы допускаются. Пустая строка — без изменений.
        """
        if adjust is None:
            return i
        s = str(adjust).strip()
        if s == "":
            return i
        if re.fullmatch(r"\s*i\s*", s, flags=re.IGNORECASE):
            return i
        m = re.fullmatch(r"\s*i\s*([+-])\s*(\d+)\s*", s, flags=re.IGNORECASE)
        if m:
            sign, num = m.group(1), int(m.group(2))
            return i + num if sign == "+" else i - num
        msg = f'[mask_get_by_index] invalid Index_Adjust "{adjust}". Expected "i", "i+K" or "i-K".'
        if strict:
            raise RuntimeError(msg)
        print("[mask_get_by_index] Warning:", msg, "— ignored.")
        return i

    @staticmethod
    def _format_output(mask_1b1hw: torch.Tensor, fmt: str) -> torch.Tensor:
        """Сформировать выход по Output_Format: B1HW|(1,1,H,W), BHW|(1,H,W), HW|(H,W)."""
        if (
            mask_1b1hw.dim() != 4
            or mask_1b1hw.shape[0] != 1
            or mask_1b1hw.shape[1] != 1
        ):
            raise RuntimeError(
                f"[mask_get_by_index] expected (1,1,H,W), got {tuple(mask_1b1hw.shape)}"
            )
        if fmt == "B1HW":
            return mask_1b1hw
        if fmt == "BHW":
            return mask_1b1hw.squeeze(1)  # (1,H,W)
        if fmt == "HW":
            return mask_1b1hw[0, 0, :, :]  # (H,W)
        raise RuntimeError(f"[mask_get_by_index] unknown Output_Format: {fmt}")

    # ---------- execute ----------

    def execute(
        self,
        Masks_or_Images,
        Index,
        Binarize_From_Image=False,
        Mask_Threshold=0.5,
        Output_Format="B1HW",
        Index_Adjust="",
        Strict=True,
    ):
        try:
            # 1) Приводим вход к базовой форме маски (B,1,H,W)
            if self._is_mask(Masks_or_Images):
                mask_b1hw = self._to_mask_tensor(Masks_or_Images)
            elif self._is_image(Masks_or_Images):
                mask_b1hw = self._image_to_mask(
                    Masks_or_Images, Binarize_From_Image, Mask_Threshold
                )
            else:
                raise RuntimeError(
                    f"[mask_get_by_index] Masks_or_Images must be MASK or IMAGE, got {self._type_name(Masks_or_Images)}"
                )

            # 2) Базовый индекс — уже INT
            j = int(Index)

            # 3) Корректировка индекса
            j = self._apply_index_adjust(j, Index_Adjust, Strict)

            # 4) Нормализация/проверка диапазона и выбор
            B = mask_b1hw.shape[0]
            if j < 0:
                j = B + j  # pythonic negative index
            if j < 0 or j >= B:
                if Strict:
                    raise RuntimeError(
                        f"[mask_get_by_index] index {j} out of range for batch size {B}"
                    )
                # clamp
                j_clamped = max(0, min(B - 1, j))
                if j != j_clamped:
                    print(
                        f"[mask_get_by_index] Warning: index {j} clamped to {j_clamped} for B={B}."
                    )
                j = j_clamped

            # Выбор одного элемента батча: (1,1,H,W)
            selected = mask_b1hw[j : j + 1, :, :, :]

            # 5) Форматирование выхода
            out = self._format_output(selected, Output_Format)
            return (out,)

        except RuntimeError:
            raise
        except Exception as e:
            raise RuntimeError(f"[mask_get_by_index] unexpected error: {e}") from e


# Регистрация ноды
NODE_CLASS_MAPPINGS = {"MaskGetByIndex": MaskGetByIndex}
NODE_DISPLAY_NAME_MAPPINGS = {"MaskGetByIndex": "🧩 Mask Get By Index"}
