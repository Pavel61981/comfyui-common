# image_ensure_rgb.py

from typing import Tuple
import math

try:
    import torch
except Exception as e:  # pragma: no cover
    raise RuntimeError("[ImageEnsureRGB] PyTorch is not available") from e


class ImageEnsureRGB:
    """
    Нода «ImageEnsureRGB»:
    Приводит входной IMAGE к корректному формату ComfyUI:
    - форма: [B, H, W, 3] (BHWC)
    - dtype: float32
    - диапазон: [0..1] (исключая режим normalize='none')
    Если вход фактически RGBA (C==4), извлекает альфу в MASK (или премультиплицирует / отбрасывает по стратегии).
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "Image": ("IMAGE",),
                "alpha_strategy": (
                    ["to_mask", "premultiply", "drop"],
                    {"default": "to_mask"},
                ),
                "normalize": (
                    ["auto_255", "clip_only", "none"],
                    {"default": "auto_255"},
                ),
                "nan_inf_policy": (
                    ["clean", "error"],
                    {"default": "clean"},
                ),
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK")
    RETURN_NAMES = ("Image", "Mask")
    FUNCTION = "execute"
    CATEGORY = "utils/image"
    OUTPUT_NODE = False

    # ---------- helpers ----------

    @staticmethod
    def _finite_minmax(x: torch.Tensor) -> Tuple[float, float]:
        finite = torch.isfinite(x)
        if not finite.any():
            return float("nan"), float("nan")
        vals = x[finite]
        return float(vals.min().item()), float(vals.max().item())

    @staticmethod
    def _p99(x: torch.Tensor) -> float:
        """Перцентиль 99 по конечным значениям; NaN/Inf игнорируем (должны быть уже почищены)."""
        finite = torch.isfinite(x)
        if not finite.any():
            return float("nan")
        try:
            q = torch.quantile(x[finite], 0.99)
            return float(q.item())
        except Exception:
            # Фолбэк: берём max, если quantile недоступен
            return float(x[finite].max().item())

    @staticmethod
    def _ensure_bhwc(image: torch.Tensor) -> torch.Tensor:
        """
        Вход типа IMAGE в ComfyUI должен уже быть BHWC. Проверяем и отдаём понятную ошибку, если нет.
        """
        if not isinstance(image, torch.Tensor):
            raise RuntimeError("[ImageEnsureRGB] Input 'Image' is not a torch.Tensor")

        if image.ndim != 4:
            raise RuntimeError(
                f"[ImageEnsureRGB] Expected 4D BHWC tensor, got rank={image.ndim} with shape {tuple(image.shape)}"
            )
        b, h, w, c = image.shape
        if min(b, h, w, c) <= 0:
            raise RuntimeError(
                f"[ImageEnsureRGB] Non-positive dimensions in IMAGE shape {tuple(image.shape)}"
            )
        return image

    @staticmethod
    def _split_rgba_to_rgb_mask(
        bhwc: torch.Tensor, alpha_strategy: str
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        На входе: BHWC, C ∈ {1,3,4}
        Возврат: rgb(BHWC, 3ch), mask(BHW)
        """
        b, h, w, c = bhwc.shape
        device = bhwc.device

        if c not in (1, 3, 4):
            raise RuntimeError(
                f"[ImageEnsureRGB] Unsupported channel count C={c}. Allowed C: 1, 3, 4."
            )

        # Единичная маска по умолчанию
        ones_mask = torch.ones((b, h, w), dtype=torch.float32, device=device)

        if c == 1:
            rgb = bhwc.repeat(1, 1, 1, 3)
            return rgb, ones_mask

        if c == 3:
            return bhwc, ones_mask

        # c == 4
        rgb = bhwc[..., :3]
        a = bhwc[..., 3]  # shape (B,H,W)

        if alpha_strategy == "to_mask":
            return rgb, a
        elif alpha_strategy == "premultiply":
            rgb = rgb * a.unsqueeze(-1)
            return rgb, ones_mask
        elif alpha_strategy == "drop":
            return rgb, ones_mask
        else:
            raise RuntimeError(
                f"[ImageEnsureRGB] Unknown alpha_strategy: {alpha_strategy}"
            )

    @staticmethod
    def _apply_nan_inf_policy(
        img: torch.Tensor, mask: torch.Tensor, policy: str
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if policy == "error":
            if not torch.isfinite(img).all():
                mn, mx = ImageEnsureRGB._finite_minmax(img)
                raise RuntimeError(
                    f"[ImageEnsureRGB] NaN/Inf found in IMAGE data. Finite min/max: {mn}..{mx}"
                )
            if not torch.isfinite(mask).all():
                mn, mx = ImageEnsureRGB._finite_minmax(mask)
                raise RuntimeError(
                    f"[ImageEnsureRGB] NaN/Inf found in MASK data. Finite min/max: {mn}..{mx}"
                )
            return img, mask

        # clean
        img = torch.nan_to_num(img, nan=0.0, posinf=0.0, neginf=0.0)
        mask = torch.nan_to_num(mask, nan=0.0, posinf=0.0, neginf=0.0)
        return img, mask

    @staticmethod
    def _apply_normalize(
        img: torch.Tensor, mask: torch.Tensor, mode: str
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if mode not in ("auto_255", "clip_only", "none"):
            raise RuntimeError(f"[ImageEnsureRGB] Unknown normalize mode: {mode}")

        if mode == "none":
            # Не трогаем значения (диапазон может быть любым).
            return img, mask

        # Для auto_255/clip_only нам нужен clamp в [0..1],
        # но в auto_255 возможно деление на 255 перед clamp.
        if mode == "auto_255":
            i_min, i_max = ImageEnsureRGB._finite_minmax(img)
            # Мягкий детектор 0..255: min>=0, max<=255 и P99>1.5 (реально не [0..1])
            if not (math.isnan(i_min) or math.isnan(i_max)):
                if (i_min >= 0.0) and (i_max <= 255.0):
                    p99 = ImageEnsureRGB._p99(img)
                    if not math.isnan(p99) and p99 > 1.5:
                        img = img / 255.0

        img = img.clamp(0.0, 1.0)
        mask = mask.clamp(0.0, 1.0)
        return img, mask

    @staticmethod
    def _finalize_types_and_checks(
        img: torch.Tensor, mask: torch.Tensor, mode: str
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Приводим к float32 и contiguous
        if img.dtype != torch.float32:
            img = img.float()
        if mask.dtype != torch.float32:
            mask = mask.float()
        img = img.contiguous()
        mask = mask.contiguous()

        # Проверки формы
        if img.ndim != 4 or img.shape[-1] != 3:
            raise RuntimeError(
                f"[ImageEnsureRGB] Final check failed: IMAGE must be [B,H,W,3], got {tuple(img.shape)}"
            )
        if mask.ndim != 3:
            raise RuntimeError(
                f"[ImageEnsureRGB] Final check failed: MASK must be [B,H,W], got {tuple(mask.shape)}"
            )

        # Для режимов, где мы гарантируем [0..1], убедимся в диапазоне
        if mode in ("auto_255", "clip_only"):
            if torch.isfinite(img).any():
                mn, mx = ImageEnsureRGB._finite_minmax(img)
                tol = 1e-5
                if mn < -tol or mx > 1.0 + tol:
                    raise RuntimeError(
                        f"[ImageEnsureRGB] Final check failed: IMAGE values out of [0,1]. min={mn}, max={mx}"
                    )
            if torch.isfinite(mask).any():
                mn, mx = ImageEnsureRGB._finite_minmax(mask)
                tol = 1e-5
                if mn < -tol or mx > 1.0 + tol:
                    raise RuntimeError(
                        f"[ImageEnsureRGB] Final check failed: MASK values out of [0,1]. min={mn}, max={mx}"
                    )

        return img, mask

    # ---------- main ----------

    def execute(self, Image, alpha_strategy, normalize, nan_inf_policy):
        try:
            img = self._ensure_bhwc(Image)

            # Убедимся, что работаем на CPU и с float (тип IMAGE в ComfyUI обычно и так float32 BHWC)
            if img.device.type != "cpu":
                img = img.to("cpu")
            if not torch.is_floating_point(img):
                img = img.float()

            # Извлечь RGB/MASK по каналам
            rgb, mask = self._split_rgba_to_rgb_mask(img, alpha_strategy=alpha_strategy)

            # Обработка NaN/Inf согласно политике (сначала чистим/проверяем)
            rgb, mask = self._apply_nan_inf_policy(rgb, mask, policy=nan_inf_policy)

            # Нормализация диапазона
            rgb, mask = self._apply_normalize(rgb, mask, mode=normalize)

            # Завершение: типы, contiguous, проверки
            rgb, mask = self._finalize_types_and_checks(rgb, mask, mode=normalize)

            return (rgb, mask)

        except RuntimeError:
            # Сообщение уже информативное
            raise
        except Exception as e:
            raise RuntimeError(f"[ImageEnsureRGB] Unexpected error: {str(e)}") from e


# Регистрация ноды в ComfyUI
NODE_CLASS_MAPPINGS = {"ImageEnsureRGB": ImageEnsureRGB}
NODE_DISPLAY_NAME_MAPPINGS = {"ImageEnsureRGB": "🧩 Image Ensure RGB"}
