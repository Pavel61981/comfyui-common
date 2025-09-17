import math
import sys
import traceback

import torch
import numpy as np
from PIL import Image

# Comfy utilities (ProgressBar)
try:
    import comfy.utils as comfy_utils
except Exception:
    comfy_utils = None

class ImageGeminiResolutionAligner:
    CATEGORY = "Image/Transform"
    DESCRIPTION = "Автоматическое приведение изображения к ближайшему стандартному соотношению сторон (Gemini/SDXL-подход)."

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "upscaler": (["auto", "bicubic", "area", "bilinear", "nearest"],),
                "resize_mode": (["Scale and Crop to Fill", "Scale to Fit"],),
                "preserve_original_size": ("BOOLEAN", {"default": True}),
                "side": ([
                    "по середине",
                    "сверху",
                    "снизу",
                    "слева",
                    "справа"
                ], {"default": "по середине"}),
            }
        }

    # Возвращаем: исходное изображение, оригинальные размеры, обработанное изображение, новые размеры
    RETURN_TYPES = ("IMAGE", "INT", "INT", "IMAGE", "INT", "INT")
    RETURN_NAMES = ("original_image", "original_width", "original_height", "new_image", "new_width", "new_height")
    FUNCTION = "process"

    # Предустановленные целевые соотношения/разрешения (как в ТЗ)
    TARGETS = [
        ("3:4", 864, 1184),
        ("9:16", 768, 1344),
        ("1:1", 1024, 1024),
        ("4:3", 1184, 864),
        ("16:9", 1344, 768),
    ]

    def _pil_from_tensor(self, t: torch.Tensor) -> Image.Image:
        """
        t: [H, W, C], float in [0,1] or [0,255]
        """
        if not torch.is_tensor(t):
            raise TypeError("Ожидался torch.Tensor")
        arr = t.detach().cpu().numpy()
        # Ensure shape H,W,C
        if arr.ndim != 3:
            raise ValueError("Ожидается тензор формы [H,W,C]")
        # If floats in 0..1, convert to 0..255
        if arr.dtype == np.float32 or arr.dtype == np.float64:
            arr = np.clip(arr, 0.0, 1.0)
            arr = (arr * 255.0).round().astype(np.uint8)
        elif arr.dtype != np.uint8:
            arr = arr.astype(np.uint8)
        return Image.fromarray(arr)

    def _tensor_from_pil(self, pil_img: Image.Image, dtype=torch.float32) -> torch.Tensor:
        arr = np.array(pil_img)  # H,W,C uint8
        if arr.ndim == 2:
            arr = np.stack([arr, arr, arr], axis=-1)
        arr = arr.astype(np.float32) / 255.0
        t = torch.from_numpy(arr)
        return t

    def _choose_interpolation(self, upscaler: str, scale_factor: float):
        # Map to PIL resampling enums
        try:
            Resampling = Image.Resampling
        except AttributeError:
            # PIL older versions
            Resampling = Image
        if upscaler == "auto":
            # уменьшение -> area/box, увеличение -> bicubic
            if scale_factor < 1.0:
                return getattr(Resampling, "BOX", Image.BOX)
            else:
                return getattr(Resampling, "BICUBIC", Image.BICUBIC)
        elif upscaler == "bicubic":
            return getattr(Resampling, "BICUBIC", Image.BICUBIC)
        elif upscaler == "area":
            return getattr(Resampling, "BOX", Image.BOX)
        elif upscaler == "bilinear":
            return getattr(Resampling, "BILINEAR", Image.BILINEAR)
        elif upscaler == "nearest":
            return getattr(Resampling, "NEAREST", Image.NEAREST)
        else:
            return getattr(Resampling, "BICUBIC", Image.BICUBIC)

    def _make_divisible_by_32(self, value: int) -> int:
        if value <= 0:
            return 32
        return max(32, (value // 32) * 32)

    def process(self, image, upscaler="auto", resize_mode="Scale and Crop to Fill", preserve_original_size=True, side="по середине"):
        """
        image: torch.Tensor shape [B, H, W, C] (B - batch)
        upscaler: one of ["auto","bicubic","area","bilinear","nearest"]
        resize_mode: "Scale and Crop to Fill" or "Scale to Fit"
        preserve_original_size: bool
        side: alignment for cropping — "по середине", "сверху", "снизу", "слева", "справа"
        """
        # Basic validations
        try:
            if image is None:
                raise ValueError("Пустой вход image.")
            if not torch.is_tensor(image):
                raise TypeError("Вход 'image' должен быть torch.Tensor формата [B,H,W,C].")
            if image.ndim != 4:
                raise ValueError("Ожидается входной тензор формы [B,H,W,C].")
            batch_size = image.shape[0]
            orig_h = int(image.shape[1])
            orig_w = int(image.shape[2])
            if orig_h <= 0 or orig_w <= 0:
                raise ValueError("Некорректные размеры изображения.")
        except Exception as e:
            print(f"[ImageAlignResolutionGemini] Ошибка проверки входа: {e}")
            traceback.print_exc()
            return (image, 0, 0, image, image.shape[2], image.shape[1])

        print(f"[ImageAlignResolutionGemini] Вход: batch={batch_size}, width={orig_w}, height={orig_h}")
        orig_ratio = orig_w / orig_h if orig_h != 0 else 1.0

        # Выбор ближайшего соотношения сторон из списка
        best_target = None
        best_diff = None
        for name, tw, th in self.TARGETS:
            target_ratio = tw / th
            diff = abs(orig_ratio - target_ratio)
            if best_diff is None or diff < best_diff:
                best_diff = diff
                best_target = (name, tw, th)
        target_name, target_w, target_h = best_target
        print(f"[ImageAlignResolutionGemini] Выбранное соотношение: {target_name} ({target_w}x{target_h})")

        # Рассчитать финальные размеры
        if preserve_original_size:
            # масштабируем целевое разрешение максимально, чтобы не превышать исходное
            scale = min(orig_w / target_w, orig_h / target_h)
            if scale <= 0:
                scale = 1.0
            new_w = max(32, int(math.floor(target_w * scale)))
            new_h = max(32, int(math.floor(target_h * scale)))
            # делаем кратным 32
            new_w = self._make_divisible_by_32(new_w)
            new_h = self._make_divisible_by_32(new_h)
            # protection: ensure <= original
            new_w = min(new_w, orig_w)
            new_h = min(new_h, orig_h)
            # after min, re-make divisible
            new_w = self._make_divisible_by_32(new_w)
            new_h = self._make_divisible_by_32(new_h)
            print(f"[ImageAlignResolutionGemini] preserve_original_size=True -> рассчитаны размеры {new_w}x{new_h}")
        else:
            new_w = self._make_divisible_by_32(int(target_w))
            new_h = self._make_divisible_by_32(int(target_h))
            print(f"[ImageAlignResolutionGemini] preserve_original_size=False -> использованы стандартные размеры {new_w}x{new_h}")

        # Ensure non-zero
        new_w = max(32, new_w)
        new_h = max(32, new_h)

        # Prepare progress bar (если доступен)
        pbar = None
        if comfy_utils is not None and hasattr(comfy_utils, "ProgressBar"):
            try:
                pbar = comfy_utils.ProgressBar(batch_size)
            except Exception:
                pbar = None

        results = []
        try:
            for idx in range(batch_size):
                pil = self._pil_from_tensor(image[idx])
                iw, ih = pil.size  # PIL: (width, height)
                # Decide interpolation: based on scale factor relative to original dims
                scale_factor = (new_w / iw + new_h / ih) / 2.0  # average scale
                resample = self._choose_interpolation(upscaler, scale_factor)

                if resize_mode == "Scale and Crop to Fill":
                    # scale preserving aspect to cover the target, then crop
                    scale = max(new_w / iw, new_h / ih)
                    intermediate_w = max(1, int(round(iw * scale)))
                    intermediate_h = max(1, int(round(ih * scale)))
                    pil_resized = pil.resize((intermediate_w, intermediate_h), resample=resample)

                    # Определяем позицию обрезки в зависимости от side
                    if side == "по середине":
                        left = (intermediate_w - new_w) // 2
                        top = (intermediate_h - new_h) // 2
                    elif side == "сверху":
                        left = (intermediate_w - new_w) // 2
                        top = 0
                    elif side == "снизу":
                        left = (intermediate_w - new_w) // 2
                        top = intermediate_h - new_h
                    elif side == "слева":
                        left = 0
                        top = (intermediate_h - new_h) // 2
                    elif side == "справа":
                        left = intermediate_w - new_w
                        top = (intermediate_h - new_h) // 2
                    else:
                        # fallback to center
                        left = (intermediate_w - new_w) // 2
                        top = (intermediate_h - new_h) // 2

                    right = left + new_w
                    bottom = top + new_h
                    pil_final = pil_resized.crop((left, top, right, bottom))

                else:
                    # Scale to Fit (растяжение без обрезки) — просто изменить до точных new_w x new_h
                    pil_final = pil.resize((new_w, new_h), resample=resample)

                t_out = self._tensor_from_pil(pil_final)
                results.append(t_out)

                # update progress
                if pbar is not None:
                    try:
                        pbar.update(1)
                    except Exception:
                        pass

            # stack into tensor [B,H,W,C]
            stacked = torch.stack(results, dim=0)

        except Exception as e:
            print(f"[ImageAlignResolutionGemini] Ошибка при обработке: {e}")
            traceback.print_exc()
            # Fallback: вернуть оригинал
            return (image, orig_w, orig_h, image, orig_w, orig_h)

        # original image should be returned unchanged (pass-through)
        original_image = image
        new_image = stacked

        print(f"[ImageAlignResolutionGemini] Готово. Новые размеры: {new_w}x{new_h}")

        return (original_image, orig_w, orig_h, new_image, new_w, new_h)

# --- Регистрация ноды в ComfyUI ---
NODE_CLASS_MAPPINGS = {"ImageGeminiResolutionAligner": ImageGeminiResolutionAligner}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ImageGeminiResolutionAligner": "Image Align Resolution (Gemini)"
}
