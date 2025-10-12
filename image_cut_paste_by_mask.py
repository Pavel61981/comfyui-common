from typing import Tuple
import torch
import torch.nn.functional as F

# ===== ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ =====

def _ensure_single_image(image: torch.Tensor) -> torch.Tensor:
    """
    Приводит батч IMAGE (BxHxWxC) к одному тензору HxWxC (float32, [0,1]).
    Вызывает ошибку, если в батче не одно изображение.
    """
    if not torch.is_tensor(image):
        raise RuntimeError(f"[image] Ожидался torch.Tensor, получено: {type(image)}")
    if image.dim() != 4:
        raise RuntimeError(
            f"[image] Ожидался батч-тензор BxHxWxC, получено: {tuple(image.shape)}"
        )
    if image.shape[0] != 1:
        raise RuntimeError(
            f"[image] Ожидался один кадр (batch=1), получено: {image.shape[0]}"
        )
    return image.squeeze(0).to(dtype=torch.float32)


def _ensure_mask(mask: torch.Tensor, target_hw: Tuple[int, int]) -> torch.Tensor:
    """
    Приводит MASK (BxHxW) к HxW (float32, [0,1]) и сверяет размер.
    """
    if not torch.is_tensor(mask):
        raise RuntimeError(f"[mask] Ожидался torch.Tensor, получено: {type(mask)}")

    if mask.dim() == 3:
        if mask.shape[0] != 1:
            raise RuntimeError(
                f"[mask] Ожидалась одна маска (batch=1), получено: {mask.shape[0]}"
            )
        mask = mask.squeeze(0)

    if mask.dim() != 2:
        raise RuntimeError(
            f"[mask] Ожидался тензор HxW, после обработки батча получено: {tuple(mask.shape)}"
        )

    h, w = target_hw
    if mask.shape[0] != h or mask.shape[1] != w:
        raise RuntimeError(
            f"[mask] Размер маски {tuple(mask.shape)} не совпадает с изображением {(h, w)}"
        )
    return mask.clamp(0.0, 1.0).to(dtype=torch.float32)


def _to_comfy_image(img: torch.Tensor) -> torch.Tensor:
    """HxWxC -> BxHxWxC для порта IMAGE."""
    return img.unsqueeze(0)


def _to_comfy_mask(mask: torch.Tensor) -> torch.Tensor:
    """HxW -> BxHxW для порта MASK."""
    return mask.unsqueeze(0)


def _compute_bbox(mask_bin: torch.Tensor) -> Tuple[int, int, int, int]:
    """
    BBox по ненулевой маске (>0). Возвращает (x, y, w, h).
    """
    idx = torch.nonzero(mask_bin > 0, as_tuple=False)
    if idx.numel() == 0:
        raise RuntimeError("[mask] Пустая маска — нечего вырезать.")
    ys = idx[:, 0]
    xs = idx[:, 1]
    x0 = int(xs.min().item())
    x1 = int(xs.max().item())
    y0 = int(ys.min().item())
    y1 = int(ys.max().item())
    return x0, y0, (x1 - x0 + 1), (y1 - y0 + 1)


def _expand_and_clamp_bbox(bbox, offset, w, h):
    x, y, bw, bh = bbox
    d = max(0, int(offset))
    x0 = max(0, x - d)
    y0 = max(0, y - d)
    x1 = min(w, x + bw + d)
    y1 = min(h, y + bh + d)
    x0, y0 = int(x0), int(y0)
    w_new = int(max(1, x1 - x0))
    h_new = int(max(1, y1 - y0))
    return x0, y0, w_new, h_new


def _crop(img: torch.Tensor, bbox: Tuple[int, int, int, int]):
    x, y, w, h = bbox
    return img[y : y + h, x : x + w]


def _resize_hwc(
    img: torch.Tensor, h: int, w: int, mode: str = "bilinear"
) -> torch.Tensor:
    """
    Масштабирует HxWxC к (h,w). Возвращает HxWxC.
    """
    if img.shape[0] == h and img.shape[1] == w:
        return img
    x = img.permute(2, 0, 1).unsqueeze(0)
    x = F.interpolate(
        x, size=(h, w), mode=mode, align_corners=False if mode == "bilinear" else None
    )
    x = x.squeeze(0).permute(1, 2, 0).contiguous()
    return x


def _resize_mask_hw(mask: torch.Tensor, h: int, w: int) -> torch.Tensor:
    """
    Масштабирует HxW к (h,w), bilinear, затем клипуется в [0,1].
    """
    if mask.shape[0] == h and mask.shape[1] == w:
        return mask
    x = mask.unsqueeze(0).unsqueeze(0)
    x = F.interpolate(x, size=(h, w), mode="bilinear", align_corners=False)
    x = x.squeeze(0).squeeze(0)
    return x.clamp(0.0, 1.0)


def _empty_debug_overlay(device: torch.device) -> torch.Tensor:
    """
    Возвращает «пустой» оверлей 1x1x3 (чёрный/прозрачный).
    Нужен, чтобы превью в ноде визуально было пустым при debug_enabled=False.
    """
    return torch.zeros((1, 1, 3), dtype=torch.float32, device=device)


# ===== НОДЫ =====


class ImageCutByMask:
    """
    ✂️ Image Cut By Mask — вырезает изображение по маске с расширением bbox и опционально
    показывает в превью ноды оверлей: обрезка + полупрозрачная красная маска.
    Также возвращает размеры обрезки (ширину/высоту).
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE", {}),
                "mask": ("MASK", {}),
                "offset": ("INT", {"default": 16, "min": 0, "max": 4096, "step": 1}),
                "debug_enabled": ("BOOLEAN", {"default": False}),
                "debug_alpha": (
                    "FLOAT",
                    {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01},
                ),
            }
        }

    RETURN_TYPES = ("IMAGE", "IMAGE", "MASK", "BBOX", "INT", "INT", "IMAGE")
    RETURN_NAMES = (
        "base_image",
        "cropped_image",
        "cropped_mask",
        "bbox",
        "crop_width",
        "crop_height",
        "debug_overlay",
    )
    FUNCTION = "execute"
    CATEGORY = "image/mask"
    OUTPUT_NODE = True  # Превью отображает debug_overlay (или «ничего», если отключено)

    def execute(self, image, mask, offset, debug_enabled, debug_alpha):
        try:
            img = _ensure_single_image(image)
            H, W, _ = img.shape
            m_soft = _ensure_mask(mask, (H, W))  # [H,W] float32, 0..1 с полутоном
            thr = 1 / 255  # или 1/255 если хотите «любой неноль»
            m_bin = (m_soft >= thr).to(torch.float32)  # для bbox

            x, y, w, h = _compute_bbox(m_bin)

            # Маска градиента сохраняется «как есть»
            feather_full = m_soft

            x_exp, y_exp, w_exp, h_exp = _expand_and_clamp_bbox(
                (x, y, w, h), int(offset), W, H
            )

            patch = _crop(img, (x_exp, y_exp, w_exp, h_exp))
            mask_cropped = _crop(feather_full, (x_exp, y_exp, w_exp, h_exp))

            bbox = (int(x_exp), int(y_exp), int(w_exp), int(h_exp))
            crop_w, crop_h = int(w_exp), int(h_exp)

            # ---- Debug overlay ----
            if debug_enabled:
                da = float(max(0.0, min(1.0, debug_alpha)))  # ограничиваем альфу
                red = torch.tensor(
                    [1.0, 0.0, 0.0], device=patch.device, dtype=patch.dtype
                ).view(1, 1, 3)
                alpha = (da * mask_cropped).unsqueeze(-1)  # HxWx1
                overlay = (1.0 - alpha) * patch + alpha * red
                overlay = overlay.clamp(0.0, 1.0)
                debug_img = overlay
            else:
                # «Пустой» оверлей, чтобы превью визуально было пустым
                debug_img = _empty_debug_overlay(device=img.device)

            return (
                image,  # base_image (B x H x W x C)
                _to_comfy_image(patch),  # cropped_image
                _to_comfy_mask(mask_cropped),  # cropped_mask
                bbox,  # bbox
                crop_w,  # crop_width
                crop_h,  # crop_height
                _to_comfy_image(debug_img),  # debug_overlay (для превью)
            )

        except Exception as e:
            msg = f"[image_cut_by_mask] Ошибка: {str(e)}"
            print(msg)
            raise RuntimeError(msg) from e


class ImagePasteByCoords:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "base_image": ("IMAGE", {}),
                "cropped_image": ("IMAGE", {}),
                "cropped_mask": ("MASK", {}),
                "bbox": ("BBOX", {}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "execute"
    CATEGORY = "image/mask"

    def execute(self, base_image, cropped_image, cropped_mask, bbox):
        try:
            base = _ensure_single_image(base_image)
            patch = _ensure_single_image(cropped_image)

            if not isinstance(bbox, (tuple, list)) or len(bbox) != 4:
                raise RuntimeError(
                    f"[bbox] Ожидался кортеж (x, y, w, h), получено: {bbox}"
                )
            x, y, w, h = [int(v) for v in bbox]
            if w <= 0 or h <= 0:
                raise RuntimeError(f"[bbox] Некорректный размер bbox: {(w,h)}")

            H, W, C = base.shape
            _, _, pc = patch.shape
            if pc != C:
                raise RuntimeError(
                    f"[patch] Каналы патча ({pc}) != каналам изображения ({C})."
                )

            if not torch.is_tensor(cropped_mask):
                raise RuntimeError(
                    f"[cropped_mask] Ожидался torch.Tensor, получено: {type(cropped_mask)}"
                )

            if cropped_mask.dim() == 3:  # BxHxW
                if cropped_mask.shape[0] != 1:
                    raise RuntimeError(
                        f"[cropped_mask] Ожидалась одна маска (batch=1), получено: {cropped_mask.shape[0]}"
                    )
                mask = cropped_mask.squeeze(0)
            elif cropped_mask.dim() == 2:  # HxW
                mask = cropped_mask
            else:
                raise RuntimeError(
                    f"[cropped_mask] Ожидался тензор HxW или BxHxW, получено: {tuple(cropped_mask.shape)}"
                )

            mask = mask.to(dtype=torch.float32)

            patch_resized = _resize_hwc(patch, h, w, mode="bilinear")
            mask_resized = _resize_mask_hw(mask, h, w)

            x0 = max(0, x)
            y0 = max(0, y)
            x1 = min(W, x + w)
            y1 = min(H, y + h)

            if x0 >= x1 or y0 >= y1:
                return (_to_comfy_image(base),)

            dx = x0 - x
            dy = y0 - y
            ww = x1 - x0
            hh = y1 - y0

            # Используем уже смасштабированные тензоры
            patch_roi = patch_resized[dy : dy + hh, dx : dx + ww, :]
            mask_roi = mask_resized[dy : dy + hh, dx : dx + ww]
            base_roi = base[y0:y1, x0:x1, :]

            m3 = mask_roi.unsqueeze(-1)
            out_roi = m3 * patch_roi + (1.0 - m3) * base_roi

            out = base.clone()
            out[y0:y1, x0:x1, :] = out_roi

            return (_to_comfy_image(out),)

        except Exception as e:
            msg = f"[image_paste_by_coords] Ошибка: {str(e)}"
            print(msg)
            raise RuntimeError(msg) from e


# ===== РЕГИСТРАЦИЯ =====

NODE_CLASS_MAPPINGS = {
    "ImageCutByMask": ImageCutByMask,
    "ImagePasteByCoords": ImagePasteByCoords,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ImageCutByMask": "✂️ Image Cut By Mask",
    "ImagePasteByCoords": "🩹 Image Paste By Coords",
}
