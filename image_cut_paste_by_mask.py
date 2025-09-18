from typing import Tuple

import torch
import torch.nn.functional as F


# ===== ВСПОМОГАТЕЛЬНЫЕ =====

def _ensure_single_image(image: torch.Tensor):
    """
    Приводит вход IMAGE к одному тензору HxWxC (float32, [0,1]).
    Поддерживаемые входные форматы:
      - [B,H,W,C] (B==1)
      - [H,W,C]
      - [ [H,W,C] ] список из одного элемента
    """
    # Случай списка [HxWxC]
    if isinstance(image, list):
        if len(image) != 1:
            raise RuntimeError(f"[image] Ожидался один кадр (batch=1), получено: {len(image)}")
        img = image[0]
    else:
        img = image

    if not torch.is_tensor(img):
        raise RuntimeError("[image] Ожидался torch.Tensor")

    if img.dim() == 4:
        # [B,H,W,C]
        if img.shape[0] != 1:
            raise RuntimeError(f"[image] Ожидался batch=1, получено: {img.shape[0]}")
        img = img[0]
    elif img.dim() != 3:
        raise RuntimeError(f"[image] Ожидался тензор [H,W,C] или [1,H,W,C], получено: {tuple(img.shape)}")

    if img.shape[2] not in (1, 3, 4):
        raise RuntimeError(f"[image] Ожидался [H,W,C] с C∈{{1,3,4}}, получено C={img.shape[2]}")

    return img.to(dtype=torch.float32)


def _ensure_mask(mask: torch.Tensor, target_hw: Tuple[int, int], device: torch.device):
    """
    Приводит MASK к HxW (float32, [0,1]) на указанном device и сверяет размер.
    """
    if not torch.is_tensor(mask):
        raise RuntimeError("[mask] Ожидался torch.Tensor")
    if mask.dim() != 2:
        raise RuntimeError(f"[mask] Ожидался тензор [H,W], получено: {tuple(mask.shape)}")
    h, w = target_hw
    if mask.shape[0] != h or mask.shape[1] != w:
        raise RuntimeError(
            f"[mask] Размер маски {tuple(mask.shape)} не совпадает с изображением {(h, w)}"
        )
    return mask.clamp(0.0, 1.0).to(device=device, dtype=torch.float32)


def _to_bhwc(img_hwc: torch.Tensor) -> torch.Tensor:
    """HxWxC -> [1,H,W,C] для порта IMAGE."""
    if img_hwc.dim() != 3:
        raise RuntimeError(f"[image] Ожидался [H,W,C], получено: {tuple(img_hwc.shape)}")
    return img_hwc.unsqueeze(0).contiguous()


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


def _resize_hwc(img: torch.Tensor, h: int, w: int) -> torch.Tensor:
    """
    Масштабирует [H,W,C] к (h,w) (bilinear, align_corners=False). Возвращает [H,W,C].
    """
    if img.shape[0] == h and img.shape[1] == w:
        return img
    # HWC -> NCHW
    x = img.permute(2, 0, 1).unsqueeze(0)
    x = F.interpolate(x, size=(h, w), mode="bilinear", align_corners=False)
    x = x.squeeze(0).permute(1, 2, 0).contiguous()
    return x


def _resize_mask_hw(mask: torch.Tensor, h: int, w: int) -> torch.Tensor:
    """
    Масштабирует [H,W] к (h,w), bilinear, затем клипуется в [0,1].
    """
    if mask.shape[0] == h and mask.shape[1] == w:
        return mask
    x = mask.unsqueeze(0).unsqueeze(0)  # [1,1,H,W]
    x = F.interpolate(x, size=(h, w), mode="bilinear", align_corners=False)
    x = x.squeeze(0).squeeze(0)
    return x.clamp(0.0, 1.0)


def _distance_transform_cityblock_inside(mask_bin: torch.Tensor) -> torch.Tensor:
    """
    Distance transform внутрь маски (манхэттен). На выходе расстояние (px)
    от каждого пикселя ВНУТРИ маски до ближайшего пикселя ВНЕ маски.
    Вне маски — нули.
    Реализация на CPU, 2 прохода O(H*W). Возвращает на исходное устройство.
    """
    device = mask_bin.device
    cpu = mask_bin.detach().to("cpu")
    H, W = cpu.shape
    inf = H + W + 5
    dist = torch.full((H, W), inf, dtype=torch.int32)
    outside = cpu <= 0
    dist[outside] = 0
    # forward
    for y in range(H):
        row = dist[y]
        for x in range(W):
            if row[x] == 0:
                continue
            v = int(row[x])
            if x > 0:
                v = min(v, int(row[x - 1]) + 1)
            if y > 0:
                v = min(v, int(dist[y - 1, x]) + 1)
            row[x] = v
    # backward
    for y in range(H - 1, -1, -1):
        row = dist[y]
        for x in range(W - 1, -1, -1):
            v = int(row[x])
            if x + 1 < W:
                v = min(v, int(row[x + 1]) + 1)
            if y + 1 < H:
                v = min(v, int(dist[y + 1, x]) + 1)
            row[x] = v
    dist = dist.to(dtype=torch.float32)
    # Обнулим вне маски (там и так 0), внутри оставим расстояния
    dist = dist * (cpu > 0).to(torch.float32)
    return dist.to(device)


# ===== НОДЫ =====

class ImageCutByMask:
    """
    Нода "image_cut_by_mask" — вырезает прямоугольную область по маске.
    - Перо (feather) рассчитывается ВНУТРЬ маски, величина задаётся в процентах
      от min(w,h) bbox, где bbox взят ДО расширения offset.
    - Возвращает: оригинал (BHWC), патч (BHWC), обрезанную маску (HW),
      полноразмерную маску (HW), bbox (x,y,w,h).
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE", {}),
                "mask": ("MASK", {}),
                "offset": ("INT", {"default": 8, "min": 0, "max": 4096, "step": 1}),
                "feather_percent": ("FLOAT", {"default": 5.0, "min": 0.0, "max": 100.0, "step": 0.1}),
            }
        }

    RETURN_TYPES = ("IMAGE", "IMAGE", "MASK", "MASK", "BBOX")
    FUNCTION = "execute"
    CATEGORY = "image/mask"
    OUTPUT_NODE = False

    def execute(self, image, mask, offset, feather_percent):
        try:
            # Приведение изображения к HWC и фиксация устройства
            img = _ensure_single_image(image)
            H, W, _ = img.shape
            device = img.device

            # Маска -> HW на device изображения
            m = _ensure_mask(mask, (H, W), device=device)
            m_bin = (m > 0.0).to(torch.float32)

            # 1) bbox по маске
            x, y, w, h = _compute_bbox(m_bin)

            # 2) feather внутрь по исходному bbox
            min_side = max(1, min(w, h))
            feather_px = int(round(min_side * max(0.0, float(feather_percent)) / 100.0))
            if feather_px <= 0:
                feather_full = m_bin  # без пера
            else:
                dist_in = _distance_transform_cityblock_inside(m_bin)
                # 0 на границе, 1 глубже чем feather_px
                feather_full = ((dist_in - 1.0) / float(feather_px)).clamp(0.0, 1.0)
                # строго внутри
                feather_full = feather_full * m_bin

            # 3) расширяем bbox наружу
            x, y, w, h = _expand_and_clamp_bbox((x, y, w, h), int(offset), W, H)

            # 4) кропы
            patch = _crop(img, (x, y, w, h))
            mask_cropped = _crop(feather_full, (x, y, w, h))

            bbox = (int(x), int(y), int(w), int(h))

            # Выходы как BHWC/маски HW
            img_bhwc = _to_bhwc(img)
            patch_bhwc = _to_bhwc(patch)

            return (img_bhwc, patch_bhwc, mask_cropped, feather_full, bbox)

        except Exception as e:
            msg = f"[image_cut_by_mask] Ошибка: {str(e)}"
            print(msg)
            raise RuntimeError(msg) from e


class ImagePasteByCoords:
    """
    Нода "image_paste_by_coords" — вклеивает патч в исходное изображение по bbox,
    альфа-смешивание по предоставленной обрезанной маске.
    - Если размер патча/маски не совпадает с (w,h) bbox — автоматически масштабирует к (w,h).
    - Поддерживает подрезку при частичном выходе bbox за границы.
    - Выход: IMAGE в формате [1,H,W,C].
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "base_image": ("IMAGE", {}),
                "patch_image": ("IMAGE", {}),
                "cropped_mask": ("MASK", {}),
                "bbox": ("BBOX", {}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "execute"
    CATEGORY = "image/mask"
    OUTPUT_NODE = False

    def execute(self, base_image, patch_image, cropped_mask, bbox):
        try:
            base = _ensure_single_image(base_image)  # HWC
            patch = _ensure_single_image(patch_image)  # HWC

            if not isinstance(bbox, tuple) or len(bbox) != 4:
                raise RuntimeError("[bbox] Ожидался кортеж (x, y, w, h).")
            x, y, w, h = [int(v) for v in bbox]
            if w <= 0 or h <= 0:
                raise RuntimeError("[bbox] Нулевой размер bbox.")

            # Приведение всех тензоров к device базового изображения
            device = base.device
            patch = patch.to(device=device)

            H, W, C = base.shape
            ph, pw, pc = patch.shape
            if pc != C:
                raise RuntimeError(
                    f"[patch] Каналы патча ({pc}) != каналам изображения ({C})."
                )

            mask = _ensure_mask(cropped_mask, (ph, pw), device=device)

            # Приведение к размеру bbox (если нужно)
            if (pw != w) or (ph != h):
                patch = _resize_hwc(patch, h, w)
                mask = _resize_mask_hw(mask, h, w)
                ph, pw = h, w

            # Клип bbox по границам base
            x0 = max(0, x)
            y0 = max(0, y)
            x1 = min(W, x + w)
            y1 = min(H, y + h)

            if x0 >= x1 or y0 >= y1:
                raise RuntimeError("[bbox] BBox вне изображения или имеет нулевой размер.")

            # ROI в патче/маске (если bbox подрезан)
            dx = x0 - x
            dy = y0 - y
            ww = x1 - x0
            hh = y1 - y0

            patch_roi = patch[dy : dy + hh, dx : dx + ww, :]
            mask_roi = mask[dy : dy + hh, dx : dx + ww].clamp(0.0, 1.0)
            base_roi = base[y0:y1, x0:x1, :]

            # Альфа-композит
            m3 = mask_roi.unsqueeze(-1)  # HxWx1
            out_roi = m3 * patch_roi + (1.0 - m3) * base_roi

            # Сборка
            out = base.clone()
            out[y0:y1, x0:x1, :] = out_roi

            return (_to_bhwc(out),)

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
    "ImageCutByMask": "✂️ image_cut_by_mask",
    "ImagePasteByCoords": "🩹 image_paste_by_coords",
}
