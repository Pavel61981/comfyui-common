# filename: image_face_detect.py

import os
import urllib.request
from typing import List, Tuple, Optional

import cv2
import numpy as np
import torch


YOLO_FACE_URLS = {
    "yolov12n-face.pt": "https://github.com/YapaLab/yolo-face/releases/download/v0.0.0/yolov12n-face.pt",
    "yolov12s-face.pt": "https://github.com/YapaLab/yolo-face/releases/download/v0.0.0/yolov12s-face.pt",
    "yolov12m-face.pt": "https://github.com/YapaLab/yolo-face/releases/download/v0.0.0/yolov12m-face.pt",
    "yolov12l-face.pt": "https://github.com/YapaLab/yolo-face/releases/download/v0.0.0/yolov12l-face.pt",
}


def yolo_face_models_dir() -> str:
    """
    Путь для весов лиц: <ComfyUI>/models/yolo-face, иначе — локально: ./models/yolo-face
    """
    try:
        from folder_paths import models_dir  # type: ignore

        if isinstance(models_dir, str) and os.path.isdir(models_dir):
            root = os.path.join(models_dir, "yolo-face")
            os.makedirs(root, exist_ok=True)
            return root
    except Exception:
        pass
    here = os.path.dirname(os.path.abspath(__file__))
    root = os.path.join(here, "models", "yolo-face")
    os.makedirs(root, exist_ok=True)
    return root


def _fallback_urls(model_name: str):
    bases = [
        "https://github.com/akanametov/yolo-face/releases/download/v0.0.0",
        "https://raw.githubusercontent.com/YapaLab/yolo-face/refs/heads/main/releases",
    ]
    return [f"{b}/{model_name}" for b in bases]


def _download_file(url: str, dst_path: str, timeout: int = 90) -> None:
    """
    Простой скачиватель с временным .part файлом.
    """
    tmp_path = dst_path + ".part"
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp, open(
            tmp_path, "wb"
        ) as out:
            while True:
                chunk = resp.read(8192)
                if not chunk:
                    break
                out.write(chunk)
        os.replace(tmp_path, dst_path)
    except Exception as e:
        try:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
        finally:
            raise RuntimeError(f"Download failed from {url}: {e}") from e


def _resolve_device(name: str) -> str:
    if name == "auto":
        return "cuda:0" if torch.cuda.is_available() else "cpu"
    return name


def _set_model_precision(yolo_model) -> None:
    try:
        m = getattr(yolo_model, "model", None)
        if m is None:
            return
        m.to(dtype=torch.float32)
    except Exception:
        pass


def _cleanup_model(model, device: str) -> None:
    """
    Аккуратно освобождает память модели (CPU/GPU).
    Без глобальных кэшей.
    """
    try:
        m = getattr(model, "model", None)
        if m is not None:
            try:
                m.to("cpu")
            except Exception:
                pass
    finally:
        try:
            del model
        except Exception:
            pass
        if device.startswith("cuda") and torch.cuda.is_available():
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass


def _clip_int(v: int, lo: int, hi: int) -> int:
    return int(max(lo, min(hi, v)))


def _tensor_to_np_image(tensor: torch.Tensor) -> np.ndarray:
    if not isinstance(tensor, torch.Tensor):
        raise RuntimeError("image must be a torch.Tensor")
    arr = tensor.detach().cpu().numpy()
    if arr.ndim == 4:
        arr = arr[0]
    elif arr.ndim != 3:
        raise RuntimeError(f"Unexpected IMAGE tensor shape: {arr.shape}")
    arr = np.clip(arr, 0.0, 1.0)
    return (arr * 255.0).astype(np.uint8)


def _np_image_to_tensor(img: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(img.astype(np.float32) / 255.0).unsqueeze(0)


def _np_mask_to_tensor(mask_u8: np.ndarray) -> torch.Tensor:
    m = (mask_u8.astype(np.float32) / 255.0)[np.newaxis, ...]
    return torch.from_numpy(m)


def _postprocess_mask(mask_u8: np.ndarray, pad_px: int, blur_r: int) -> np.ndarray:
    """
    Базовая морфология + паддинг маски + опциональный GaussianBlur.
    Порядок: close(3x3 эллипс) -> dilate(3x3 эллипс) -> padding(±px, эллипс ядро) -> blur.
    """
    if mask_u8 is None or mask_u8.size == 0:
        return mask_u8

    try:
        m = mask_u8
        if m.dtype != np.uint8:
            m = m.astype(np.uint8, copy=False)

        kern_base = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, kern_base)
        m = cv2.morphologyEx(m, cv2.MORPH_DILATE, kern_base)

        if isinstance(pad_px, (int, np.integer)) and pad_px != 0:
            k = int(abs(pad_px))
            ksize = max(1, 2 * k + 1)  # нечетный размер
            kern_pad = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
            if pad_px > 0:
                m = cv2.dilate(m, kern_pad, iterations=1)
            else:
                m = cv2.erode(m, kern_pad, iterations=1)

        if isinstance(blur_r, (int, np.integer)) and blur_r > 0:
            ksize = 2 * int(blur_r) + 1
            m = cv2.GaussianBlur(m, (ksize, ksize), 0)

        return m
    except Exception:
        # в случае ошибки вернём изначальную маску
        return mask_u8


def _iou_xywh(a: Tuple[int, int, int, int], b: Tuple[int, int, int, int]) -> float:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    ax2, ay2 = ax + aw, ay + ah
    bx2, by2 = bx + bw, by + bh
    ix1, iy1 = max(ax, bx), max(ay, by)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    union = aw * ah + bw * bh - inter
    return float(inter) / float(union) if union > 0 else 0.0


def _nms_by_iou_xywh(
    boxes: List[Tuple[int, int, int, int]],
    masks: List[np.ndarray],
    iou_thr: float = 0.6,
) -> Tuple[List[Tuple[int, int, int, int]], List[np.ndarray]]:
    """
    Жадный NMS по IoU для XYWH. Критерий — сортировка по убыванию площади.
    """
    if not boxes:
        return boxes, masks
    areas = [b[2] * b[3] for b in boxes]
    order = sorted(range(len(boxes)), key=lambda i: areas[i], reverse=True)
    keep_idx = []
    suppressed = [False] * len(order)
    for oi, i in enumerate(order):
        if suppressed[oi]:
            continue
        keep_idx.append(i)
        bi = boxes[i]
        for oj, j in enumerate(order[oi + 1 :], start=oi + 1):
            if suppressed[oj]:
                continue
            if _iou_xywh(bi, boxes[j]) > iou_thr:
                suppressed[oj] = True
    kept_boxes = [boxes[i] for i in keep_idx]
    kept_masks = [masks[i] for i in keep_idx]
    return kept_boxes, kept_masks


def _masks_to_image_batch(masks_u8: List[np.ndarray], h: int, w: int) -> torch.Tensor:
    """
    Конвертирует список HxW (uint8 0/255) в батч IMAGE: (B,H,W,3) float32 [0..1].
    Если список пуст — возвращает один нулевой кадр.
    """
    if not masks_u8:
        z = np.zeros((1, h, w, 3), dtype=np.float32)
        return torch.from_numpy(z)
    stack = np.stack(
        [(m.astype(np.float32) / 255.0) for m in masks_u8], axis=0
    )  # (B,H,W)
    stack = stack[..., np.newaxis]  # (B,H,W,1)
    stack = np.repeat(stack, 3, axis=3)  # (B,H,W,3)
    return torch.from_numpy(stack)


def _pad_crops_to_batch(
    crops: List[np.ndarray], fallback_h: int, fallback_w: int
) -> torch.Tensor:
    """
    Приводит список цветных кропов (HxWx3, uint8) к батчу IMAGE одинакового размера методом pad-to-max.
    Если список пуст — возвращает один чёрный кадр размера исходника (fallback_h, fallback_w).
    """
    if not crops:
        z = np.zeros((1, fallback_h, fallback_w, 3), dtype=np.float32)
        return torch.from_numpy(z)

    max_h = max(c.shape[0] for c in crops)
    max_w = max(c.shape[1] for c in crops)
    batch = []
    for c in crops:
        h, w = c.shape[:2]
        canvas = np.zeros((max_h, max_w, 3), dtype=np.uint8)
        off_y = (max_h - h) // 2
        off_x = (max_w - w) // 2
        canvas[off_y : off_y + h, off_x : off_x + w, :] = c
        batch.append(canvas.astype(np.float32) / 255.0)
    return torch.from_numpy(np.stack(batch, axis=0))


def _draw_dashed_rect(
    img: np.ndarray,
    pt1: Tuple[int, int],
    pt2: Tuple[int, int],
    color: Tuple[int, int, int],
    thickness: int,
    dash: int = 10,
    gap: int = 6,
) -> None:
    """
    Пунктирная рамка по периметру прямоугольника.
    """
    x1, y1 = pt1
    x2, y2 = pt2

    def _draw_segment(p0, p1):
        cv2.line(img, p0, p1, color, thickness, lineType=cv2.LINE_8)

    # Верх/низ
    x = x1
    while x < x2:
        x_end = min(x + dash, x2)
        _draw_segment((x, y1), (x_end, y1))
        _draw_segment((x, y2), (x_end, y2))
        x += dash + gap
    # Лево/право
    y = y1
    while y < y2:
        y_end = min(y + dash, y2)
        _draw_segment((x1, y), (x1, y_end))
        _draw_segment((x2, y), (x2, y_end))
        y += dash + gap


class ImageFaceDetect:
    """
    Детекция лиц (YOLO-face).
    Режимы:
      • BODY_BBOXES непустой → одно лицо на тело (самое крупное в ROI), глобальный NMS.
      • BODY_BBOXES пустой/нет → детекция по всему кадру, берём все лица (после порога площади и NMS).
    Маска лица — заполненный овал по bbox. Дополнительно: кропы лиц и общий кроп всех лиц с паддингом.

    Параметр fp16_infer удалён. Добавлен параметр mask_padding_px.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
            },
            "optional": {
                "BODY_BBOXES": ("BBOX_LIST",),
                "device": (["auto", "cpu", "cuda:0"], {"default": "auto"}),
                "conf": (
                    "FLOAT",
                    {"default": 0.25, "min": 0.0, "max": 1.0, "step": 0.01},
                ),
                "mask_blur_radius": (
                    "INT",
                    {"default": 0, "min": 0, "max": 50, "step": 1},
                ),
                "mask_padding_px": (
                    "INT",
                    {"default": 0, "min": -256, "max": 256, "step": 1},
                ),
                "debug_bbox_thickness": (
                    "INT",
                    {"default": 6, "min": 1, "max": 10, "step": 1},
                ),
                "face_model": (
                    [
                        "yolov12n-face.pt",
                        "yolov12s-face.pt",
                        "yolov12m-face.pt",
                        "yolov12l-face.pt",
                    ],
                    {"default": "yolov12s-face.pt"},
                ),
                "face_min_component_percent": (
                    "FLOAT",
                    {"default": 1.0, "min": 0.0, "max": 5.0, "step": 0.01},
                ),
                "face_roi_pad_percent": (
                    "FLOAT",
                    {"default": 10.0, "min": 0.0, "max": 50.0, "step": 1.0},
                ),
                "face_crop_pad_percent": (
                    "FLOAT",
                    {"default": 10.0, "min": 0.0, "max": 50.0, "step": 1.0},
                ),
            },
        }

    RETURN_TYPES = (
        "IMAGE",  # IMAGE (исходник)
        "BBOX_LIST",  # FACE_BBOXES
        "IMAGE",  # FACE_MASKS (батч IMAGE: B×H×W×3)
        "BOOLEAN",  # FOUND_FACE
        "IMAGE",  # DEBUG_IMAGE
        "IMAGE",  # FACE_CROPS (батч кропов лиц, pad-to-max)
        "IMAGE",  # ALL_FACES_CROP (один общий кроп)
    )
    RETURN_NAMES = (
        "IMAGE",
        "FACE_BBOXES",
        "FACE_MASKS",
        "FOUND_FACE",
        "DEBUG_IMAGE",
        "FACE_CROPS",
        "ALL_FACES_CROP",
    )
    FUNCTION = "execute"
    CATEGORY = "Masquerade/Detect"
    OUTPUT_NODE = False

    def _get_ultra(self):
        try:
            from ultralytics import YOLO  # type: ignore
        except Exception as e:
            raise RuntimeError(
                f"[ImageFaceDetect] ultralytics not installed: {e}"
            ) from e
        return YOLO

    def _ensure_face_weight_file(self, model_name: str) -> str:
        """
        Проверяет наличие весов; при отсутствии — скачивает из известных источников.
        """
        face_dir = yolo_face_models_dir()
        dst = os.path.join(face_dir, model_name)
        if os.path.isfile(dst):
            return dst

        urls = []
        if model_name in YOLO_FACE_URLS:
            urls.append(YOLO_FACE_URLS[model_name])
        urls.extend(_fallback_urls(model_name))

        last_err = None
        for url in urls:
            try:
                _download_file(url, dst, timeout=90)
                if os.path.isfile(dst):
                    return dst
            except Exception as e:
                last_err = e
                continue

        raise RuntimeError(
            f"[ImageFaceDetect] Failed to download face weights '{model_name}'. "
            f"Tried URLs: {urls}. Error: {last_err}"
        )

    def _load_face_model(self, model_name: str, device: str):
        """
        Загружает YOLO-face модель без изменения точности (dtype).
        """
        YOLO = self._get_ultra()
        local_path = self._ensure_face_weight_file(model_name)
        try:
            model = YOLO(local_path)
            model.to(device)
            _set_model_precision(model)
            return model
        except Exception as e:
            raise RuntimeError(
                f"[ImageFaceDetect] Failed to load face model '{model_name}' from '{local_path}': {e}"
            ) from e

    def _detect_faces_for_bodies(
        self,
        img: np.ndarray,
        body_bboxes: List[Tuple[int, int, int, int]],
        model,
        conf: float,
        face_min_component_percent: float,
        mask_padding_px: int,
        mask_blur_radius: int,
        pad_percent: float,
    ) -> Tuple[
        List[np.ndarray],
        List[Tuple[int, int, int, int]],
        List[Tuple[int, int, int, int]],
    ]:
        """
        Детекция лиц в ROI вокруг каждого тела.
        Порог минимальной площади лица считается как
        (face_min_component_percent/100) * площадь bbox ТЕЛА (bw*bh), БЕЗ учёта ROI-паддинга.

        Возвращает тройку:
          (accepted_masks_u8, accepted_bboxes_xywh, rejected_bboxes_xywh)
        где mask — заполненный овал по bbox лица, прошедший постобработку.
        """
        H, W = img.shape[:2]
        if not body_bboxes:
            return [], [], []

        # Подготовка ROI по телам (+процентный pad только для детекции),
        # и параллельный список площадей тел (bw*bh) для порогов.
        rois: List[Optional[np.ndarray]] = []
        roi_rects: List[Tuple[int, int, int, int]] = []
        roi_body_areas: List[float] = []

        for bx, by, bw, bh in body_bboxes:
            body_area = float(max(0, bw) * max(0, bh))
            pad = int(
                round(
                    min(max(0, bw), max(0, bh)) * max(0.0, float(pad_percent)) / 100.0
                )
            )

            rx1 = _clip_int(bx - pad, 0, W - 1)
            ry1 = _clip_int(by - pad, 0, H - 1)
            rx2 = _clip_int(bx + bw + pad, 0, W)
            ry2 = _clip_int(by + bh + pad, 0, H)

            if rx2 <= rx1 or ry2 <= ry1:
                rois.append(None)
                roi_rects.append((0, 0, 0, 0))
                roi_body_areas.append(0.0)
                continue

            rois.append(img[ry1:ry2, rx1:rx2, :])
            roi_rects.append((rx1, ry1, rx2, ry2))
            roi_body_areas.append(body_area)

        batched_inputs = [r for r in rois if r is not None]
        if not batched_inputs:
            return [], [], []

        try:
            results = model.predict(
                source=batched_inputs,
                conf=conf,
                iou=0.45,
                max_det=50,
                verbose=False,
                device=None,
            )
        except Exception as e:
            raise RuntimeError(
                f"[ImageFaceDetect] YOLO face predict failed: {e}"
            ) from e

        accepted_masks: List[np.ndarray] = []
        accepted_bboxes: List[Tuple[int, int, int, int]] = []
        rejected_bboxes: List[Tuple[int, int, int, int]] = []

        res_iter = iter(results if isinstance(results, list) else [results])

        # Для каждого ROI считаем свой порог от площади ТЕЛА (bw*bh)
        pct = max(0.0, float(face_min_component_percent)) / 100.0

        for roi, (rx1, ry1, rx2, ry2), body_area in zip(
            rois, roi_rects, roi_body_areas
        ):
            if roi is None:
                continue

            fr = next(res_iter, None)
            if fr is None:
                continue

            # boxes в координатах ROI → перевод в координаты полного кадра
            try:
                f_xyxy = (
                    fr.boxes.xyxy.detach().cpu().numpy().astype(np.float32)
                    if getattr(fr, "boxes", None) is not None
                    else np.zeros((0, 4), dtype=np.float32)
                )
            except Exception:
                f_xyxy = np.zeros((0, 4), dtype=np.float32)

            thr_area_body = pct * body_area  # порог для этого тела

            # Выбираем ровно одно лицо на тело — с наибольшей площадью bbox,
            # но только если оно >= thr_area_body
            best = None  # (area, (x,y,w,h))
            for i in range(f_xyxy.shape[0]):
                fx1f, fy1f, fx2f, fy2f = f_xyxy[i]
                fx1 = _clip_int(int(round(rx1 + fx1f)), 0, W - 1)
                fy1 = _clip_int(int(round(ry1 + fy1f)), 0, H - 1)
                fx2 = _clip_int(int(round(rx1 + fx2f)), 0, W)
                fy2 = _clip_int(int(round(ry1 + fy2f)), 0, H)
                fw, fh = max(0, fx2 - fx1), max(0, fy2 - fy1)
                if fw <= 0 or fh <= 0:
                    continue

                fbbox = (fx1, fy1, fw, fh)
                a = float(fw * fh)
                if a < thr_area_body:
                    rejected_bboxes.append(fbbox)
                    continue
                if (best is None) or (a > best[0]):
                    best = (a, fbbox)

            if best is None:
                continue

            _, fbbox = best
            fx1, fy1, fw, fh = fbbox

            # Маска — заполненный овал по bbox (fallback: прямоугольник)
            full = np.zeros((H, W), dtype=np.uint8)
            cx = int(round(fx1 + fw / 2.0))
            cy = int(round(fy1 + fh / 2.0))
            ax = max(1, int(round(fw / 2.0)))
            ay = max(1, int(round(fh / 2.0)))
            try:
                cv2.ellipse(full, (cx, cy), (ax, ay), 0, 0, 360, 255, -1)
                raw_mask = full
            except Exception:
                raw_mask = np.zeros((H, W), dtype=np.uint8)
                cv2.rectangle(
                    raw_mask, (fx1, fy1), (fx1 + fw, fy1 + fh), 255, thickness=-1
                )

            proc_mask = _postprocess_mask(
                raw_mask, pad_px=mask_padding_px, blur_r=mask_blur_radius
            )
            accepted_masks.append(proc_mask)
            accepted_bboxes.append(fbbox)

        # Удаляем возможные дубли между пересекающимися ROI (глобально)
        accepted_bboxes, accepted_masks = _nms_by_iou_xywh(
            accepted_bboxes, accepted_masks, iou_thr=0.6
        )

        # Сортировка слева→направо
        order = sorted(range(len(accepted_bboxes)), key=lambda i: accepted_bboxes[i][0])
        accepted_masks = [accepted_masks[i] for i in order]
        accepted_bboxes = [accepted_bboxes[i] for i in order]

        return accepted_masks, accepted_bboxes, rejected_bboxes

    def _detect_faces_global(
        self,
        img: np.ndarray,
        model,
        conf: float,
        thr_area: float,
        mask_padding_px: int,
        mask_blur_radius: int,
    ) -> Tuple[
        List[np.ndarray],
        List[Tuple[int, int, int, int]],
        List[Tuple[int, int, int, int]],
    ]:
        """
        Детекция по всему изображению. Возвращает все лица, прошедшие порог площади и NMS.
        """
        H, W = img.shape[:2]
        try:
            results = model.predict(
                source=[img],
                conf=conf,
                iou=0.45,
                max_det=200,
                verbose=False,
                device=None,
            )
        except Exception as e:
            raise RuntimeError(
                f"[ImageFaceDetect] YOLO face predict failed: {e}"
            ) from e

        accepted_masks: List[np.ndarray] = []
        accepted_bboxes: List[Tuple[int, int, int, int]] = []
        rejected_bboxes: List[Tuple[int, int, int, int]] = []

        fr = results[0] if isinstance(results, list) and len(results) > 0 else results
        try:
            f_xyxy = (
                fr.boxes.xyxy.detach().cpu().numpy().astype(np.float32)
                if getattr(fr, "boxes", None) is not None
                else np.zeros((0, 4), dtype=np.float32)
            )
        except Exception:
            f_xyxy = np.zeros((0, 4), dtype=np.float32)

        for i in range(f_xyxy.shape[0]):
            x1f, y1f, x2f, y2f = f_xyxy[i]
            x1 = _clip_int(int(round(x1f)), 0, W - 1)
            y1 = _clip_int(int(round(y1f)), 0, H - 1)
            x2 = _clip_int(int(round(x2f)), 0, W)
            y2 = _clip_int(int(round(y2f)), 0, H)
            w = max(0, x2 - x1)
            h = max(0, y2 - y1)
            if w <= 0 or h <= 0:
                continue
            bbox = (x1, y1, w, h)
            area = float(w * h)
            if area < thr_area:
                rejected_bboxes.append(bbox)
                continue

            full = np.zeros((H, W), dtype=np.uint8)
            cx = int(round(x1 + w / 2.0))
            cy = int(round(y1 + h / 2.0))
            ax = max(1, int(round(w / 2.0)))
            ay = max(1, int(round(h / 2.0)))
            try:
                cv2.ellipse(full, (cx, cy), (ax, ay), 0, 0, 360, 255, -1)
                raw_mask = full
            except Exception:
                raw_mask = np.zeros((H, W), dtype=np.uint8)
                cv2.rectangle(raw_mask, (x1, y1), (x1 + w, y1 + h), 255, thickness=-1)

            proc_mask = _postprocess_mask(
                raw_mask, pad_px=mask_padding_px, blur_r=mask_blur_radius
            )
            accepted_masks.append(proc_mask)
            accepted_bboxes.append(bbox)

        # Глобальный NMS
        accepted_bboxes, accepted_masks = _nms_by_iou_xywh(
            accepted_bboxes, accepted_masks, iou_thr=0.6
        )

        # Сортировка слева→направо
        order = sorted(range(len(accepted_bboxes)), key=lambda i: accepted_bboxes[i][0])
        accepted_masks = [accepted_masks[i] for i in order]
        accepted_bboxes = [accepted_bboxes[i] for i in order]

        return accepted_masks, accepted_bboxes, rejected_bboxes

    def _compute_padded_rect(
        self, bbox: Tuple[int, int, int, int], pad_percent: float, W: int, H: int
    ) -> Tuple[int, int, int, int]:
        """
        Возвращает паддинговый прямоугольник (x,y,w,h), ограниченный границами изображения.
        """
        x, y, w, h = bbox
        pad_x = int(round(max(0.0, float(pad_percent)) * w / 100.0))
        pad_y = int(round(max(0.0, float(pad_percent)) * h / 100.0))
        x1 = _clip_int(x - pad_x, 0, W - 1)
        y1 = _clip_int(y - pad_y, 0, H - 1)
        x2 = _clip_int(x + w + pad_x, 0, W)
        y2 = _clip_int(y + h + pad_y, 0, H)
        nx, ny = x1, y1
        nw, nh = max(0, x2 - x1), max(0, y2 - y1)
        return nx, ny, nw, nh

    def execute(
        self,
        image: torch.Tensor,
        BODY_BBOXES: Optional[List[Tuple[int, int, int, int]]] = None,
        device: str = "auto",
        conf: float = 0.25,
        mask_blur_radius: int = 0,
        mask_padding_px: int = 0,
        debug_bbox_thickness: int = 6,
        face_model: str = "yolov12s-face.pt",
        face_min_component_percent: float = 1.0,
        face_roi_pad_percent: float = 10.0,
        face_crop_pad_percent: float = 10.0,
    ):
        # --- вход ---
        try:
            img = _tensor_to_np_image(image)  # uint8 HxWx3
        except Exception as e:
            raise RuntimeError(f"[ImageFaceDetect] Invalid IMAGE tensor: {e}") from e

        try:
            body_bboxes_xywh = []
            if BODY_BBOXES:
                body_bboxes_xywh = [
                    (int(x), int(y), int(w), int(h)) for (x, y, w, h) in BODY_BBOXES
                ]
        except Exception as e:
            raise RuntimeError(f"[ImageFaceDetect] Invalid BODY_BBOXES: {e}") from e

        H, W = img.shape[:2]
        frame_area = float(W * H)
        thr_face_area = max(0.0, float(face_min_component_percent)) / 100.0 * frame_area

        # --- девайс ---
        dev = _resolve_device(device)

        # --- модель лиц ---
        model_face = self._load_face_model(face_model, dev)

        # --- детекция ---
        try:
            if body_bboxes_xywh:
                face_masks_u8, face_bboxes_xywh, rejected_face_bboxes = (
                    self._detect_faces_for_bodies(
                        img=img,
                        body_bboxes=body_bboxes_xywh,
                        model=model_face,
                        conf=conf,
                        face_min_component_percent=face_min_component_percent,
                        mask_padding_px=mask_padding_px,
                        mask_blur_radius=mask_blur_radius,
                        pad_percent=face_roi_pad_percent,
                    )
                )
            else:
                face_masks_u8, face_bboxes_xywh, rejected_face_bboxes = (
                    self._detect_faces_global(
                        img=img,
                        model=model_face,
                        conf=conf,
                        thr_area=thr_face_area,
                        mask_padding_px=mask_padding_px,
                        mask_blur_radius=mask_blur_radius,
                    )
                )
        finally:
            _cleanup_model(model_face, dev)

        found_face = len(face_bboxes_xywh) > 0

        # --- конвертация масок в батч IMAGE ---
        face_masks_batch = _masks_to_image_batch(face_masks_u8, H, W)
        face_bboxes_list = [
            [int(x), int(y), int(w), int(h)] for (x, y, w, h) in face_bboxes_xywh
        ]

        # --- построение паддинговых кропов для лиц ---
        padded_rects = []
        face_crops_np: List[np.ndarray] = []
        if found_face:
            for x, y, w, h in face_bboxes_xywh:
                px, py, pw, ph = self._compute_padded_rect(
                    (x, y, w, h), face_crop_pad_percent, W, H
                )
                if pw > 0 and ph > 0:
                    crop = img[py : py + ph, px : px + pw, :].copy()
                    face_crops_np.append(crop)
                padded_rects.append((px, py, pw, ph))
        # батч кропов (pad-to-max)
        face_crops_batch = _pad_crops_to_batch(face_crops_np, H, W)

        # --- общий кроп всех лиц ---
        if found_face and padded_rects:
            xs1 = [r[0] for r in padded_rects]
            ys1 = [r[1] for r in padded_rects]
            xs2 = [r[0] + r[2] for r in padded_rects]
            ys2 = [r[1] + r[3] for r in padded_rects]
            ux1 = int(max(0, min(xs1)))
            uy1 = int(max(0, min(ys1)))
            ux2 = int(min(W, max(xs2)))
            uy2 = int(min(H, max(ys2)))
            if ux2 > ux1 and uy2 > uy1:
                all_faces_crop_np = img[uy1:uy2, ux1:ux2, :].copy()
            else:
                all_faces_crop_np = img.copy()
        else:
            all_faces_crop_np = img.copy()
        all_faces_crop_tensor = _np_image_to_tensor(all_faces_crop_np)

        # --- DEBUG ---
        debug = img.copy()
        overlay = np.zeros_like(debug)

        if face_masks_u8:
            mf = np.any(np.stack([m > 0 for m in face_masks_u8], axis=0), axis=0)
            overlay[mf] = [0, 255, 0]  # зелёный

        debug = cv2.addWeighted(overlay, 0.35, debug, 0.65, 0)

        t = int(max(1, min(10, debug_bbox_thickness)))
        # входные BODY_BBOXES — жёлтые (контекст)
        for x, y, w_, h_ in body_bboxes_xywh:
            x2, y2 = x + w_, y + h_
            cv2.rectangle(debug, (x, y), (x2, y2), (0, 255, 255), thickness=t)

        # лица — бирюзовые
        for i, (x, y, w_, h_) in enumerate(face_bboxes_xywh):
            x2, y2 = x + w_, y + h_
            cv2.rectangle(debug, (x, y), (x2, y2), (255, 255, 0), thickness=t)
            cv2.putText(
                debug,
                f"Face #{i}",
                (x + 5, max(15, y + 15)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 0),
                1,
                cv2.LINE_AA,
            )

        # отклонённые лица — красные перечёркнутые (только те, что меньше порога)
        for x, y, w_, h_ in rejected_face_bboxes:
            x2, y2 = x + w_, y + h_
            cv2.rectangle(debug, (x, y), (x2, y2), (0, 0, 255), thickness=t)
            cv2.line(debug, (x, y), (x2, y2), (0, 0, 255), thickness=t)
            cv2.line(debug, (x2, y), (x, y2), (0, 0, 255), thickness=t)

        # паддинговые bbox для лиц — пурпурные
        PURPLE = (255, 0, 255)
        if padded_rects:
            for px, py, pw, ph in padded_rects:
                if pw > 0 and ph > 0:
                    cv2.rectangle(
                        debug,
                        (px, py),
                        (px + pw, py + ph),
                        PURPLE,
                        thickness=max(1, t - 2),
                    )

            # общий bbox — белая пунктирная рамка
            xs1 = [r[0] for r in padded_rects]
            ys1 = [r[1] for r in padded_rects]
            xs2 = [r[0] + r[2] for r in padded_rects]
            ys2 = [r[1] + r[3] for r in padded_rects]
            ux1 = int(max(0, min(xs1)))
            uy1 = int(max(0, min(ys1)))
            ux2 = int(min(W, max(xs2)))
            uy2 = int(min(H, max(ys2)))
            if ux2 > ux1 and uy2 > uy1:
                _draw_dashed_rect(
                    debug,
                    (ux1, uy1),
                    (ux2, uy2),
                    (255, 255, 255),
                    max(1, t - 3),
                    dash=10,
                    gap=6,
                )
                cv2.putText(
                    debug,
                    "ALL_FACES",
                    (ux1 + 5, max(uy1 + 15, 15)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 255),
                    1,
                    cv2.LINE_AA,
                )

        return (
            _np_image_to_tensor(img),  # IMAGE
            face_bboxes_list,  # FACE_BBOXES
            face_masks_batch,  # FACE_MASKS (IMAGE батч)
            bool(found_face),  # FOUND_FACE
            _np_image_to_tensor(debug),  # DEBUG_IMAGE
            face_crops_batch,  # FACE_CROPS (батч)
            all_faces_crop_tensor,  # ALL_FACES_CROP (один кадр)
        )


# Регистрация в ComfyUI
NODE_CLASS_MAPPINGS = {"ImageFaceDetect": ImageFaceDetect}
NODE_DISPLAY_NAME_MAPPINGS = {"ImageFaceDetect": "Image Face Detect"}
