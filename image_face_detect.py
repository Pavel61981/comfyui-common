# filename: image_face_detect.py

import os
import urllib.request
from typing import List, Tuple

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


def _prefer_precision(fp16_infer: bool, device: str) -> str:
    if not fp16_infer:
        return "fp32"
    if device.startswith("cuda"):
        try:
            if (
                hasattr(torch.cuda, "is_bf16_supported")
                and torch.cuda.is_bf16_supported()
            ):
                return "bf16"
        except Exception:
            pass
        return "fp16"
    return "fp32"


def _set_model_precision(yolo_model, precision: str) -> None:
    try:
        m = getattr(yolo_model, "model", None)
        if m is None:
            return
        if precision == "bf16":
            m.to(dtype=torch.bfloat16)
        elif precision == "fp16":
            m.to(dtype=torch.float16)
        else:
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


def _postprocess_mask(mask_u8: np.ndarray, blur_r: int) -> np.ndarray:
    if mask_u8 is None or mask_u8.size == 0:
        return mask_u8
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    m = cv2.morphologyEx(mask_u8, cv2.MORPH_CLOSE, kernel)
    m = cv2.morphologyEx(m, cv2.MORPH_DILATE, kernel)
    if blur_r > 0:
        m = cv2.GaussianBlur(m, (blur_r * 2 + 1, blur_r * 2 + 1), 0)
    return m


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


class ImageFaceDetect:
    """
    Детекция лиц в ROI тел (YOLO-face).
    Для каждого тела -> одно лицо (наибольшая площадь), NMS глобально.
    Маска лица всегда создаётся как заполненный овал, вписанный в bbox лица.
    Без кэширования моделей; после использования — очистка памяти.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "BODY_BBOXES": ("BBOX_LIST",),
            },
            "optional": {
                "device": (["auto", "cpu", "cuda:0"], {"default": "auto"}),
                "conf": (
                    "FLOAT",
                    {"default": 0.25, "min": 0.0, "max": 1.0, "step": 0.01},
                ),
                "mask_blur_radius": (
                    "INT",
                    {"default": 0, "min": 0, "max": 50, "step": 1},
                ),
                "debug_bbox_thickness": (
                    "INT",
                    {"default": 6, "min": 1, "max": 10, "step": 1},
                ),
                "fp16_infer": ("BOOLEAN", {"default": False}),
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
            },
        }

    RETURN_TYPES = (
        "IMAGE",  # IMAGE (исходник)
        "BBOX_LIST",  # FACE_BBOXES
        "IMAGE",  # FACE_MASKS (батч IMAGE: B×H×W×3)
        "BOOLEAN",  # FOUND_FACE
        "IMAGE",  # DEBUG_IMAGE
    )
    RETURN_NAMES = (
        "IMAGE",
        "FACE_BBOXES",
        "FACE_MASKS",
        "FOUND_FACE",
        "DEBUG_IMAGE",
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

    def _load_face_model(self, model_name: str, device: str, precision: str):
        YOLO = self._get_ultra()
        local_path = self._ensure_face_weight_file(model_name)
        try:
            model = YOLO(local_path)
            model.to(device)
            _set_model_precision(model, precision)
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
        thr_area: float,
        mask_blur_radius: int,
        pad_percent: float,
    ) -> Tuple[
        List[np.ndarray],
        List[Tuple[int, int, int, int]],
        List[Tuple[int, int, int, int]],
    ]:
        """
        Возвращает (accepted_masks_u8, accepted_bboxes_xywh, rejected_bboxes_xywh)
        Для каждого тела выбирается ровно ОДНО лицо — с наибольшей площадью bbox.
        Маска всегда — заполненный овал по bbox.
        """
        H, W = img.shape[:2]
        if not body_bboxes:
            return [], [], []

        rois, roi_rects = [], []
        for bx, by, bw, bh in body_bboxes:
            pad = int(round(min(bw, bh) * max(0.0, float(pad_percent)) / 100.0))
            rx1 = _clip_int(bx - pad, 0, W - 1)
            ry1 = _clip_int(by - pad, 0, H - 1)
            rx2 = _clip_int(bx + bw + pad, 0, W)
            ry2 = _clip_int(by + bh + pad, 0, H)
            if rx2 <= rx1 or ry2 <= ry1:
                rois.append(None)
                roi_rects.append((0, 0, 0, 0))
                continue
            rois.append(img[ry1:ry2, rx1:rx2, :])
            roi_rects.append((rx1, ry1, rx2, ry2))

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
        for roi, (rx1, ry1, rx2, ry2) in zip(rois, roi_rects):
            if roi is None:
                continue
            fr = next(res_iter, None)
            if fr is None:
                continue

            # boxes
            try:
                f_xyxy = (
                    fr.boxes.xyxy.detach().cpu().numpy().astype(np.float32)
                    if getattr(fr, "boxes", None) is not None
                    else np.zeros((0, 4), dtype=np.float32)
                )
            except Exception:
                f_xyxy = np.zeros((0, 4), dtype=np.float32)

            # choose best (largest area) that passes area threshold
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
                if a < thr_area:
                    rejected_bboxes.append(fbbox)
                    continue
                if (best is None) or (a > best[0]):
                    best = (a, fbbox)

            if best is None:
                continue

            _, fbbox = best
            fx1, fy1, fw, fh = fbbox

            # === Маска как заполненный овал по bbox (ВСЕГДА) ===
            full = np.zeros((H, W), dtype=np.uint8)
            cx = int(round(fx1 + fw / 2.0))
            cy = int(round(fy1 + fh / 2.0))
            ax = max(1, int(round(fw / 2.0)))
            ay = max(1, int(round(fh / 2.0)))
            try:
                cv2.ellipse(full, (cx, cy), (ax, ay), 0, 0, 360, 255, -1)
                raw_mask = full
            except Exception:
                # На случай неожиданных ошибок — безопасный прямоугольник
                raw_mask = np.zeros((H, W), dtype=np.uint8)
                cv2.rectangle(
                    raw_mask, (fx1, fy1), (fx1 + fw, fy1 + fh), 255, thickness=-1
                )

            proc_mask = _postprocess_mask(raw_mask, mask_blur_radius)
            accepted_masks.append(proc_mask)
            accepted_bboxes.append(fbbox)

        # удаляем возможные дубли между пересекающимися ROI (глобально)
        accepted_bboxes, accepted_masks = _nms_by_iou_xywh(
            accepted_bboxes, accepted_masks, iou_thr=0.6
        )

        # сортировка слева→направо
        order = sorted(range(len(accepted_bboxes)), key=lambda i: accepted_bboxes[i][0])
        accepted_masks = [accepted_masks[i] for i in order]
        accepted_bboxes = [accepted_bboxes[i] for i in order]
        return accepted_masks, accepted_bboxes, rejected_bboxes

    def execute(
        self,
        image: torch.Tensor,
        BODY_BBOXES,
        device: str = "auto",
        conf: float = 0.25,
        mask_blur_radius: int = 0,
        debug_bbox_thickness: int = 6,
        fp16_infer: bool = False,
        face_model: str = "yolov12s-face.pt",
        face_min_component_percent: float = 1.0,
        face_roi_pad_percent: float = 10.0,
    ):
        # --- вход ---
        try:
            img = _tensor_to_np_image(image)
        except Exception as e:
            raise RuntimeError(f"[ImageFaceDetect] Invalid IMAGE tensor: {e}") from e

        try:
            body_bboxes_xywh = [
                (int(x), int(y), int(w), int(h)) for (x, y, w, h) in BODY_BBOXES
            ]
        except Exception as e:
            raise RuntimeError(f"[ImageFaceDetect] Invalid BODY_BBOXES: {e}") from e

        H, W = img.shape[:2]
        frame_area = float(W * H)
        thr_face_area = max(0.0, float(face_min_component_percent)) / 100.0 * frame_area

        # --- девайс / точность ---
        dev = _resolve_device(device)
        precision = _prefer_precision(fp16_infer, dev)

        # --- модель лиц ---
        model_face = self._load_face_model(face_model, dev, precision)

        try:
            face_masks_u8, face_bboxes_xywh, rejected_face_bboxes = (
                self._detect_faces_for_bodies(
                    img=img,
                    body_bboxes=body_bboxes_xywh,
                    model=model_face,
                    conf=conf,
                    thr_area=thr_face_area,
                    mask_blur_radius=mask_blur_radius,
                    pad_percent=face_roi_pad_percent,
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

        return (
            _np_image_to_tensor(img),  # IMAGE
            face_bboxes_list,  # FACE_BBOXES
            face_masks_batch,  # FACE_MASKS (IMAGE батч)
            bool(found_face),  # FOUND_FACE
            _np_image_to_tensor(debug),  # DEBUG_IMAGE
        )


# Регистрация в ComfyUI
NODE_CLASS_MAPPINGS = {"ImageFaceDetect": ImageFaceDetect}
NODE_DISPLAY_NAME_MAPPINGS = {"ImageFaceDetect": "Image Face Detect"}
