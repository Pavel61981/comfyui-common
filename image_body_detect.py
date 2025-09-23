# filename: image_body_detect.py

import os
from typing import List, Tuple

import cv2
import numpy as np
import torch


def yolo_body_models_dir() -> str:
    """
    Путь для весов тел: <ComfyUI>/models/yolo-body, иначе — локально: ./models/yolo-body
    """
    try:
        from folder_paths import models_dir  # type: ignore

        if isinstance(models_dir, str) and os.path.isdir(models_dir):
            root = os.path.join(models_dir, "yolo-body")
            os.makedirs(root, exist_ok=True)
            return root
    except Exception:
        pass
    here = os.path.dirname(os.path.abspath(__file__))
    root = os.path.join(here, "models", "yolo-body")
    os.makedirs(root, exist_ok=True)
    return root


def _resolve_device(name: str) -> str:
    if name == "auto":
        return "cuda:0" if torch.cuda.is_available() else "cpu"
    return name


def _prefer_precision(fp16_infer: bool, device: str) -> str:
    """
    'fp32' | 'bf16' | 'fp16'
    """
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
    """
    Перевод модели Ultralytics в нужный dtype (тихо игнорирует ошибки).
    """
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
    """
    Небольшая морфология (фикс 3x3) + опциональный GaussianBlur.
    Без кэша ядра.
    """
    if mask_u8 is None or mask_u8.size == 0:
        return mask_u8
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    m = cv2.morphologyEx(mask_u8, cv2.MORPH_CLOSE, kernel)
    m = cv2.morphologyEx(m, cv2.MORPH_DILATE, kernel)
    if blur_r > 0:
        m = cv2.GaussianBlur(m, (blur_r * 2 + 1, blur_r * 2 + 1), 0)
    return m


def _area_polys_fast(polys: List[np.ndarray]) -> float:
    area = 0.0
    for p in polys:
        if p is None or len(p) < 3:
            continue
        x = p[:, 0].astype(np.float64)
        y = p[:, 1].astype(np.float64)
        area += 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))
    return float(area)


def _rasterize_polygons(polys: List[np.ndarray], h: int, w: int) -> np.ndarray:
    mask = np.zeros((h, w), dtype=np.uint8)
    if polys:
        cv2.fillPoly(mask, pts=polys, color=255)
    return mask


def _mask_from_bbox(h: int, w: int, bbox: Tuple[int, int, int, int]) -> np.ndarray:
    x, y, bw, bh = bbox
    m = np.zeros((h, w), dtype=np.uint8)
    if bw > 0 and bh > 0:
        cv2.rectangle(m, (x, y), (x + bw, y + bh), 255, thickness=-1)
    return m


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


class ImageBodyDetect:
    """
    Детекция тел (YOLOv8-seg, класс person).
    Возвращает bbox тел, батч масок (IMAGE) и debug-изображение.
    Без кэширования моделей; освобождение памяти после использования.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {"image": ("IMAGE",)},
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
                "body_model": (
                    ["yolov8n-seg.pt", "yolov8s-seg.pt", "yolov8m-seg.pt"],
                    {"default": "yolov8s-seg.pt"},
                ),
                "body_min_component_percent": (
                    "FLOAT",
                    {"default": 10.0, "min": 0.0, "max": 50.0, "step": 0.1},
                ),
            },
        }

    RETURN_TYPES = (
        "IMAGE",  # IMAGE (исходник)
        "BBOX_LIST",  # BODY_BBOXES
        "IMAGE",  # BODY_MASKS (батч IMAGE: B×H×W×3)
        "BOOLEAN",  # FOUND_BODY
        "IMAGE",  # DEBUG_IMAGE
    )
    RETURN_NAMES = (
        "IMAGE",
        "BODY_BBOXES",
        "BODY_MASKS",
        "FOUND_BODY",
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
                f"[ImageBodyDetect] ultralytics not installed: {e}"
            ) from e
        return YOLO

    def _load_body_model(self, model_name: str, device: str, precision: str):
        YOLO = self._get_ultra()
        candidates = []
        body_dir = yolo_body_models_dir()
        local_path = os.path.join(body_dir, model_name)
        if os.path.isfile(local_path):
            candidates.append(local_path)
        here = os.path.dirname(os.path.abspath(__file__))
        near_path = os.path.join(here, model_name)
        if os.path.isfile(near_path):
            candidates.append(near_path)

        last_err = None
        for cand in candidates + [model_name]:
            try:
                model = YOLO(cand)
                model.to(device)
                _set_model_precision(model, precision)
                return model
            except Exception as e:
                last_err = e
                continue

        raise RuntimeError(
            "[ImageBodyDetect] Failed to load body model "
            f"'{model_name}'. Ensure Ultralytics can auto-download it, or place the file here:\n"
            f" - {os.path.join(yolo_body_models_dir(), model_name)}\n"
            f" - {os.path.join(os.path.dirname(os.path.abspath(__file__)), model_name)}\n"
            f"Last error: {last_err}"
        )

    def _detect_bodies(
        self,
        img: np.ndarray,
        model,
        conf: float,
        thr_area: float,
        mask_blur_radius: int,
    ) -> Tuple[
        List[np.ndarray],
        List[Tuple[int, int, int, int]],
        List[Tuple[int, int, int, int]],
    ]:
        """
        Возвращает (accepted_masks_u8, accepted_bboxes_xywh, rejected_bboxes_xywh)
        """
        H, W = img.shape[:2]
        try:
            results = model.predict(
                source=img,
                conf=conf,
                iou=0.45,
                max_det=20,
                classes=[0],  # person
                verbose=False,
                device=None,  # модель уже на девайсе
            )
        except Exception as e:
            raise RuntimeError(
                f"[ImageBodyDetect] YOLO body predict failed: {e}"
            ) from e

        r = results[0] if results else None
        accepted_masks: List[np.ndarray] = []
        accepted_bboxes: List[Tuple[int, int, int, int]] = []
        rejected_bboxes: List[Tuple[int, int, int, int]] = []

        if r is None:
            return accepted_masks, accepted_bboxes, rejected_bboxes

        # Boxes
        try:
            xyxy = (
                r.boxes.xyxy.detach().cpu().numpy().astype(np.float32)
                if getattr(r, "boxes", None) is not None
                else np.zeros((0, 4), dtype=np.float32)
            )
        except Exception:
            xyxy = np.zeros((0, 4), dtype=np.float32)

        # Masks: prefer tensor data (faster), else polygons
        masks_data = None
        if getattr(r, "masks", None) is not None:
            try:
                masks_data = getattr(r.masks, "data", None)  # torch.Tensor [N,h,w]
            except Exception:
                masks_data = None

        polys_src = []
        if masks_data is None:
            try:
                polys_src = (
                    r.masks.xy
                    if (
                        getattr(r, "masks", None) is not None
                        and getattr(r.masks, "xy", None) is not None
                    )
                    else []
                )
            except Exception:
                polys_src = []

        for i in range(xyxy.shape[0]):
            x1f, y1f, x2f, y2f = xyxy[i]
            x1 = _clip_int(int(round(x1f)), 0, W - 1)
            y1 = _clip_int(int(round(y1f)), 0, H - 1)
            x2 = _clip_int(int(round(x2f)), 0, W)
            y2 = _clip_int(int(round(y2f)), 0, H)
            bw, bh = max(0, x2 - x1), max(0, y2 - y1)
            if bw <= 0 or bh <= 0:
                continue

            bbox = (x1, y1, bw, bh)
            bbox_area = float(bw * bh)
            if bbox_area < thr_area:
                rejected_bboxes.append(bbox)
                continue

            raw_mask = None
            if masks_data is not None:
                try:
                    md = masks_data[i]
                    mu8 = (md.detach().float().cpu().numpy() >= 0.5).astype(
                        np.uint8
                    ) * 255
                    mu8 = cv2.resize(mu8, (W, H), interpolation=cv2.INTER_NEAREST)
                    if float((mu8 > 0).sum()) < thr_area:
                        rejected_bboxes.append(bbox)
                        continue
                    raw_mask = mu8
                except Exception:
                    raw_mask = None

            if raw_mask is None and polys_src:
                polys_int: List[np.ndarray] = []
                if i < len(polys_src) and polys_src[i] is not None:
                    p = polys_src[i]
                    if isinstance(p, list):
                        for pp in p:
                            if pp is None or len(pp) < 3:
                                continue
                            arr = np.asarray(pp, dtype=np.float32)
                            arr[:, 0] = np.clip(arr[:, 0], 0, W - 1)
                            arr[:, 1] = np.clip(arr[:, 1], 0, H - 1)
                            polys_int.append(arr.astype(np.int32))
                    else:
                        arr = np.asarray(p, dtype=np.float32)
                        if arr.ndim == 2 and arr.shape[0] >= 3:
                            arr[:, 0] = np.clip(arr[:, 0], 0, W - 1)
                            arr[:, 1] = np.clip(arr[:, 1], 0, H - 1)
                            polys_int.append(arr.astype(np.int32))
                if polys_int:
                    if _area_polys_fast(polys_int) < thr_area:
                        rejected_bboxes.append(bbox)
                        continue
                    raw_mask = _rasterize_polygons(polys_int, H, W)

            if raw_mask is None:
                raw_mask = _mask_from_bbox(H, W, bbox)

            proc_mask = _postprocess_mask(raw_mask, mask_blur_radius)
            accepted_masks.append(proc_mask)
            accepted_bboxes.append(bbox)

        order = sorted(range(len(accepted_bboxes)), key=lambda i: accepted_bboxes[i][0])
        accepted_masks = [accepted_masks[i] for i in order]
        accepted_bboxes = [accepted_bboxes[i] for i in order]
        return accepted_masks, accepted_bboxes, rejected_bboxes

    def execute(
        self,
        image: torch.Tensor,
        device: str = "auto",
        conf: float = 0.25,
        mask_blur_radius: int = 0,
        debug_bbox_thickness: int = 6,
        fp16_infer: bool = False,
        body_model: str = "yolov8s-seg.pt",
        body_min_component_percent: float = 10.0,
    ):
        # --- вход ---
        try:
            img = _tensor_to_np_image(image)
        except Exception as e:
            raise RuntimeError(f"[ImageBodyDetect] Invalid IMAGE tensor: {e}") from e

        H, W = img.shape[:2]
        frame_area = float(W * H)
        thr_body_area = max(0.0, float(body_min_component_percent)) / 100.0 * frame_area

        # --- девайс / точность ---
        dev = _resolve_device(device)
        precision = _prefer_precision(fp16_infer, dev)

        # --- модели ---
        model_body = self._load_body_model(body_model, dev, precision)

        try:
            # --- тела ---
            body_masks_u8, body_bboxes_xywh, rejected_body_bboxes = self._detect_bodies(
                img=img,
                model=model_body,
                conf=conf,
                thr_area=thr_body_area,
                mask_blur_radius=mask_blur_radius,
            )
        finally:
            _cleanup_model(model_body, dev)

        found_body = len(body_bboxes_xywh) > 0

        # --- конвертация масок в батч IMAGE ---
        body_masks_batch = _masks_to_image_batch(body_masks_u8, H, W)
        body_bboxes_list = [
            [int(x), int(y), int(w), int(h)] for (x, y, w, h) in body_bboxes_xywh
        ]

        # --- DEBUG ---
        debug = img.copy()
        overlay = np.zeros_like(debug)

        if body_masks_u8:
            mb = np.any(np.stack([m > 0 for m in body_masks_u8], axis=0), axis=0)
            overlay[mb] = [0, 0, 255]  # синий (BGR)

        debug = cv2.addWeighted(overlay, 0.35, debug, 0.65, 0)

        t = int(max(1, min(10, debug_bbox_thickness)))
        # тела — жёлтые
        for i, (x, y, w_, h_) in enumerate(body_bboxes_xywh):
            x2, y2 = x + w_, y + h_
            cv2.rectangle(debug, (x, y), (x2, y2), (0, 255, 255), thickness=t)
            cv2.putText(
                debug,
                f"Person #{i}",
                (x + 5, max(15, y + 15)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 255),
                1,
                cv2.LINE_AA,
            )

        # отклонённые тела — красные перечёркнутые
        for x, y, w_, h_ in rejected_body_bboxes:
            x2, y2 = x + w_, y + h_
            cv2.rectangle(debug, (x, y), (x2, y2), (0, 0, 255), thickness=t)
            cv2.line(debug, (x, y), (x2, y2), (0, 0, 255), thickness=t)
            cv2.line(debug, (x2, y), (x, y2), (0, 0, 255), thickness=t)

        return (
            _np_image_to_tensor(img),  # IMAGE
            body_bboxes_list,  # BODY_BBOXES
            body_masks_batch,  # BODY_MASKS (IMAGE батч)
            bool(found_body),  # FOUND_BODY
            _np_image_to_tensor(debug),  # DEBUG_IMAGE
        )


# Регистрация в ComfyUI
NODE_CLASS_MAPPINGS = {"ImageBodyDetect": ImageBodyDetect}
NODE_DISPLAY_NAME_MAPPINGS = {"ImageBodyDetect": "Image Body Detect"}
