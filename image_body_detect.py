# filename: image_body_detect.py

import os
from typing import List, Tuple

import cv2
import numpy as np
import torch
import logging

# ------------------------------------------------------------
# ОТКЛЮЧЕНИЕ СЕТЕВЫХ ПРОВЕРОК ULTRALYTICS
# Это предотвращает долгие задержки при инициализации модели
# ------------------------------------------------------------
try:
    from ultralytics.utils import SETTINGS

    SETTINGS.update({"HUB": False, "VERBOSE": False})
    # Это сообщение будет видно в консоли при запуске ComfyUI
    print(
        "[ImageBodyDetect] Ultralytics HUB/update checks disabled for faster startup."
    )
except Exception as e:
    print(f"[ImageBodyDetect] Warning: Failed to disable Ultralytics HUB checks: {e}")


# ------------------------------------------------------------
# ЛОГИРОВАНИЕ
# ------------------------------------------------------------
LOGGER_NAME = "ImageBodyDetect"
logger = logging.getLogger(LOGGER_NAME)
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter("[%(name)s] %(levelname)s: %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
logger.setLevel(logging.INFO)


# ------------------------------------------------------------
# МОДЕЛИ
# ------------------------------------------------------------
ALLOWED_BODY_MODELS = {
    "yolov8n-seg.pt",
    "yolov8s-seg.pt",
    "yolov8m-seg.pt",
    "yolov8l-seg.pt",
    "yolov8x-seg.pt",
}


# ------------------------------------------------------------
# ХЕЛПЕРЫ
# ------------------------------------------------------------
def _resolve_device(name: str) -> str:
    if name == "auto":
        dev = "cuda:0" if torch.cuda.is_available() else "cpu"
        logger.info(f"Выбран device=auto → {dev}")
        return dev
    if name.startswith("cuda") and not torch.cuda.is_available():
        logger.warning(f"Запрошен {name}, но CUDA недоступна. Переключаемся на cpu.")
        return "cpu"
    return name


def _set_model_precision(yolo_model) -> None:
    try:
        m = getattr(yolo_model, "model", None)
        if m is not None:
            m.to(dtype=torch.float32)
            logger.info("Модель приведена к dtype=float32")
    except Exception as e:
        logger.warning(f"Не удалось установить dtype=float32: {e}")


def _cleanup_model(model, device: str) -> None:
    """Аккуратно освобождает память модели (CPU/GPU)."""
    try:
        m = getattr(model, "model", None)
        if m is not None:
            m.to("cpu")
    except Exception as e:
        logger.debug(f"Не удалось перенести модель на CPU при очистке: {e}")
    finally:
        del model
        if device.startswith("cuda") and torch.cuda.is_available():
            try:
                torch.cuda.empty_cache()
                logger.info("CUDA cache очищен")
            except Exception as e:
                logger.debug(f"torch.cuda.empty_cache() ошибка: {e}")


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


def _postprocess_mask(mask_u8: np.ndarray, pad_px: int, blur_r: int) -> np.ndarray:
    if mask_u8 is None or mask_u8.size == 0:
        return mask_u8
    try:
        m = mask_u8
        if m.dtype != np.uint8:
            m = m.astype(np.uint8, copy=False)
        kern_base = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, kern_base)
        m = cv2.morphologyEx(m, cv2.MORPH_DILATE, kern_base)
        if isinstance(pad_px, int) and pad_px != 0:
            k = int(abs(pad_px))
            ksize = max(1, 2 * k + 1)
            kern_pad = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
            if pad_px > 0:
                m = cv2.dilate(m, kern_pad, iterations=1)
            else:
                m = cv2.erode(m, kern_pad, iterations=1)
        if isinstance(blur_r, int) and blur_r > 0:
            ksize = 2 * int(blur_r) + 1
            m = cv2.GaussianBlur(m, (ksize, ksize), 0)
        return m
    except Exception:
        return mask_u8


def _mask_from_bbox(h: int, w: int, bbox: Tuple[int, int, int, int]) -> np.ndarray:
    x, y, bw, bh = bbox
    m = np.zeros((h, w), dtype=np.uint8)
    if bw > 0 and bh > 0:
        cv2.rectangle(m, (x, y), (x + bw, y + bh), 255, thickness=-1)
    return m


def _masks_to_image_batch(masks_u8: List[np.ndarray], h: int, w: int) -> torch.Tensor:
    if not masks_u8:
        return torch.from_numpy(np.zeros((1, h, w, 3), dtype=np.float32))
    stack = np.stack([(m.astype(np.float32) / 255.0) for m in masks_u8], axis=0)
    stack = stack[..., np.newaxis]
    stack = np.repeat(stack, 3, axis=3)
    return torch.from_numpy(stack)


class ImageBodyDetect:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {"image": ("IMAGE",)},
            "optional": {
                "device": (["auto", "cpu", "cuda:0"], {"default": "auto"}),
                "body_model": (
                    list(sorted(ALLOWED_BODY_MODELS)),
                    {"default": "yolov8m-seg.pt"},
                ),
                "conf": (
                    "FLOAT",
                    {"default": 0.25, "min": 0.0, "max": 1.0, "step": 0.01},
                ),
                "body_min_component_percent": (
                    "FLOAT",
                    {"default": 10.0, "min": 0.0, "max": 50.0, "step": 0.1},
                ),
                "mask_padding_px": (
                    "INT",
                    {"default": 0, "min": -256, "max": 256, "step": 1},
                ),
                "mask_blur_radius": (
                    "INT",
                    {"default": 0, "min": 0, "max": 50, "step": 1},
                ),
                "debug_bbox_thickness": (
                    "INT",
                    {"default": 5, "min": 1, "max": 10, "step": 1},
                ),
            },
        }

    RETURN_TYPES = ("IMAGE", "BBOX_LIST", "IMAGE", "BOOLEAN", "IMAGE")
    RETURN_NAMES = ("IMAGE", "BODY_BBOXES", "BODY_MASKS", "FOUND_BODY", "DEBUG_IMAGE")
    FUNCTION = "execute"
    CATEGORY = "Masquerade/Detect"
    OUTPUT_NODE = False

    def _get_ultra(self):
        try:
            from ultralytics import YOLO

            return YOLO
        except Exception as e:
            raise RuntimeError(
                f"[ImageBodyDetect] Ultralytics не установлен: {e}"
            ) from e

    def _load_body_model(self, model_name: str, device: str):
        if model_name not in ALLOWED_BODY_MODELS:
            raise RuntimeError(f"Некорректное имя модели: {model_name}")

        YOLO = self._get_ultra()
        logger.info(f"Загрузка модели '{model_name}'...")
        try:
            model = YOLO(model_name)
            model.to(device)
            # _set_model_precision(model)
            logger.info(
                f"Модель '{model_name}' успешно загружена на устройство '{device}'."
            )
            return model
        except Exception as e:
            raise RuntimeError(
                f"Не удалось загрузить/скачать модель '{model_name}'. Ошибка: {e}"
            ) from e

    def _detect_bodies(
        self,
        img: np.ndarray,
        model,
        conf: float,
        thr_area: float,
        mask_padding_px: int,
        mask_blur_radius: int,
    ):
        H, W = img.shape[:2]
        try:
            results = model.predict(
                source=img, conf=conf, iou=0.45, max_det=20, classes=[0], verbose=False
            )
        except Exception as e:
            raise RuntimeError(f"YOLO predict failed: {e}") from e

        r = results[0] if results else None
        accepted_masks, accepted_bboxes, rejected_bboxes = [], [], []
        if r is None:
            return accepted_masks, accepted_bboxes, rejected_bboxes

        xyxy = (
            r.boxes.xyxy.detach().cpu().numpy()
            if getattr(r, "boxes", None) is not None
            else np.zeros((0, 4))
        )
        masks_data = (
            getattr(r.masks, "data", None)
            if getattr(r, "masks", None) is not None
            else None
        )

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
            if float(bw * bh) < thr_area:
                rejected_bboxes.append(bbox)
                continue

            raw_mask = None
            if masks_data is not None and i < len(masks_data):
                md = masks_data[i]
                mu8 = (md.detach().float().cpu().numpy() >= 0.5).astype(np.uint8) * 255
                raw_mask = cv2.resize(mu8, (W, H), interpolation=cv2.INTER_NEAREST)

            if raw_mask is None:
                raw_mask = _mask_from_bbox(H, W, bbox)

            proc_mask = _postprocess_mask(
                raw_mask, pad_px=mask_padding_px, blur_r=mask_blur_radius
            )
            accepted_masks.append(proc_mask)
            accepted_bboxes.append(bbox)

        order = sorted(range(len(accepted_bboxes)), key=lambda j: accepted_bboxes[j][0])
        accepted_masks = [accepted_masks[j] for j in order]
        accepted_bboxes = [accepted_bboxes[j] for j in order]
        logger.info(
            f"Детекций принято: {len(accepted_bboxes)}, отклонено: {len(rejected_bboxes)}"
        )
        return accepted_masks, accepted_bboxes, rejected_bboxes

    def execute(
        self,
        image: torch.Tensor,
        device: str,
        conf: float,
        mask_blur_radius: int,
        debug_bbox_thickness: int,
        body_model: str,
        body_min_component_percent: float,
        mask_padding_px: int,
    ):

        img = _tensor_to_np_image(image)
        H, W = img.shape[:2]
        thr_body_area = (float(body_min_component_percent) / 100.0) * (W * H)
        logger.info(
            f"Порог площади тела: {body_min_component_percent:.2f}% → {thr_body_area:.1f} пикс."
        )

        dev = _resolve_device(device)
        model_body = self._load_body_model(body_model, dev)

        try:
            body_masks_u8, body_bboxes_xywh, rejected_body_bboxes = self._detect_bodies(
                img=img,
                model=model_body,
                conf=conf,
                thr_area=thr_body_area,
                mask_padding_px=mask_padding_px,
                mask_blur_radius=mask_blur_radius,
            )
        finally:
            # Освобождаем VRAM после каждого запуска
            _cleanup_model(model_body, dev)

        found_body = len(body_bboxes_xywh) > 0
        body_masks_batch = _masks_to_image_batch(body_masks_u8, H, W)
        body_bboxes_list = [[int(v) for v in box] for box in body_bboxes_xywh]

        # DEBUG-изображение
        debug = img.copy()
        if body_masks_u8:
            combined_mask = np.any(
                np.stack([m > 0 for m in body_masks_u8], axis=0), axis=0
            )
            overlay = np.zeros_like(debug)
            overlay[combined_mask] = [0, 0, 255]  # Синий
            debug = cv2.addWeighted(overlay, 0.35, debug, 0.65, 0)

        t = int(max(1, debug_bbox_thickness))
        # Принятые тела - желтые
        for i, (x, y, w, h) in enumerate(body_bboxes_xywh):
            cv2.rectangle(debug, (x, y), (x + w, y + h), (0, 255, 255), t)
            cv2.putText(
                debug,
                f"Person #{i}",
                (x + 5, y + 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 255),
                2,
                cv2.LINE_AA,
            )
        # Отклоненные тела - красные, перечеркнутые
        for x, y, w, h in rejected_body_bboxes:
            cv2.rectangle(debug, (x, y), (x + w, y + h), (0, 0, 255), t)
            cv2.line(debug, (x, y), (x + w, y + h), (0, 0, 255), t)
            cv2.line(debug, (x + w, y), (x, y + h), (0, 0, 255), t)

        return (
            _np_image_to_tensor(img),
            body_bboxes_list,
            body_masks_batch,
            bool(found_body),
            _np_image_to_tensor(debug),
        )


# Регистрация в ComfyUI
NODE_CLASS_MAPPINGS = {"ImageBodyDetect": ImageBodyDetect}
NODE_DISPLAY_NAME_MAPPINGS = {"ImageBodyDetect": "Image Body Detect"}
