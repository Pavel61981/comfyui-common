# filename: PeopleSegGetMasks.py
"""
ComfyUI custom nodes: PeopleSegModelsSetup, PeopleSegGetMasks
Версия: rotated-head + ellipse, без blur/offset, с параметрами чувствительности и поворотами.

Изменения:
- Голова НЕ вырезается из тела (независимые маски).
- Ротационно-устойчивый поиск лица: повороты 0/±45/±90 (по пресету).
- Эллипс головы, ориентированный по углу лица (из landmarks либо из угла поворота).
- Порог чувствительности детекторов в UI.
- Параметры: rot_search, rot_search_preset, head_geom (ellipse|rect).
"""

from __future__ import annotations
import os
import gc
import math
import time
import urllib.request
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

# ComfyUI paths helper
try:
    import folder_paths
    COMFY_MODELS_DIR = folder_paths.get_full_path("models")
except Exception:
    COMFY_MODELS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "models"))

PEOPLE_SEG_DIR = os.path.join(COMFY_MODELS_DIR, "people_seg")
os.makedirs(PEOPLE_SEG_DIR, exist_ok=True)

# ---------------------------
# Download helpers
# ---------------------------

def _download_with_retry(url: str, dst_path: str, min_size: int = 600_000, retries: int = 1, timeout: int = 30) -> None:
    last_err = None
    for attempt in range(retries + 1):
        try:
            tmp = dst_path + ".part"
            if os.path.exists(tmp):
                os.remove(tmp)
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=timeout) as r, open(tmp, "wb") as f:
                f.write(r.read())
            if os.path.getsize(tmp) < min_size:
                raise RuntimeError(f"Downloaded file too small: {tmp}")
            with open(tmp, "rb") as f:
                head = f.read(1024).lower()
                if b"<html" in head or b"<!doctype" in head:
                    raise RuntimeError("Downloaded HTML instead of binary file")
            os.replace(tmp, dst_path)
            return
        except Exception as e:
            last_err = e
            time.sleep(1.0)
    raise RuntimeError(f"Failed to download {url} -> {dst_path}: {last_err}")

def _ensure_weights() -> Dict[str, Optional[str]]:
    yolo_s = os.path.join(PEOPLE_SEG_DIR, "yolov8s-seg.pt")
    yunet = os.path.join(PEOPLE_SEG_DIR, "face_yunet.onnx")

    YOLO_S_URL = "https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8s-seg.pt"
    YUNET_URLS = [
        "https://raw.githubusercontent.com/opencv/opencv_zoo/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx",
        "https://raw.githubusercontent.com/opencv/opencv_zoo/master/models/face_detection_yunet/face_detection_yunet_2023mar.onnx",
        "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx",
        "https://github.com/opencv/opencv_zoo/raw/master/models/face_detection_yunet/face_detection_yunet_2023mar.onnx",
    ]

    if not os.path.exists(yolo_s):
        print("[PeopleSeg] Downloading YOLOv8s-seg weights...")
        _download_with_retry(YOLO_S_URL, yolo_s, min_size=5_000_000, retries=1)

    if not os.path.exists(yunet):
        print("[PeopleSeg] Downloading YuNet ONNX (optional)...")
        ok = False
        for u in YUNET_URLS:
            try:
                _download_with_retry(u, yunet, min_size=600_000, retries=0)
                ok = True
                break
            except Exception as e:
                print(f"[PeopleSeg] YuNet mirror failed: {u} -> {e}")
        if not ok:
            print("[PeopleSeg] YuNet weights not available; will use fallback head logic.")

    return {"yolo_s": yolo_s, "yunet": yunet if os.path.exists(yunet) else None}

# ---------------------------
# Tensor/image helpers
# ---------------------------

def _to_numpy_image_uint8(img_t: torch.Tensor) -> np.ndarray:
    if img_t.dim() != 4 or img_t.shape[0] != 1 or img_t.shape[-1] != 3:
        raise RuntimeError(f"[PeopleSeg] Expected IMAGE shape (1,H,W,3), got {tuple(img_t.shape)}")
    img = img_t[0].detach().cpu().numpy()
    img = np.clip(img * 255.0, 0, 255).astype(np.uint8)
    return img[:, :, ::-1].copy()  # RGB->BGR

def _to_bchw(mask: torch.Tensor) -> Tuple[torch.Tensor, str]:
    if mask.dim() == 2:  # (H,W)
        return mask.unsqueeze(0).unsqueeze(0).float(), "2d"
    if mask.dim() == 3:  # (N,H,W)
        return mask.unsqueeze(1).float(), "3d"
    raise RuntimeError(f"[PeopleSeg] Unsupported mask dim {mask.dim()}.")

def _from_bchw(bchw: torch.Tensor, kind: str) -> torch.Tensor:
    if kind == "2d":
        return bchw[0, 0]
    if kind == "3d":
        return bchw[:, 0]
    raise RuntimeError(f"[PeopleSeg] Unknown shape kind '{kind}'.")

def _dilate(mask_bin: torch.Tensor, k: int) -> torch.Tensor:
    if k <= 0:
        return mask_bin
    m, kind = _to_bchw(mask_bin)
    for _ in range(k):
        m = F.max_pool2d(m, kernel_size=3, stride=1, padding=1)
    return _from_bchw(m, kind)

def _erode(mask_bin: torch.Tensor, k: int) -> torch.Tensor:
    if k <= 0:
        return mask_bin
    inv = 1.0 - mask_bin
    inv = _dilate(inv, k)
    return 1.0 - inv

def _morph_close(mask_bin: torch.Tensor, k: int = 1) -> torch.Tensor:
    if k <= 0:
        return mask_bin
    return _erode(_dilate(mask_bin, k), k)

def _morph_open(mask_bin: torch.Tensor, k: int = 1) -> torch.Tensor:
    if k <= 0:
        return mask_bin
    return _dilate(_erode(mask_bin, k), k)

def _resize_mask(mask: torch.Tensor, H: int, W: int) -> torch.Tensor:
    m = F.interpolate(mask[None, None, ...], size=(H, W), mode="bilinear", align_corners=False)
    return m[0, 0].clamp(0.0, 1.0)

def _bbox_xyxy_to_xywh(xyxy: np.ndarray, H: int, W: int) -> List[int]:
    x1, y1, x2, y2 = xyxy
    x1 = int(max(0, min(x1, W - 1)))
    y1 = int(max(0, min(y1, H - 1)))
    x2 = int(max(0, min(x2, W - 1)))
    y2 = int(max(0, min(y2, H - 1)))
    x = max(0, min(x1, x2))
    y = max(0, min(y1, y2))
    w = max(0, abs(x2 - x1))
    h = max(0, abs(y2 - y1))
    return [x, y, w, h]

def _crop_with_pad(img_bgr: np.ndarray, bbox_xywh: List[int], pad_ratio: float = 0.05) -> Tuple[np.ndarray, Tuple[int,int]]:
    H, W = img_bgr.shape[:2]
    x, y, w, h = bbox_xywh
    pad_x = int(round(w * pad_ratio))
    pad_y = int(round(h * pad_ratio))
    x0 = max(0, x - pad_x)
    y0 = max(0, y - pad_y)
    x1 = min(W, x + w + pad_x)
    y1 = min(H, y + h + pad_y)
    return img_bgr[y0:y1, x0:x1].copy(), (x0, y0)

def _rect_mask_from_xywh(H: int, W: int, xywh: List[int], device: torch.device) -> torch.Tensor:
    x, y, w, h = xywh
    m = torch.zeros((H, W), dtype=torch.float32, device=device)
    x2 = min(W, x + w)
    y2 = min(H, y + h)
    if w > 0 and h > 0:
        m[y:y2, x:x2] = 1.0
    return m

def _intersect(mask_a: torch.Tensor, mask_b: torch.Tensor) -> torch.Tensor:
    return (mask_a * mask_b).clamp(0.0, 1.0)

def _resolve_overlaps_by_centroids(bodies_bin: torch.Tensor, bboxes_xywh: List[List[int]]) -> torch.Tensor:
    N, H, W = bodies_bin.shape
    if N <= 1:
        return bodies_bin
    device = bodies_bin.device
    yy, xx = torch.meshgrid(torch.arange(H, device=device), torch.arange(W, device=device), indexing="ij")
    centers = []
    for (x, y, w, h) in bboxes_xywh:
        centers.append((x + w / 2.0, y + h / 2.0))
    centers = torch.tensor(centers, dtype=torch.float32, device=device)  # (N,2)

    dx = xx[None, :, :] - centers[:, 0][:, None, None]
    dy = yy[None, :, :] - centers[:, 1][:, None, None]
    dist2 = dx * dx + dy * dy  # (N,H,W)

    union = (bodies_bin.sum(dim=0) > 0.5)
    nearest = torch.argmin(dist2, dim=0)  # (H,W)

    exclusive = torch.zeros_like(bodies_bin)
    for i in range(N):
        assigned = (nearest == i) & union
        exclusive[i] = (assigned.float() * bodies_bin[i]).clamp(0.0, 1.0)
    return exclusive

# -------- rotation helpers (pure torch) --------

def _rotate_rgb_u8_same_size(rgb: np.ndarray, angle_deg: float) -> np.ndarray:
    """Rotate RGB uint8 image by angle_deg around center, keep same size, zero padding."""
    if abs(angle_deg) < 1e-3:
        return rgb.copy()
    H, W = rgb.shape[:2]
    img = torch.from_numpy(rgb.astype(np.float32) / 255.0).permute(2, 0, 1)[None]  # (1,3,H,W)
    theta = torch.tensor([
        [math.cos(math.radians(angle_deg)), -math.sin(math.radians(angle_deg)), 0.0],
        [math.sin(math.radians(angle_deg)),  math.cos(math.radians(angle_deg)), 0.0]
    ], dtype=torch.float32)[None]  # (1,2,3)
    grid = F.affine_grid(theta, size=img.size(), align_corners=False)
    rot = F.grid_sample(img, grid, mode="bilinear", padding_mode="zeros", align_corners=False)
    out = (rot[0].permute(1, 2, 0).clamp(0, 1).numpy() * 255.0).astype(np.uint8)
    return out

def _rect_corners(x: int, y: int, w: int, h: int) -> np.ndarray:
    return np.array([[x, y],
                     [x + w, y],
                     [x + w, y + h],
                     [x, y + h]], dtype=np.float32)

def _rotate_points(points: np.ndarray, center: Tuple[float, float], angle_deg: float) -> np.ndarray:
    """Rotate points around center by angle (deg, CCW)."""
    ang = math.radians(angle_deg)
    R = np.array([[math.cos(ang), -math.sin(ang)],
                  [math.sin(ang),  math.cos(ang)]], dtype=np.float32)
    pts = points - np.asarray(center, dtype=np.float32)
    return (pts @ R.T) + np.asarray(center, dtype=np.float32)

def _map_rect_from_rot_to_src(xywh_rot: List[int], rot_angle_deg: float, w: int, h: int) -> List[int]:
    """Axis-aligned rect in rotated image -> axis-aligned rect in original (same size) via inverse rotation."""
    if abs(rot_angle_deg) < 1e-3:
        return [int(xywh_rot[0]), int(xywh_rot[1]), int(xywh_rot[2]), int(xywh_rot[3])]
    cx, cy = w / 2.0, h / 2.0
    corners = _rect_corners(*xywh_rot)
    # map by inverse rotation (-angle)
    mapped = _rotate_points(corners, (cx, cy), -rot_angle_deg)
    x1y1 = mapped.min(axis=0)
    x2y2 = mapped.max(axis=0)
    x1, y1 = np.clip(x1y1, [0, 0], [w - 1, h - 1])
    x2, y2 = np.clip(x2y2, [0, 0], [w - 1, h - 1])
    x = int(max(0, min(x1, x2)))
    y = int(max(0, min(y1, y2)))
    W = int(max(0, x2 - x1))
    H = int(max(0, y2 - y1))
    return [x, y, W, H]

# -------- ellipse rasterizer (no OpenCV) --------

def _ellipse_mask(H: int, W: int, cx: float, cy: float, ax: float, by: float, angle_deg: float, device: torch.device) -> torch.Tensor:
    """
    Filled rotated ellipse mask (H,W) float {0,1}.
    (x',y') = rotation of (x-cx, y-cy) by -angle; inside if (x'/ax)^2 + (y'/by)^2 <= 1.
    """
    yy, xx = torch.meshgrid(torch.arange(H, device=device), torch.arange(W, device=device), indexing="ij")
    xx = xx.float() - float(cx)
    yy = yy.float() - float(cy)
    ang = math.radians(angle_deg)
    cos, sin = math.cos(ang), math.sin(ang)
    # rotate by -angle to align ellipse with axes
    xprime =  xx * cos + yy * sin
    yprime = -xx * sin + yy * cos
    ax = max(1e-6, float(ax))
    by = max(1e-6, float(by))
    v = (xprime / ax) ** 2 + (yprime / by) ** 2
    return (v <= 1.0).float()

# ---------------------------
# Face detectors
# ---------------------------

class _FaceDetectorMediapipe:
    """
    detect(rgb_u8) -> list of dicts:
      {"xywh":[x,y,w,h], "score":float, "angle":float or None}
    angle вычисляется по линии глаз, если ключевые точки доступны.
    """
    def __init__(self):
        import mediapipe as mp
        self.mp = mp
        self.det = mp.solutions.face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5)

    def detect(self, rgb_u8: np.ndarray) -> List[Dict[str, Any]]:
        H, W = rgb_u8.shape[:2]
        res = self.det.process(rgb_u8)
        out: List[Dict[str, Any]] = []
        if res.detections:
            for d in res.detections:
                bbox = d.location_data.relative_bounding_box
                x = max(0, int(bbox.xmin * W))
                y = max(0, int(bbox.ymin * H))
                w = max(0, int(bbox.width * W))
                h = max(0, int(bbox.height * H))
                score = float(d.score[0]) if d.score else 0.0
                ang = None
                # попробуем вычислить угол по keypoints (left/right eye)
                try:
                    rkp = d.location_data.relative_keypoints
                    if rkp and len(rkp) >= 2:
                        lx = float(rkp[0].x) * W
                        ly = float(rkp[0].y) * H
                        rx = float(rkp[1].x) * W
                        ry = float(rkp[1].y) * H
                        ang = math.degrees(math.atan2(ry - ly, rx - lx))  # угол линии глаз
                except Exception:
                    pass
                if w > 0 and h > 0:
                    out.append({"xywh": [x, y, w, h], "score": score, "angle": ang})
        return out

class _FaceDetectorYuNetOpenCV:
    """
    YuNet через OpenCV FaceDetectorYN.
    detect(rgb_u8) -> list of dicts: {"xywh":[x,y,w,h], "score":float, "angle":float}
    angle по линии глаз из landmark'ов.
    """
    def __init__(self, onnx_path: str):
        import cv2
        self.cv2 = cv2
        if not hasattr(cv2, "FaceDetectorYN_create"):
            raise RuntimeError("OpenCV FaceDetectorYN not available (need opencv-contrib-python>=4.6).")
        if not onnx_path or not os.path.exists(onnx_path):
            raise RuntimeError(f"YuNet ONNX not found: {onnx_path}")
        self.det = cv2.FaceDetectorYN_create(onnx_path, "", (320, 320))

    def detect(self, rgb_u8: np.ndarray) -> List[Dict[str, Any]]:
        H, W = rgb_u8.shape[:2]
        bgr = rgb_u8[:, :, ::-1].copy()
        self.det.setInputSize((W, H))
        faces, _ = self.det.detect(bgr)
        out: List[Dict[str, Any]] = []
        if faces is not None and len(faces) > 0:
            for row in faces:
                # row: x,y,w,h, 10 landmarks (x1,y1,...,x5,y5), score
                x, y, w, h = [int(max(0, v)) for v in row[:4]]
                score = float(row[-1])
                # глаза: landmarks 0: (x1,y1), 1:(x2,y2) обычно л/п глаз
                try:
                    lx, ly = float(row[4]), float(row[5])
                    rx, ry = float(row[6]), float(row[7])
                    ang = math.degrees(math.atan2(ry - ly, rx - lx))
                except Exception:
                    ang = None
                if w > 0 and h > 0:
                    out.append({"xywh": [x, y, w, h], "score": score, "angle": ang})
        return out

# ---------------------------
# Node 1: Models Setup
# ---------------------------

class PeopleSegModelsSetup:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {}}

    RETURN_TYPES = ("PEOPLE_MODELS",)
    RETURN_NAMES = ("PEOPLE_MODELS",)
    FUNCTION = "execute"
    CATEGORY = "segmentation/people"
    OUTPUT_NODE = False

    def execute(self):
        paths = _ensure_weights()

        try:
            from ultralytics import YOLO
        except Exception as e:
            raise RuntimeError(
                "[PeopleSeg] Требуется пакет 'ultralytics' (pip install ultralytics). "
                f"Импорт не удался: {e}"
            ) from e

        try:
            seg_model = YOLO(paths["yolo_s"])
            seg_model.model.eval()
            seg_model.to("cpu")
        except Exception as e:
            raise RuntimeError(f"[PeopleSeg] Не удалось загрузить YOLOv8-seg: {e}") from e

        face_primary = None
        mp_version = None
        try:
            import mediapipe as mp  # noqa: F401
            face_primary = _FaceDetectorMediapipe()
            mp_version = getattr(mp, "__version__", "unknown")
            print("[PeopleSeg] MediaPipe: OK")
        except Exception as e:
            print(f"[PeopleSeg] MediaPipe недоступен: {e}")

        face_fallback = None
        cv_version = None
        try:
            if paths.get("yunet"):
                import cv2  # noqa: F401
                face_fallback = _FaceDetectorYuNetOpenCV(paths["yunet"])
                cv_version = getattr(cv2, "__version__", "unknown")
                print("[PeopleSeg] YuNet (OpenCV FaceDetectorYN): OK")
            else:
                print("[PeopleSeg] YuNet weights missing; skip OpenCV fallback.")
        except Exception as e:
            print(f"[PeopleSeg] YuNet/OpenCV недоступен: {e}")

        meta = {
            "paths": paths,
            "versions": {
                "ultralytics": getattr(__import__("ultralytics"), "__version__", "unknown"),
                "mediapipe": mp_version,
                "opencv": cv_version,
                "torch": torch.__version__,
            },
            "availability": {
                "mediapipe": face_primary is not None,
                "yunet_opencv": face_fallback is not None,
                "yolo_sizes": ["s"],
            },
        }

        return ({
            "seg_model": seg_model,
            "face_primary": face_primary,
            "face_fallback": face_fallback,
            "meta": meta,
        },)

# ---------------------------
# Node 2: Get Masks (rot+ellipse)
# ---------------------------

class ImagePeopleSegGetMasks:
    """
    YOLOv8-seg (person) -> головы (MediaPipe/YuNet/OpenCV -> rot-search -> ellipse|rect) -> морфология -> согласование тел.
    Голова НЕ вырезается из тела.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "IMAGE": ("IMAGE",),
                "PEOPLE_MODELS": ("PEOPLE_MODELS",),
                "max_side_process": ("INT", {"default": 1280, "min": 512, "max": 4096, "step": 64}),
                # чувствительность лица
                "face_conf_primary": ("FLOAT", {"default": 0.50, "min": 0.10, "max": 0.90, "step": 0.05}),
                "face_conf_fallback": ("FLOAT", {"default": 0.70, "min": 0.10, "max": 0.99, "step": 0.01}),
                "head_expand_ratio": ("FLOAT", {"default": 1.35, "min": 1.00, "max": 1.80, "step": 0.05}),
                "head_fallback_top_frac": ("FLOAT", {"default": 0.30, "min": 0.20, "max": 0.50, "step": 0.05}),
                # повороты и геометрия
                "rot_search": ("BOOLEAN", {"default": True}),
                "rot_search_preset": (["±90", "±45"], {"default": "±90"}),
                "head_geom": (["ellipse", "rect"], {"default": "ellipse"}),
            }
        }

    RETURN_TYPES = ("PERSON_INSTANCES", "MASK_LIST_SOFT", "MASK_LIST_SOFT")
    RETURN_NAMES = ("INSTANCES", "LIST_MASK_BODY", "LIST_MASK_HEAD")
    FUNCTION = "execute"
    CATEGORY = "segmentation/people"
    OUTPUT_NODE = False

    # Константы (без UI)
    CONF_THRES = 0.35
    NMS_IOU = 0.60
    MAX_INST = 20
    MIN_AREA_PX_BODY = 800
    MIN_AREA_PX_HEAD = 300
    MORPH_CLOSE_PX = 1

    def _infer_yolo(self, model, img_bgr: np.ndarray, imgsz: int, device_str: str, half: bool):
        results = model.predict(
            source=img_bgr,
            imgsz=imgsz,
            conf=self.CONF_THRES,
            iou=self.NMS_IOU,
            max_det=self.MAX_INST,
            device=device_str,
            half=half,
            classes=[0],  # person
            verbose=False,
        )
        return results

    def _best_head_mask_from_candidates(
        self,
        cand_abs: List[Dict[str, Any]],
        body_mask_bin: torch.Tensor,
        xywh_body: List[int],
        head_expand_ratio: float,
        head_geom: str,
        head_fallback_top_frac: float,
        H: int,
        W: int,
    ) -> torch.Tensor:
        """Выбирает лучшего кандидата и строит маску головы (ellipse|rect)."""
        # top cap для подсказки
        top_cap_xywh = [xywh_body[0], xywh_body[1], xywh_body[2], int(round(xywh_body[3] * head_fallback_top_frac))]

        device = body_mask_bin.device
        best = None
        best_combo = -1e9

        if cand_abs:
            body_np = body_mask_bin.detach().cpu().numpy()
            for c in cand_abs:
                rect = c["xywh"]
                det_s = float(c.get("score", 0.0))
                ang = c.get("angle_orig")  # угол в исходной системе, если известен
                # доля пикселей внутри тела
                x, y, w, h = rect
                x2, y2 = min(W, x + w), min(H, y + h)
                x, y = max(0, x), max(0, y)
                w_eff, h_eff = max(0, x2 - x), max(0, y2 - y)
                if w_eff == 0 or h_eff == 0:
                    continue
                inside = float(body_np[y:y + h_eff, x:x + w_eff].sum()) / float(max(1, w_eff * h_eff))
                # iou с верхней «шапкой»
                iou_cap = self._rect_iou([x, y, w_eff, h_eff], top_cap_xywh)
                combo = det_s + 0.3 * inside + 0.2 * iou_cap
                if combo > best_combo:
                    best_combo = combo
                    best = {"rect": [x, y, w_eff, h_eff], "angle": ang}

        if best is not None:
            x, y, w, h = best["rect"]
            cx, cy = x + w / 2.0, y + h / 2.0
            angle = float(best["angle"]) if best["angle"] is not None else 0.0
            if head_geom == "ellipse":
                # полуоси из размеров лица + масштаб
                ax = 0.6 * h * head_expand_ratio
                by = 0.55 * w * head_expand_ratio
                # небольшое смещение вверх относительно лица
                cy_shift = cy - 0.12 * h * math.cos(math.radians(0))  # по вертикали кадра
                mask = _ellipse_mask(H, W, cx, cy_shift, ax, by, angle, device)
            else:  # rect
                scale = head_expand_ratio
                rw = int(round(w * scale))
                rh = int(round(h * scale))
                rx = int(round(cx - rw / 2.0))
                ry = int(round(cy - rh / 2.0))
                mask = _rect_mask_from_xywh(H, W, [rx, ry, rw, rh], device)
            return _intersect(mask, body_mask_bin)

        # Fallback — верхняя часть тела
        return _intersect(
            _rect_mask_from_xywh(H, W, [xywh_body[0], xywh_body[1], xywh_body[2], int(round(xywh_body[3] * head_fallback_top_frac))], device),
            body_mask_bin
        )

    @staticmethod
    def _rect_iou(a: List[int], b: List[int]) -> float:
        ax, ay, aw, ah = a
        bx, by, bw, bh = b
        ax2, ay2 = ax + aw, ay + ah
        bx2, by2 = bx + bw, by + bh
        ix1, iy1 = max(ax, bx), max(ay, by)
        ix2, iy2 = min(ax2, bx2), min(ay2, by2)
        iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
        inter = iw * ih
        union = aw * ah + bw * bh - inter
        return float(inter / union) if union > 0 else 0.0

    def execute(self, IMAGE: torch.Tensor, PEOPLE_MODELS: Dict[str, Any],
                max_side_process: int,
                face_conf_primary: float, face_conf_fallback: float,
                head_expand_ratio: float, head_fallback_top_frac: float,
                rot_search: bool, rot_search_preset: str, head_geom: str):

        if IMAGE.shape[0] != 1:
            raise RuntimeError("[PeopleSeg] Поддерживается только B=1. Передан батч.")

        cuda_ok = torch.cuda.is_available()
        device_str = "cuda" if cuda_ok else "cpu"
        half = cuda_ok

        if not isinstance(PEOPLE_MODELS, dict) or "seg_model" not in PEOPLE_MODELS:
            raise RuntimeError("[PeopleSeg] Неверный PEOPLE_MODELS. Сначала вызовите PeopleSegModelsSetup.")
        seg_model = PEOPLE_MODELS.get("seg_model")
        face_primary = PEOPLE_MODELS.get("face_primary")
        face_fallback = PEOPLE_MODELS.get("face_fallback")

        try:
            seg_model.to(device_str)
        except Exception:
            pass
        try:
            seg_model.model.eval()
        except Exception:
            pass

        img_bgr = _to_numpy_image_uint8(IMAGE)
        H, W = img_bgr.shape[:2]
        imgsz = min(max_side_process, max(H, W))

        tried_downscale = False
        while True:
            try:
                results = self._infer_yolo(seg_model, img_bgr, imgsz, device_str, half)
                break
            except RuntimeError as e:
                msg = str(e).lower()
                if ("out of memory" in msg or "cuda out of memory" in msg) and not tried_downscale:
                    imgsz = max(512, int(round(imgsz * 0.75)))
                    print(f"[PeopleSeg] OOM, автоповтор с imgsz={imgsz}")
                    tried_downscale = True
                    if cuda_ok:
                        torch.cuda.empty_cache()
                    continue
                raise

        res = results[0]
        if getattr(res, "masks", None) is None or res.masks is None or res.masks.data is None:
            print("[PeopleSeg] Людей не найдено.")
            try:
                seg_model.to("cpu")
            except Exception:
                pass
            if cuda_ok:
                torch.cuda.empty_cache()
            gc.collect()
            return ([], [], [])

        mdata = res.masks.data.detach().float().cpu()  # (N,h',w')
        N = mdata.shape[0]
        bodies_soft_resized = F.interpolate(mdata[:, None, :, :], size=(H, W), mode="bilinear", align_corners=False)[:, 0]
        bodies_bin = (bodies_soft_resized > 0.5).float()
        bodies_bin = _morph_close(bodies_bin, self.MORPH_CLOSE_PX)
        bodies_bin = _morph_open(bodies_bin, 1)

        try:
            xyxy = res.boxes.xyxy.detach().cpu().numpy()
            confs = res.boxes.conf.detach().cpu().numpy()
            classes = res.boxes.cls.detach().cpu().numpy() if hasattr(res.boxes, "cls") else np.zeros((N,), dtype=np.float32)
        except Exception:
            xyxy = np.zeros((N, 4), dtype=np.float32)
            confs = np.zeros((N,), dtype=np.float32)
            classes = np.zeros((N,), dtype=np.float32)

        keep_idx = [i for i in range(N) if int(classes[i]) == 0]
        if len(keep_idx) != N:
            bodies_bin = bodies_bin[keep_idx]
            xyxy = xyxy[keep_idx]
            confs = confs[keep_idx]
            N = bodies_bin.shape[0]
            if N == 0:
                print("[PeopleSeg] Людей не найдено после фильтра класса.")
                try:
                    seg_model.to("cpu")
                except Exception:
                    pass
                if cuda_ok:
                    torch.cuda.empty_cache()
                gc.collect()
                return ([], [], [])

        bboxes_xywh: List[List[int]] = []
        conf_list: List[float] = []
        valid_mask = []
        for i in range(N):
            area = int(bodies_bin[i].sum().item())
            xywh = _bbox_xyxy_to_xywh(xyxy[i], H, W)
            if area >= self.MIN_AREA_PX_BODY and xywh[2] > 0 and xywh[3] > 0:
                bboxes_xywh.append(xywh)
                conf_list.append(float(confs[i]))
                valid_mask.append(True)
            else:
                valid_mask.append(False)

        if not any(valid_mask):
            print("[PeopleSeg] Все тела отфильтрованы (слишком маленькие).")
            try:
                seg_model.to("cpu")
            except Exception:
                pass
            if cuda_ok:
                torch.cuda.empty_cache()
            gc.collect()
            return ([], [], [])

        bodies_bin = bodies_bin[valid_mask]
        xyxy = xyxy[valid_mask]
        N = bodies_bin.shape[0]

        bodies_bin = _resolve_overlaps_by_centroids(bodies_bin, bboxes_xywh)

        # ----- HEADS -----
        instances: List[Dict[str, Any]] = []
        head_masks_bin: List[torch.Tensor] = []

        # углы поворота
        angles = [0.0]
        if rot_search:
            if rot_search_preset == "±45":
                angles = [0.0, 45.0, -45.0]
            else:
                angles = [0.0, 45.0, -45.0, 90.0, -90.0]

        for idx in range(N):
            xywh = bboxes_xywh[idx]
            score_body = conf_list[idx]
            body_mask_bin = bodies_bin[idx].clone()

            crop_bgr, (x0, y0) = _crop_with_pad(img_bgr, xywh, pad_ratio=0.05)
            crop_rgb0 = crop_bgr[:, :, ::-1].copy()  # RGB
            ch, cw = crop_rgb0.shape[:2]

            candidates_abs: List[Dict[str, Any]] = []

            for ang in angles:
                # rotate crop and detect
                crop_rgb = _rotate_rgb_u8_same_size(crop_rgb0, ang)
                # MediaPipe
                if face_primary is not None:
                    try:
                        faces = face_primary.detect(crop_rgb)
                        for d in faces:
                            if d["score"] < face_conf_primary:
                                continue
                            rect_rot = d["xywh"]
                            rect_src = _map_rect_from_rot_to_src(rect_rot, ang, cw, ch)
                            # map to absolute image coords
                            rect_abs = [x0 + rect_src[0], y0 + rect_src[1], rect_src[2], rect_src[3]]
                            # angle in rotated frame -> original
                            a_det = d.get("angle")
                            a_orig = (a_det if a_det is not None else 0.0) - ang
                            candidates_abs.append({"xywh": rect_abs, "score": float(d["score"]), "angle_orig": a_orig})
                    except Exception as e:
                        print(f"[PeopleSeg] MediaPipe detect error: {e}")
                # YuNet
                if face_fallback is not None:
                    try:
                        faces = face_fallback.detect(crop_rgb)
                        for d in faces:
                            if d["score"] < face_conf_fallback:
                                continue
                            rect_rot = d["xywh"]
                            rect_src = _map_rect_from_rot_to_src(rect_rot, ang, cw, ch)
                            rect_abs = [x0 + rect_src[0], y0 + rect_src[1], rect_src[2], rect_src[3]]
                            a_det = d.get("angle")
                            a_orig = (a_det if a_det is not None else 0.0) - ang
                            candidates_abs.append({"xywh": rect_abs, "score": float(d["score"]), "angle_orig": a_orig})
                    except Exception as e:
                        print(f"[PeopleSeg] YuNet/OpenCV detect error: {e}")

            head_mask_bin = self._best_head_mask_from_candidates(
                candidates_abs, body_mask_bin, xywh,
                head_expand_ratio, head_geom, head_fallback_top_frac,
                H, W
            )

            head_masks_bin.append(head_mask_bin)

            area = int(body_mask_bin.sum().item())
            instances.append({
                "id": idx,
                "score": float(score_body),
                "bbox": xywh,
                "area": area,
            })

        # Итоги (0/1 float)
        list_mask_body: List[torch.Tensor] = [b.clone().detach().cpu().float() for b in bodies_bin]
        list_mask_head: List[torch.Tensor] = [h.clone().detach().cpu().float() for h in head_masks_bin]

        try:
            seg_model.to("cpu")
        except Exception:
            pass
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

        return (instances, list_mask_body, list_mask_head)

# ---------------------------
# Register
# ---------------------------

NODE_CLASS_MAPPINGS = {
    "PeopleSegModelsSetup": PeopleSegModelsSetup,
    "ImagePeopleSegGetMasks": ImagePeopleSegGetMasks,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "PeopleSegModelsSetup": "🧩 People Seg — Models Setup",
    "ImagePeopleSegGetMasks": "🧩 People Seg — Get Masks",
}
