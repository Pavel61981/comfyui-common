# filename: image_is_not_empty.py

import torch


class ImageIsNotEmpty:
    """
    Нода-валидатор "ImageIsNotEmpty".
    Проверяет входной IMAGE и выбрасывает RuntimeError, если он пустой/некорректный:
      - None/не tensor/пустой тензор/невалидные размеры,
      - NaN/Inf,
      - одноцветность по RGB (max-min ≤ UNIFORM_TOL),
      - почти полная прозрачность по альфа-каналу (если RGBA).
    При валидности возвращает входной IMAGE (пасстру), чтобы гарантировать выполнение ноды в графе.
    """

    # Жёсткие константы (фиксированы по ТЗ)
    CHECK_NULL = True
    CHECK_NAN_INF = True
    CHECK_UNIFORM = True
    UNIFORM_TOL = 1e-5
    CHECK_TRANSPARENT = True
    OPAQUE_RATIO_MIN = 0.01
    ALPHA_EPS = 1e-6

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"Image": ("IMAGE",)}}

    # Пасстру-выход: тот же IMAGE
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("Image",)
    FUNCTION = "execute"
    CATEGORY = "utils/validation"
    OUTPUT_NODE = False

    # -------- internal helpers --------

    def _raise(self, msg: str):
        """Поднять RuntimeError с префиксом ноды."""
        raise RuntimeError(f"[ImageIsNotEmpty] {msg}")

    def _normalize_image(self, image: torch.Tensor) -> torch.Tensor:
        """
        Приводит изображение к форме [B,H,W,C], если оно 3D [H,W,C] — добавляет размер батча.
        Возвращает ссылку/перестроенный тензор без изменения данных.
        """
        if image.dim() == 3:
            image = image.unsqueeze(0)
        return image

    # -------- main --------

    def execute(self, Image):
        """
        Проверяет входной IMAGE и выбрасывает RuntimeError при нарушениях.
        При успехе — возвращает входной IMAGE (пасстру), чтобы нода гарантированно исполнялась в графе.
        """
        try:
            # 1) Null / пустой вход / базовые размеры
            if self.CHECK_NULL:
                if Image is None:
                    self._raise("EmptyInput: input is None.")
                if not isinstance(Image, torch.Tensor):
                    self._raise("EmptyInput: expected torch.Tensor of type IMAGE.")
                # При некоторых состояниях .numel() обращение допустимо только у Tensor
                if Image.numel() == 0:
                    self._raise("EmptyInput: tensor has zero elements.")

            img = self._normalize_image(Image)

            if img.dim() != 4:
                self._raise(
                    f"EmptyInput: invalid tensor dims {img.dim()} (expected 4 or 3)."
                )

            b, h, w, c = img.shape
            if b == 0 or h == 0 or w == 0:
                self._raise("EmptyInput: zero batch or spatial dimension.")
            if c not in (3, 4):
                self._raise(
                    f"EmptyInput: invalid channel count C={c} (expected 3 or 4)."
                )

            # 2) Поэлементные проверки (останавливаемся на первом невалидном)
            with torch.no_grad():
                for i in range(b):
                    frame = img[i]  # [H,W,C]

                    # 2.1) NaN / Inf
                    if self.CHECK_NAN_INF:
                        # isfinite == not (NaN or +/-Inf)
                        if not torch.isfinite(frame).all():
                            self._raise(f"InvalidNumbers at index {i}: found NaN/Inf.")

                    # 2.2) Одноцветность (RGB)
                    if self.CHECK_UNIFORM:
                        rgb = frame[..., :3].reshape(-1)  # [H*W*3]
                        # Если картинка пустая по данным, сюда бы не дошли (numel проверили выше)
                        rgb_max = torch.max(rgb)
                        rgb_min = torch.min(rgb)
                        delta = (rgb_max - rgb_min).item()
                        if delta <= self.UNIFORM_TOL:
                            self._raise(
                                f"UniformImage at index {i} (max-min={delta:.2e} ≤ tol={self.UNIFORM_TOL:.2e})."
                            )

                    # 2.3) Прозрачность (если RGBA)
                    if self.CHECK_TRANSPARENT and c == 4:
                        alpha = frame[..., 3].reshape(-1)  # [H*W]
                        threshold = 1.0 - self.ALPHA_EPS
                        # Доля пикселей, которые почти полностью непрозрачны
                        opaque_ratio = (alpha >= threshold).float().mean().item()
                        if opaque_ratio < self.OPAQUE_RATIO_MIN:
                            self._raise(
                                f"FullyTransparent at index {i} (opaque_ratio={opaque_ratio:.3f} < {self.OPAQUE_RATIO_MIN:.2f})."
                            )

            # Всё хорошо — возвращаем исходный тензор без изменений (пасстру)
            return (Image,)

        except RuntimeError:
            # Сообщения уже корректно оформлены
            raise
        except Exception as e:
            self._raise(f"Unexpected error: {str(e)}")


# Регистрация ноды в ComfyUI
NODE_CLASS_MAPPINGS = {"ImageIsNotEmpty": ImageIsNotEmpty}
NODE_DISPLAY_NAME_MAPPINGS = {"ImageIsNotEmpty": "🧪 Image: Is Not Empty"}
