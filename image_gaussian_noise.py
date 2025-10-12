# filename: image_gaussian_noise.py
import torch


class ImageGaussianNoise:
    """
    Нода "ImageGaussianNoise" — добавляет к входному IMAGE аддитивный гауссов шум.

    Параметры:
      - Sigma: стандартное отклонение шума (0..1).
      - Seed: -1 для случайного шума; >=0 для детерминированного (для батча используется Seed + i).

    Вход:
      - Image: torch.Tensor[B, H, W, 3] со значениями в [0..1].

    Выход:
      - Noisy_Image: torch.Tensor[B, H, W, 3] со значениями в [0..1].
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "Image": ("IMAGE",),
                "Sigma": (
                    "FLOAT",
                    {"default": 0.10, "min": 0.0, "max": 1.0, "step": 0.01},
                ),
                "Seed": ("INT", {"default": -1, "min": -1, "step": 1}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("Noisy_Image",)
    FUNCTION = "execute"
    CATEGORY = "utils/noise"
    OUTPUT_NODE = False

    @staticmethod
    def _check_image(image_tensor):
        """Проверяет тензор изображения на соответствие формату [B, H, W, 3]."""
        if image_tensor is None:
            raise RuntimeError("[ImageGaussianNoise] Пустой вход Image")

        if not torch.is_tensor(image_tensor):
            raise RuntimeError("[ImageGaussianNoise] Image должен быть torch.Tensor")

        if image_tensor.ndim != 4 or image_tensor.shape[-1] != 3:
            raise RuntimeError(
                f"[ImageGaussianNoise] Ожидается форма [B, H, W, 3], получено {tuple(image_tensor.shape)}"
            )

    @staticmethod
    def _make_noise_like(image_tensor, seed):
        """
        Создаёт шум той же формы и на том же устройстве, что и image_tensor.
        При seed >= 0 генерирует детерминированный шум для каждого элемента батча c seed + i.
        """
        B, H, W, C = image_tensor.shape
        device = image_tensor.device
        dtype = torch.float32

        if seed is None or seed < 0:
            # Случайный шум одним вызовом
            return torch.randn((B, H, W, C), device=device, dtype=dtype)

        # Детерминированный шум: отдельный seed на элемент батча
        noise = torch.empty((B, H, W, C), device=device, dtype=dtype)
        for i in range(B):
            g = torch.Generator()  # CPU-генератор (совместимо везде)
            g.manual_seed(int(seed) + int(i))
            n_i = torch.randn((H, W, C), generator=g, dtype=dtype)
            # Копия на устройство входа (GPU/CPU)
            if device.type != "cpu":
                n_i = n_i.to(device)
            noise[i] = n_i
        return noise

    def execute(self, Image, Sigma, Seed):
        """
        Добавляет гауссов шум к Image и клампит результат в [0, 1].
        Возвращает (Noisy_Image,).
        """
        try:
            self._check_image(Image)

            # Приведение к float32, сохранение устройства
            img = Image
            if img.dtype != torch.float32:
                img = img.to(torch.float32)

            sigma = float(Sigma)
            if sigma < 0.0:
                sigma = 0.0
            if sigma == 0.0:
                # Возвращаем копию, чтобы не мутировать вход
                return (img.clone(),)

            # Генерация шума
            noise = self._make_noise_like(img, int(Seed))

            # Применение шума и кламп
            noisy = torch.clamp(img + sigma * noise, 0.0, 1.0)

            return (noisy,)

        except Exception as e:
            msg = f"[ImageGaussianNoise] Ошибка: {str(e)}"
            print(msg)
            raise RuntimeError(msg) from e


# Регистрация ноды в ComfyUI
NODE_CLASS_MAPPINGS = {"ImageGaussianNoise": ImageGaussianNoise}
NODE_DISPLAY_NAME_MAPPINGS = {"ImageGaussianNoise": "🧩 Image Gaussian Noise"}
