import os
import torch
import numpy as np
from PIL import Image
import folder_paths


class SaveImageWithFormat:
    """
    Кастомная нода для ComfyUI, которая сохраняет одно изображение в выбранном формате
    с настройками для максимального качества (без потерь или с минимальными потерями).
    """

    # Указываем директорию для вывода по умолчанию
    OUTPUT_DIR = folder_paths.get_output_directory()
    # Список поддерживаемых форматов для сохранения
    SUPPORTED_FORMATS = ["png", "webp", "tiff", "jpeg"]

    # Категория, в которой будет находиться нода
    CATEGORY = "image"

    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        """
        Определение входных параметров ноды.
        """
        return {
            "required": {
                "image": ("IMAGE",),
                "directory_path": (
                    "STRING",
                    {"default": s.OUTPUT_DIR, "multiline": False},
                ),
                "file_name": ("STRING", {"default": "image_name", "multiline": False}),
                "file_extension": (s.SUPPORTED_FORMATS, {"default": "png"}),
            },
            "optional": {
                "subdirectory_name": ("STRING", {"default": "", "multiline": False}),
            },
        }

    # Типы данных, которые нода возвращает
    RETURN_TYPES = (
        "IMAGE",
        "STRING",
    )
    # Имена выходных слотов
    RETURN_NAMES = (
        "image",
        "full_path",
    )

    # Указываем, что у ноды нет вывода в интерфейсе
    OUTPUT_NODE = True

    # Указываем, какая функция будет выполняться
    FUNCTION = "save_image"

    def save_image(
        self,
        image: torch.Tensor,
        directory_path: str,
        file_name: str,
        file_extension: str,
        subdirectory_name: str = "",
    ):
        """
        Основная логика работы ноды.
        """
        # --- 1. Валидация и сборка пути ---

        clean_dir = directory_path.strip()
        clean_subdir = subdirectory_name.strip().strip("/\\")
        clean_filename_base = file_name.strip().strip("/\\")

        # base_name, extension = os.path.splitext(clean_filename_base)
        # if extension:
        #     raise ValueError(
        #         f"Имя файла '{file_name}' не должно содержать расширение. Введите только имя."
        #     )

        final_filename = f"{clean_filename_base}.{file_extension}"
        full_path = os.path.join(clean_dir, clean_subdir, final_filename)
        output_dir = os.path.dirname(full_path)
        os.makedirs(output_dir, exist_ok=True)

        # --- 2. Конвертация тензора в изображение ---

        single_image = image[0]
        img_array = 255.0 * single_image.cpu().numpy()
        img_array = np.clip(img_array, 0, 255).astype(np.uint8)
        img = Image.fromarray(img_array)

        # --- 3. Подготовка к сохранению ---

        # ИСПРАВЛЕНИЕ: Если формат JPEG, а изображение имеет альфа-канал (RGBA),
        # конвертируем его в RGB, чтобы избежать ошибки.
        if file_extension == "jpeg" and img.mode == "RGBA":
            # Создаем белое фоновое изображение того же размера
            background = Image.new("RGB", img.size, (255, 255, 255))
            # Накладываем исходное изображение (с его альфа-каналом) на белый фон
            background.paste(img, mask=img.split()[3])
            img = background

        # Выбор параметров сохранения в зависимости от формата
        save_options = {}
        if file_extension == "png":
            save_options = {
                "compress_level": 6
            }  # Быстро + без потерь + разумный размер
        elif file_extension == "webp":
            save_options = {"lossless": True, "quality": 100}
        elif file_extension == "jpeg":
            save_options = {"quality": 100, "subsampling": 0}
            if img.mode != "RGB":
                img = img.convert("RGB")
        elif file_extension == "tiff":
            save_options = {"compression": "none"}

        # Сохраняем изображение с выбранными параметрами
        try:
            img.save(full_path, **save_options)
        except Exception as e:
            raise RuntimeError(f"Ошибка при сохранении изображения: {e}")

        return (
            image,
            full_path,
        )


# --- Регистрация ноды в ComfyUI ---
NODE_CLASS_MAPPINGS = {"SaveImageWithFormat": SaveImageWithFormat}
NODE_DISPLAY_NAME_MAPPINGS = {"SaveImageWithFormat": "💾 Image save with format"}
