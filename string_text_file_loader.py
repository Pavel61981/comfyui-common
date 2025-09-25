# file: string_text_file_loader.py
import os


class StringTextFileLoader:
    """
    Нода "StringTextFileLoader" — читает текстовый файл UTF-8, путь собирается из:
    Directory + Filename + Extension. Возвращает содержимое файла строкой.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "Directory": ("STRING", {"default": "", "placeholder": "/path/to/dir"}),
                "Filename": (
                    "STRING",
                    {"default": "", "placeholder": "file_name_without_ext"},
                ),
                "Extension": ("STRING", {"default": "txt", "placeholder": "txt"}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("Text",)
    FUNCTION = "execute"
    CATEGORY = "io/text"
    OUTPUT_NODE = False

    @staticmethod
    def _validate_name_component(name: str, field: str):
        """
        Валидирует Filename/Extension: запрет разделителей пути и '..'.
        """
        if name is None:
            raise RuntimeError(
                f"[StringTextFileLoader] Поле '{field}' не должно быть пустым."
            )
        forbidden = ["..", "/", "\\"]
        sep = os.sep
        altsep = os.altsep if os.altsep else None
        for token in forbidden + ([sep] if sep else []) + ([altsep] if altsep else []):
            if token and token in name:
                raise RuntimeError(
                    f"[StringTextFileLoader] Недопустимый символ/последовательность в '{field}': '{token}'."
                )
        if name.strip() == "":
            raise RuntimeError(
                f"[StringTextFileLoader] Поле '{field}' не должно быть пустой строкой."
            )

    def execute(self, Directory: str, Filename: str, Extension: str):
        """
        Формирует путь и читает текстовый файл в кодировке UTF-8.
        При любой ошибке выбрасывает RuntimeError.
        """
        try:
            filename = (Filename or "").strip()
            ext = (Extension or "").strip().lower()
            if ext.startswith("."):
                ext = ext[1:]

            self._validate_name_component(filename, "Filename")
            self._validate_name_component(ext, "Extension")

            path = os.path.join(Directory, f"{filename}.{ext}")

            if not os.path.exists(path):
                raise RuntimeError(f"[StringTextFileLoader] Файл не найден: {path}")
            if not os.path.isfile(path):
                raise RuntimeError(
                    f"[StringTextFileLoader] Указанный путь не является файлом: {path}"
                )

            with open(path, "r", encoding="utf-8") as f:
                text = f.read()

            return (text,)

        except RuntimeError:
            raise
        except Exception as e:
            msg = f"[StringTextFileLoader] Ошибка чтения файла: {str(e)}"
            print(msg)
            raise RuntimeError(msg) from e


# Регистрация ноды в ComfyUI
NODE_CLASS_MAPPINGS = {
    "StringTextFileLoader": StringTextFileLoader,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "StringTextFileLoader": "🧩 Load Text File",
}
