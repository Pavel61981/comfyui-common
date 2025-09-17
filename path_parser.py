import os


class PathParser:
    """
    Нода для разбора пути к файлу.
    Она принимает полный путь к файлу и возвращает его компоненты.
    """

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "file_path": ("STRING", {"default": "C:\\folder\\filename.txt"}),
            },
            "optional": {
                "new_extension": ("STRING", {"default": ""}),
                "prefix": ("STRING", {"default": ""}),
                "postfix": ("STRING", {"default": ""}),
            },
        }

    RETURN_TYPES = (
        "STRING",
        "STRING",
        "STRING",
        "STRING",
        "STRING",
        "STRING",
        "STRING",
        "STRING",
    )
    RETURN_NAMES = (
        "full_path",
        "full_path_with_new_ext",
        "directory_path",
        "filename",
        "filename_with_original_ext",
        "filename_with_new_ext",
        "filename_with_prefix_postfix",
        "filename_with_prefix_postfix_ext",
    )

    FUNCTION = "parse_path"
    CATEGORY = "Utilities"

    def parse_path(self, file_path, new_extension="", prefix="", postfix=""):
        if not file_path:
            raise ValueError("file_path не может быть пустым")

        directory_path = os.path.dirname(file_path)
        filename_with_original_ext = os.path.basename(file_path)
        filename, original_extension = os.path.splitext(filename_with_original_ext)

        # Применяем префикс и постфикс к базовому имени файла
        filename_with_prefix_postfix = prefix + filename + postfix

        # Формируем имя файла с новым расширением
        new_extension = new_extension.strip()
        if new_extension and not new_extension.startswith("."):
            new_extension = "." + new_extension

        if new_extension:
            filename_with_new_ext = filename + new_extension
            filename_with_prefix_postfix_ext = (
                filename_with_prefix_postfix + new_extension
            )
        else:
            filename_with_new_ext = filename
            filename_with_prefix_postfix_ext = filename_with_prefix_postfix

        full_path_with_new_ext = os.path.join(
            directory_path, filename_with_prefix_postfix_ext
        )

        return (
            file_path,
            full_path_with_new_ext,
            directory_path,
            filename,
            filename_with_original_ext,
            filename_with_new_ext,
            filename_with_prefix_postfix,
            filename_with_prefix_postfix_ext,
        )


NODE_CLASS_MAPPINGS = {"PathParser": PathParser}
NODE_DISPLAY_NAME_MAPPINGS = {"PathParser": "Path Parser"}
