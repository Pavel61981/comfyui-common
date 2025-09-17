import re


class TextReplacer:
    """
    Кастомная нода для ComfyUI, которая производит замену текста по ключам.
    Она принимает на вход текст и список замен в формате "ключ|значение".
    Поиск ключей производится без учета регистра.
    """

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "text": ("STRING", {"forceInput": True}),
                "replacements": (
                    "STRING",
                    {"multiline": True, "default": "key1|value1\nkey2|value2"},
                ),
            },
        }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "replace"
    CATEGORY = "text"

    def replace(self, text, replacements):
        # Разделяем многострочный текст на отдельные строки
        lines = replacements.strip().split("\n")

        # Проходим по каждой строке с заменами
        for line in lines:
            # Пропускаем пустые строки
            if not line.strip():
                continue

            # Разделяем строку на ключ и значение по символу "|"
            parts = line.split("|", 1)
            if len(parts) == 2:
                key, value = parts
                # Используем регулярные выражения для замены без учета регистра
                # re.escape(key.strip()) для экранирования спецсимволов в ключе
                try:
                    text = re.sub(
                        re.escape(key.strip()), value.strip(), text, flags=re.IGNORECASE
                    )
                except re.error as e:
                    print(
                        f"Ошибка регулярного выражения для ключа '{key.strip()}': {e}"
                    )

        return (text,)


# Словарь для регистрации ноды в ComfyUI
NODE_CLASS_MAPPINGS = {"TextReplacerKeyValue": TextReplacer}

# Словарь для отображаемого имени ноды в интерфейсе
NODE_DISPLAY_NAME_MAPPINGS = {"TextReplacerKeyValue": "Text Replacer (Key-Value)"}
