# file: string_json_prompt_parser.py
import json


class StringJsonPromptParser:
    """
    Нода "StringJsonPromptParser" — парсит JSON-строку вида:
    {"photo_type": "__photo_type__", "prompt": "__prompt__"}
    и возвращает два значения: photo_type и prompt (оба строки).
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "JSON_String": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": "",
                        "placeholder": '{"photo_type": "...", "prompt": "..."}',
                    },
                ),
            }
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("photo_type", "prompt")
    FUNCTION = "execute"
    CATEGORY = "utils/json"
    OUTPUT_NODE = False

    def execute(self, JSON_String: str):
        """
        Парсит JSON_String, валидирует схему и возвращает (photo_type, prompt).
        При ошибках выбрасывает RuntimeError.
        """
        try:
            data = json.loads(JSON_String)
        except Exception as e:
            msg = f"[StringJsonPromptParser] Ошибка парсинга JSON: {str(e)}"
            print(msg)
            raise RuntimeError(msg) from e

        if not isinstance(data, dict):
            msg = "[StringJsonPromptParser] Ожидался JSON-объект (dict), получено иное."
            print(msg)
            raise RuntimeError(msg)

        required_keys = ("photo_type", "prompt")
        for key in required_keys:
            if key not in data:
                msg = f"[StringJsonPromptParser] Отсутствует обязательный ключ '{key}'."
                print(msg)
                raise RuntimeError(msg)
            if not isinstance(data[key], str):
                msg = f"[StringJsonPromptParser] Значение по ключу '{key}' должно быть строкой."
                print(msg)
                raise RuntimeError(msg)

        photo_type = data["photo_type"].strip()
        prompt = data["prompt"].strip()
        return (photo_type, prompt)


# Регистрация ноды в ComfyUI
NODE_CLASS_MAPPINGS = {
    "StringJsonPromptParser": StringJsonPromptParser,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "StringJsonPromptParser": "🧩 JSON → photo_type & prompt",
}
