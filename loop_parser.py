# filename: string_loop_json_parse.py
import json


class StringLoopJsonParse:
    """
    Нода "StringLoopJsonParse" — принимает на вход LOOP (любой тип, "*"):
    - либо строку с JSON;
    - либо уже разобранный JSON-объект (dict).

    Возвращает строго определенные поля:
    start(INT), end(INT), step(INT), index(INT), finished(BOOLEAN), condition_open(BOOLEAN).

    Пример допустимого входа (строка):
    {
      "start": 1,
      "end": 10,
      "step": 1,
      "index": 1,
      "finished": false,
      "condition_open": true
    }
    """

    @classmethod
    def INPUT_TYPES(cls):
        # Для "*" поля ComfyUI не рисует текстовое поле; порт только для соединений.
        # Оставляем только подключаемый вход.
        return {
            "required": {
                "LOOP": ("*",),
            }
        }

    RETURN_TYPES = ("INT", "INT", "INT", "INT", "BOOLEAN", "BOOLEAN")
    RETURN_NAMES = ("start", "end", "step", "index", "finished", "condition_open")
    FUNCTION = "execute"
    CATEGORY = "utils/json"
    OUTPUT_NODE = False

    def _is_int_strict(self, value):
        """
        Строгая проверка целого числа.
        Исключает bool, т.к. bool является подклассом int в Python.
        """
        return isinstance(value, int) and not isinstance(value, bool)

    def _normalize_input(self, loop_input):
        """
        Приводит вход LOOP к dict:
        - если строка -> json.loads
        - если dict -> вернуть как есть
        Иначе -> RuntimeError
        """
        # Строка JSON
        if isinstance(loop_input, str):
            try:
                parsed = json.loads(loop_input)
            except Exception as e:
                msg = f"[loop_json_parse] Invalid JSON: {str(e)}"
                print(msg)
                raise RuntimeError(msg) from e
            return parsed

        # Уже разобранный объект
        if isinstance(loop_input, dict):
            return loop_input

        # Ничего другого не поддерживаем (например, list, tuple, bytes и т.п.)
        msg = f"[loop_json_parse] Unsupported input type for LOOP: {type(loop_input).__name__}"
        print(msg)
        raise RuntimeError(msg)

    def execute(self, LOOP):
        """
        Преобразует вход LOOP к dict, валидирует наличие и типы ключей,
        возвращает значения в порядке, указанном в RETURN_TYPES/RETURN_NAMES.
        """
        data = self._normalize_input(LOOP)

        if not isinstance(data, dict):
            msg = "[loop_json_parse] Expected a JSON object at top level"
            print(msg)
            raise RuntimeError(msg)

        required_schema = {
            "start": "INT",
            "end": "INT",
            "step": "INT",
            "index": "INT",
            "finished": "BOOLEAN",
            "condition_open": "BOOLEAN",
        }

        # Проверка наличия ключей
        missing = [k for k in required_schema.keys() if k not in data]
        if missing:
            msg = f"[loop_json_parse] Missing key: {', '.join(missing)}"
            print(msg)
            raise RuntimeError(msg)

        # Извлечение и проверка типов
        errors = []

        start = data.get("start")
        end = data.get("end")
        step = data.get("step")
        index = data.get("index")
        finished = data.get("finished")
        condition_open = data.get("condition_open")

        if not self._is_int_strict(start):
            errors.append("Invalid type for 'start': expected INT")
        if not self._is_int_strict(end):
            errors.append("Invalid type for 'end': expected INT")
        if not self._is_int_strict(step):
            errors.append("Invalid type for 'step': expected INT")
        if not self._is_int_strict(index):
            errors.append("Invalid type for 'index': expected INT")

        if not isinstance(finished, bool):
            errors.append("Invalid type for 'finished': expected BOOLEAN")
        if not isinstance(condition_open, bool):
            errors.append("Invalid type for 'condition_open': expected BOOLEAN")

        if errors:
            msg = "[loop_json_parse] " + "; ".join(errors)
            print(msg)
            raise RuntimeError(msg)

        return (start, end, step, index, finished, condition_open)


# Регистрация ноды в ComfyUI
NODE_CLASS_MAPPINGS = {"StringLoopJsonParse": StringLoopJsonParse}
NODE_DISPLAY_NAME_MAPPINGS = {"StringLoopJsonParse": "LOOP Parser"}
