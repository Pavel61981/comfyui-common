import re

class StringFormatABC:
    """
    Нода "string_format_abc" — формирует строку по шаблону с плейсхолдерами {A},{B},{C},{D},{E}.

    Входы:
        - A,B,C,D,E (STRING): строки для подстановки.
        - Template (STRING): шаблон, содержащий плейсхолдеры вида {A}..{E}.

    Правила:
        - Разрешены ТОЛЬКО {A},{B},{C},{D},{E}.
        - Пустые строки валидны и подставляются как есть.
        - Любые другие плейсхолдеры (например {X}) или лишние фигурные скобки — ошибка.

    Выход:
        - Result (STRING): результат подстановки.
    """

    _ALLOWED_KEYS = {"A", "B", "C", "D", "E"}
    _PLACEHOLDER_RE = re.compile(r"\{([A-Z]+)\}")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "A": ("STRING", {"default": ""}),
                "B": ("STRING", {"default": ""}),
                "C": ("STRING", {"default": ""}),
                "D": ("STRING", {"default": ""}),
                "E": ("STRING", {"default": ""}),
                "Template": ("STRING", {"default": "{A}{B}{C}"}),
            }
        }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "execute"
    CATEGORY = "utils/text"
    OUTPUT_NODE = False

    def _validate_string(self, value, name):
        if not isinstance(value, str):
            raise RuntimeError(f"[string_format_abc] '{name}' должен быть строкой, получено: {type(value).__name__}")
        return value

    def _validate_template(self, template):
        """
        Проверяет, что в шаблоне используются только допустимые плейсхолдеры,
        и что нет «голых» фигурных скобок.
        Возвращает список найденных ключей-плейсхолдеров (в порядке появления).
        """
        # Найдём все подпоследовательности вида {...}
        matches = list(self._PLACEHOLDER_RE.finditer(template))

        # Проверка на недопустимые плейсхолдеры
        keys = []
        for m in matches:
            key = m.group(1)
            if key not in self._ALLOWED_KEYS:
                raise RuntimeError(f"[string_format_abc] Недопустимый плейсхолдер '{{{key}}}' в Template")
            keys.append(key)

        # Убедимся, что в строке не осталось «голых» фигурных скобок
        # Заменим корректные плейсхолдеры на маркер и проверим остаток
        tmp = self._PLACEHOLDER_RE.sub("§", template)
        if "{" in tmp or "}" in tmp:
            raise RuntimeError("[string_format_abc] Обнаружены недопустимые фигурные скобки в Template")

        return keys

    def execute(self, A, B, C, D, E, Template):
        """
        Выполняет подстановку по шаблону. Возвращает кортеж (Result,).
        """
        try:
            # Валидация типов
            A = self._validate_string(A, "A")
            B = self._validate_string(B, "B")
            C = self._validate_string(C, "C")
            D = self._validate_string(D, "D")
            E = self._validate_string(E, "E")
            template = self._validate_string(Template, "Template")

            # Проверка шаблона и сбор использованных ключей
            _ = self._validate_template(template)

            # Карта значений для подстановки
            values = {"A": A, "B": B, "C": C, "D": D, "E": E}

            # Функция-заменитель для re.sub
            def replacer(match):
                key = match.group(1)
                return values.get(key, "")

            result = self._PLACEHOLDER_RE.sub(replacer, template)
            return (result,)

        except RuntimeError:
            raise
        except Exception as e:
            msg = f"[string_format_abc] Неожиданная ошибка: {str(e)}"
            print(msg)
            raise RuntimeError(msg) from e


# Регистрация ноды в ComfyUI
NODE_CLASS_MAPPINGS = {"StringFormatABC": StringFormatABC}
NODE_DISPLAY_NAME_MAPPINGS = {"StringFormatABC": "🧩 string_format_abc"}
