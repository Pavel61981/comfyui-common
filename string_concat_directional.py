class StringConcatDirectional:
    """
    Нода "string_concat_directional" — объединяет входной текст и пользовательский текст
    с возможностью расположить пользовательский текст перед или после входного.

    Входы:
        - Input_Text (STRING): входной текст из графа.
        - User_Text  (STRING): пользовательский текст для добавления.
        - Mode       (CHOICE): "before" | "after" — положение User_Text.
        - Separator  (STRING): строка-разделитель между частями (по умолчанию пробел).
        - Trim       (CHOICE): "no" | "both" | "input_only" | "user_only" — обрезка пробелов.
        - Skip_Separator_If_Empty (BOOLEAN): пропускать разделитель, если один из текстов пуст.

    Выход:
        - Result (STRING): склеенная строка.

    Исключения:
        - RuntimeError при неверных типах входных значений.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "Input_Text": ("STRING", {"default": ""}),
                "User_Text": ("STRING", {"default": ""}),
                "Mode": (["before", "after"], {"default": "after"}),
                "Separator": ("STRING", {"default": " "}),
                "Trim": (["no", "both", "input_only", "user_only"], {"default": "no"}),
                "Skip_Separator_If_Empty": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "execute"
    CATEGORY = "utils/text"
    OUTPUT_NODE = False

    def _validate_str(self, value, name):
        if not isinstance(value, str):
            raise RuntimeError(f"[string_concat_directional] '{name}' должен быть строкой, получено: {type(value).__name__}")
        return value

    def _apply_trim(self, input_text, user_text, trim_mode):
        if trim_mode == "no":
            return input_text, user_text
        if trim_mode == "both":
            return input_text.strip(), user_text.strip()
        if trim_mode == "input_only":
            return input_text.strip(), user_text
        if trim_mode == "user_only":
            return input_text, user_text.strip()
        # На случай некорректного значения (не должен случиться из-за списка выбора)
        raise RuntimeError(f"[string_concat_directional] Некорректный режим Trim: {trim_mode}")

    def execute(self, Input_Text, User_Text, Mode, Separator, Trim, Skip_Separator_If_Empty):
        """
        Склеивает строки согласно выбранному режиму и настройкам.
        Возвращает кортеж с одной строкой (Result).
        """
        try:
            # Валидация типов
            input_text = self._validate_str(Input_Text, "Input_Text")
            user_text = self._validate_str(User_Text, "User_Text")
            separator = self._validate_str(Separator, "Separator")

            # Trim по настройке
            input_text, user_text = self._apply_trim(input_text, user_text, Trim)

            # Определяем, нужен ли разделитель
            if Skip_Separator_If_Empty:
                need_sep = (len(input_text) > 0) and (len(user_text) > 0)
            else:
                need_sep = True

            sep = separator if need_sep else ""

            # Склейка по режиму
            if Mode == "before":
                result = f"{user_text}{sep}{input_text}"
            elif Mode == "after":
                result = f"{input_text}{sep}{user_text}"
            else:
                # Подстраховка: список выбора в UI не даст иное значение
                raise RuntimeError(f"[string_concat_directional] Некорректный режим Mode: {Mode}")

            return (result,)

        except RuntimeError:
            # Пробрасываем уже осмысленные ошибки без изменения
            raise
        except Exception as e:
            msg = f"[string_concat_directional] Неожиданная ошибка: {str(e)}"
            print(msg)
            raise RuntimeError(msg) from e


# Регистрация ноды в ComfyUI
NODE_CLASS_MAPPINGS = {"StringConcatDirectional": StringConcatDirectional}
NODE_DISPLAY_NAME_MAPPINGS = {"StringConcatDirectional": "🧩 string_concat_directional"}
