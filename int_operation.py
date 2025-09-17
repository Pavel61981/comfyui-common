class IntOperation:
    """
    Нода "Int Operation" — выполняет различные операции над двумя целыми числами.
    Поддерживает арифметические, сравнительные и статистические операции.
    Обрабатывает ошибки (например, деление на ноль).
    Примечание: операция "average" использует целочисленное деление (округление вниз).
    Операция "sum" является алиасом для "+".
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "Input_A": ("INT", {"default": 0, "step": 1}),  # ← Исправлено имя
                "Input_B": ("INT", {"default": 0, "step": 1}),  # ← Исправлено имя
                "Operation": (
                    ["+", "-", "*", "/", "%", "//", "max", "min", "sum", "average"],
                    {"default": "+"},
                ),
            }
        }

    RETURN_TYPES = ("INT",)
    FUNCTION = "execute"
    CATEGORY = "utils/math"
    OUTPUT_NODE = False

    def execute(self, Input_A, Input_B, Operation):
        try:
            if Operation == "+":
                result = Input_A + Input_B
            elif Operation == "-":
                result = Input_A - Input_B
            elif Operation == "*":
                result = Input_A * Input_B
            elif Operation == "/":
                if Input_B == 0:
                    raise ZeroDivisionError("Division by zero in '/' operation")
                result = int(Input_A / Input_B)
            elif Operation == "%":
                if Input_B == 0:
                    raise ZeroDivisionError("Division by zero in '%' operation")
                result = Input_A % Input_B
            elif Operation == "//":
                if Input_B == 0:
                    raise ZeroDivisionError("Division by zero in '//' operation")
                result = Input_A // Input_B
            elif Operation == "max":
                result = max(Input_A, Input_B)
            elif Operation == "min":
                result = min(Input_A, Input_B)
            elif Operation == "sum":
                result = Input_A + Input_B  # Алиас для сложения
            elif Operation == "average":
                result = (Input_A + Input_B) // 2  # Целочисленное среднее
            else:
                raise ValueError(f"Unknown operation: {Operation}")

            return (result,)

        except ZeroDivisionError as e:
            error_msg = f"[Int Operation] Error: {str(e)}"
            print(error_msg)
            raise RuntimeError(error_msg) from e  # ← Сохраняем цепочку исключений
        except Exception as e:
            error_msg = f"[Int Operation] Unexpected error: {str(e)}"
            print(error_msg)
            raise RuntimeError(error_msg) from e


# Регистрация ноды в ComfyUI
NODE_CLASS_MAPPINGS = {"IntOperation": IntOperation}

NODE_DISPLAY_NAME_MAPPINGS = {"IntOperation": "🔢 Int Operation"}
