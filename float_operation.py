import math


class FloatOperation:
    """
    Нода "Float Operation" — выполняет различные операции над двумя числами с плавающей точкой.
    Поддерживает арифметические, сравнительные, статистические и дополнительные математические операции.
    Обрабатывает ошибки (деление на ноль, некорректные степени и т.д.).

    Примечания:
    - Операция "//" возвращает результат целочисленного деления, приведённый к float (с отброшенной дробной частью).
    - Операция "average" возвращает точное арифметическое среднее без округления.
    - Операция "pow" не поддерживает отрицательные основания с нецелыми показателями (вернёт ошибку).
    - Операция "sum" является алиасом для "+".
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "Input_A": (  # ← Исправлено имя
                    "FLOAT",
                    {"default": 0.0, "step": 0.01, "tooltip": "Первое число"},
                ),
                "Input_B": (  # ← Исправлено имя
                    "FLOAT",
                    {"default": 0.0, "step": 0.01, "tooltip": "Второе число"},
                ),
                "Operation": (
                    [
                        "+",
                        "-",
                        "*",
                        "/",
                        "%",
                        "//",
                        "max",
                        "min",
                        "sum",
                        "average",
                        "pow",
                        "hypot",
                    ],
                    {"default": "+", "tooltip": "Выберите операцию для выполнения"},
                ),
            }
        }

    RETURN_TYPES = ("FLOAT",)
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
                if Input_B == 0.0:
                    raise ZeroDivisionError("Division by zero in '/' operation")
                result = Input_A / Input_B
            elif Operation == "%":
                if Input_B == 0.0:
                    raise ZeroDivisionError("Division by zero in '%' operation")
                result = Input_A % Input_B
            elif Operation == "//":
                if Input_B == 0.0:
                    raise ZeroDivisionError("Division by zero in '//' operation")
                result = float(math.trunc(Input_A / Input_B))  # ← Упрощено и исправлено
            elif Operation == "max":
                result = max(Input_A, Input_B)
            elif Operation == "min":
                result = min(Input_A, Input_B)
            elif Operation == "sum":
                result = Input_A + Input_B  # Алиас для сложения
            elif Operation == "average":
                result = (Input_A + Input_B) / 2.0
            elif Operation == "pow":
                if Input_A == 0.0 and Input_B < 0:
                    raise ValueError("0 cannot be raised to a negative power")
                if Input_A < 0 and not Input_B.is_integer():
                    raise ValueError(
                        "Negative base with fractional exponent is not real"
                    )
                result = Input_A**Input_B
                if not math.isfinite(result):
                    raise OverflowError("Result too large (overflow)")
            elif Operation == "hypot":
                result = math.hypot(Input_A, Input_B)
            else:
                raise ValueError(f"Unknown operation: {Operation}")

            # Финальная проверка на валидность результата
            if not math.isfinite(result):
                raise ValueError("Result is NaN or Inf")

            return (result,)

        except (ZeroDivisionError, ValueError, OverflowError) as e:
            error_msg = f"[Float Operation] Error: {str(e)}"
            print(error_msg)
            raise RuntimeError(error_msg) from e
        except Exception as e:
            error_msg = f"[Float Operation] Unexpected error: {str(e)}"
            print(error_msg)
            raise RuntimeError(error_msg) from e


# Регистрация
NODE_CLASS_MAPPINGS = {"FloatOperation": FloatOperation}
NODE_DISPLAY_NAME_MAPPINGS = {"FloatOperation": "🧮 Float Operation"}
