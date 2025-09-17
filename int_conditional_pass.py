class IntConditionalPass:
    """
    Нода "Int Conditional Pass": передаёт input_int, если condition=True,
    иначе — fallback_int.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "input_int": ("INT", {"default": 0, "step": 1, "display": "number"}),
                "condition": ("BOOLEAN", {"default": True}),
            },
            "optional": {
                "fallback_int": ("INT", {"default": 0, "step": 1, "display": "number"}),
            },
        }

    RETURN_TYPES = ("INT",)
    FUNCTION = "execute"
    CATEGORY = "Logic/Int"

    def execute(self, input_int, condition, fallback_int=0):
        """
        Логика: если condition == True → вернуть input_int, иначе → fallback_int
        """
        result = input_int if condition else fallback_int
        return (result,)


# Регистрация ноды в ComfyUI
NODE_CLASS_MAPPINGS = {"IntConditionalPass": IntConditionalPass}

NODE_DISPLAY_NAME_MAPPINGS = {"IntConditionalPass": "Int Conditional Pass"}
