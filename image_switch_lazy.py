# switch_lazy_image.py
# Класс: ImageSwitchLazy
# Принимает:
#   use_first - BOOLEAN (не lazy)  -> определяет, какое изображение вернуть
#   image1    - IMAGE   (lazy=True)
#   image2    - IMAGE   (lazy=True)
# Возвращает:
#   одно IMAGE — либо image1 (если use_first==True), либо image2


class ImageSwitchLazy:
    @classmethod
    def INPUT_TYPES(cls):
        # порядок полей здесь определяет порядок аргументов, в котором
        # check_lazy_status и функция switch будут получать значения.
        return {
            "required": {
                # Булево значение не помечаем как lazy — оно должно быть вычислено заранее.
                "use_first": (
                    "BOOLEAN",
                    {"default": True, "label_on": "Первое", "label_off": "Второе"},
                ),
                # Оба изображения помечаем lazy=True
                "image1": ("IMAGE", {"lazy": True}),
                "image2": ("IMAGE", {"lazy": True}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "switch"
    CATEGORY = "Custom"

    # ВАЖНО: это обычный метод (не @classmethod).
    # Аргументы должны идти в том же порядке, что и ключи в INPUT_TYPES.
    def check_lazy_status(self, use_first, image1, image2):
        """
        Движок вызывает эту функцию, если есть хотя бы один lazy вход, который ещё не вычислен.
        Доступные входы передаются как их значения; невычисленные lazy входы = None.
        Возвращаем список имён lazy-входов, которые нужно вычислить дальше.
        """
        needed = []
        # Если пользователь выбрал первое изображение — требуется только image1
        if use_first:
            # Если image1 ещё не вычислен — помечаем его как необходимый
            if image1 is None:
                needed.append("image1")
        else:
            # Иначе требуется только image2
            if image2 is None:
                needed.append("image2")
        return needed

    def switch(self, use_first, image1, image2):
        """
        Основная функция. К моменту вызова сюда уже должны быть вычислены те lazy-входы,
        которые были в списке, возвращённом из check_lazy_status.
        Возвращаем кортеж с одним IMAGE.
        """
        if use_first:
            # Безопасная защита: если image1 всё же None, можно вернуть ExecutionBlocker
            if image1 is None:
                # лениво: импортируем локально, чтобы не ломать, если модуль недоступен
                from comfy_execution.graph import ExecutionBlocker

                return (ExecutionBlocker("image1 не был вычислен"),)
            return (image1,)
        else:
            if image2 is None:
                from comfy_execution.graph import ExecutionBlocker

                return (ExecutionBlocker("image2 не был вычислен"),)
            return (image2,)


# === Регистрация ноды ===
NODE_CLASS_MAPPINGS = {
    "ImageSwitchLazy": ImageSwitchLazy,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ImageSwitchLazy": "🔀 Image Switch Lazy",
}
