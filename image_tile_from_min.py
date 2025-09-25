# class ImageTileFromMin:
#     """
#     Нода "image_tile_from_min" — вычисляет оптимальный размер плитки на основе
#     минимальной стороны изображения (H/W) с учётом верхнего предела Max_Tile.
#     Размер плитки всегда кратен 64 и не превышает минимальную сторону.
#     Поддерживает батчи: для батча возвращает список int по каждому изображению.
#     """

#     @classmethod
#     def INPUT_TYPES(cls):
#         return {
#             "required": {
#                 "Image": ("IMAGE",),
#                 "Max_Tile": ("INT", {"default": 1024, "step": 64}),
#             }
#         }

#     RETURN_TYPES = ("INT",)
#     FUNCTION = "execute"
#     CATEGORY = "utils/image"
#     OUTPUT_NODE = False

#     def _validate_max_tile(self, max_tile: int) -> None:
#         if max_tile <= 0:
#             raise RuntimeError(
#                 "[image_tile_from_min] Max_Tile должен быть положительным и кратным 64."
#             )
#         if max_tile % 64 != 0:
#             raise RuntimeError(
#                 "[image_tile_from_min] Max_Tile должен быть кратен 64. "
#                 f"Получено: {max_tile}"
#             )

#     @staticmethod
#     def _shape_of_image(image):
#         """
#         Возвращает кортеж (batch, height, width, channels) из входного IMAGE.
#         Не выполняет копий данных.
#         """
#         # ComfyUI обычно передает torch.Tensor: [B, H, W, C]
#         # Но аккуратно работаем только со shape, без импорта torch.
#         try:
#             shape = tuple(image.shape)
#         except Exception as e:
#             raise RuntimeError(
#                 f"[image_tile_from_min] Не удалось получить shape изображения: {e}"
#             ) from e

#         if len(shape) != 4:
#             raise RuntimeError(
#                 f"[image_tile_from_min] Ожидалась размерность [B,H,W,C], получено: {shape}"
#             )

#         b, h, w, c = shape
#         if not (isinstance(b, int) and isinstance(h, int) and isinstance(w, int) and isinstance(c, int)):
#             # Бывают тензоры с torch.Size, приведем к int через map
#             try:
#                 b, h, w, c = map(int, (b, h, w, c))
#             except Exception:
#                 raise RuntimeError(
#                     f"[image_tile_from_min] Невалидная размерность изображения: {shape}"
#                 )

#         if h <= 0 or w <= 0:
#             raise RuntimeError(
#                 f"[image_tile_from_min] Некорректные размеры изображения: H={h}, W={w}"
#             )

#         return b, h, w, c

#     @staticmethod
#     def _compute_tile_for_min_side(min_side: int, max_tile: int) -> int:
#         """
#         Возвращает оптимальный размер плитки для одной картинки.
#         Бросает RuntimeError, если min_side < 64.
#         """
#         if min_side < 64:
#             raise RuntimeError(
#                 f"[image_tile_from_min] Минимальная сторона изображения {min_side} < 64."
#             )

#         limit = min(min_side, max_tile)
#         # кратно 64, не превышает limit
#         tile = (limit // 64) * 64

#         # На случай, если limit в (64..63)? Практически невозможно из-за проверки выше,
#         # но оставим защиту.
#         if tile < 64:
#             raise RuntimeError(
#                 f"[image_tile_from_min] Не удалось подобрать кратный 64 размер плитки "
#                 f"для минимальной стороны {min_side} и Max_Tile {max_tile}."
#             )

#         return int(tile)

#     def execute(self, Image, Max_Tile):
#         """
#         Выполняет расчет размера плитки.
#         - Для одиночного изображения возвращает int.
#         - Для батча возвращает список int (по одному на элемент батча).
#         """
#         try:
#             self._validate_max_tile(Max_Tile)
#             b, h, w, _ = self._shape_of_image(Image)

#             # Если батч из B изображений, у них одинаковые H/W в большинстве пайплайнов ComfyUI.
#             # Но корректнее посчитать по каждому кадру (на случай вариаций).
#             # Изображение приходит как тензор [B,H,W,C]; H/W одинаковые для всех,
#             # поэтому рассчет по одному значению min_side корректен для всего батча.
#             min_side = min(h, w)

#             # Вычисляем одно значение и, если батч, размножаем.
#             tile_value = self._compute_tile_for_min_side(min_side, Max_Tile)

#             if b == 1:
#                 return (tile_value,)
#             else:
#                 # Стандартная практика: вернуть список скаляров для батча.
#                 return ([tile_value] * b,)

#         except RuntimeError:
#             # Уже информативное сообщение
#             raise
#         except Exception as e:
#             msg = f"[image_tile_from_min] Неожиданная ошибка: {str(e)}"
#             print(msg)
#             raise RuntimeError(msg) from e


# # Регистрация ноды в ComfyUI
# NODE_CLASS_MAPPINGS = {"ImageTileFromMin": ImageTileFromMin}
# NODE_DISPLAY_NAME_MAPPINGS = {"ImageTileFromMin": "🧩 image_tile_from_min"}
