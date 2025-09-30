# filename: image_tile_planner.py
import math


class ImageIntTilePlanner:
    """
    Нода 'Image Tile Planner' — рассчитывает:
      - размер тайла (Tile_Size), кратный 64,
      - шаг (Tile_Stride) = выбранная пропорция от Tile_Size ("3 / 4" или "1 / 2"),
        округлён вниз до кратности 64, минимум 64,
      - флаг Use_Tiles = (max(H, W) > Start_Tiling_At),
      - количество тайлов по ширине/высоте (Tiles_X/Tiles_Y) в режиме 'cover':
        последний тайл сдвигается к границе при необходимости, чтобы покрыть всё изображение.

    Первый выход — исходное изображение (весь батч) без изменений.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "Image": ("IMAGE",),
                "Max_Tile_Size": ("INT", {"default": 1024, "min": 64, "step": 64}),
                "Start_Tiling_At": ("INT", {"default": 2048, "min": 64, "step": 64}),
                "Stride_Ratio": (["3 / 4", "1 / 2"],),
            }
        }

    RETURN_TYPES = ("IMAGE", "INT", "INT", "BOOLEAN", "INT", "INT")
    RETURN_NAMES = (
        "Image",
        "Tile_Size",
        "Tile_Stride",
        "Use_Tiles",
        "Tiles_X",
        "Tiles_Y",
    )
    FUNCTION = "execute"
    CATEGORY = "utils/tiling"
    OUTPUT_NODE = False

    @staticmethod
    def _round_down_multiple(value: int, base: int) -> int:
        """Округляет вниз до ближайшего кратного base."""
        return (value // base) * base

    @staticmethod
    def _validate_and_fix_params(max_tile_size: int, start_tiling_at: int):
        """Корректирует параметры до нижней границы 64 и логирует предупреждения."""
        if max_tile_size < 64:
            print("[Image Tile Planner] Warning: Max_Tile_Size < 64; corrected to 64.")
            max_tile_size = 64
        if start_tiling_at < 64:
            print(
                "[Image Tile Planner] Warning: Start_Tiling_At < 64; corrected to 64."
            )
            start_tiling_at = 64
        return max_tile_size, start_tiling_at

    @staticmethod
    def _parse_stride_ratio(ratio_choice: str):
        """
        Преобразует строку вида '3 / 4' или '1 / 2' в числитель и знаменатель.
        Возвращает (3,4) или (1,2). По умолчанию — (3,4) с предупреждением.
        """
        if not isinstance(ratio_choice, str):
            print(
                "[Image Tile Planner] Warning: Stride_Ratio is not a string; defaulting to '3 / 4'."
            )
            return 3, 4
        normalized = ratio_choice.replace(" ", "")
        mapping = {
            "3/4": (3, 4),
            "1/2": (1, 2),
        }
        if normalized in mapping:
            return mapping[normalized]
        print(
            f"[Image Tile Planner] Warning: Unsupported Stride_Ratio '{ratio_choice}'; defaulting to '3 / 4'."
        )
        return 3, 4

    @staticmethod
    def _tiles_cover_count(dim: int, tile: int, stride: int) -> int:
        """
        Количество тайлов для полного покрытия измерения dim тайлом размера tile
        при базовом шаге stride (последний шаг может быть меньше stride).
        Формула: max(1, ceil((dim - tile) / stride) + 1), при условии tile <= dim и stride >= 1.
        """
        remainder = dim - tile
        if remainder <= 0:
            return 1
        return max(1, math.ceil(remainder / stride) + 1)

    def execute(
        self, Image, Max_Tile_Size: int, Start_Tiling_At: int, Stride_Ratio: str
    ):
        """
        Возвращает:
        (Image, Tile_Size, Tile_Stride, Use_Tiles, Tiles_X, Tiles_Y)
        """
        try:
            # Параметры
            Max_Tile_Size, Start_Tiling_At = self._validate_and_fix_params(
                int(Max_Tile_Size), int(Start_Tiling_At)
            )

            # Проверки изображения
            if Image is None:
                raise RuntimeError("[Image Tile Planner] Image is None.")
            if not hasattr(Image, "shape"):
                raise RuntimeError(
                    "[Image Tile Planner] Unsupported image type: no .shape attribute."
                )

            shape = Image.shape  # ожидается [B, H, W, C]
            if len(shape) != 4:
                raise RuntimeError(
                    f"[Image Tile Planner] Expected image shape [B,H,W,C], got {shape}."
                )

            _, H, W, C = shape
            if C not in (3, 4):
                print(
                    f"[Image Tile Planner] Warning: channel count is {C}, expected 3 or 4."
                )

            if H < 64 or W < 64:
                raise RuntimeError(
                    f"[Image Tile Planner] Image is too small for tiling rules: H={H}, W={W} (need >=64)."
                )

            # Размеры
            min_dim = H if H < W else W
            max_dim = H if H > W else W

            # Размер тайла
            candidate = min(min_dim, Max_Tile_Size)
            tile_size = self._round_down_multiple(int(candidate), 64)
            if tile_size < 64:
                raise RuntimeError(
                    f"[Image Tile Planner] Computed tile size < 64 (got {tile_size}). "
                    f"Check inputs: min_dim={min_dim}, Max_Tile_Size={Max_Tile_Size}."
                )

            # Пропорция шага
            num, den = self._parse_stride_ratio(Stride_Ratio)

            # Шаг тайла (stride)
            stride_base = int(tile_size * num / den)
            stride = self._round_down_multiple(stride_base, 64)
            if stride < 64:
                stride = 64

            # Флаг использования тайлов
            use_tiles = bool(max_dim > Start_Tiling_At)

            # Подсчёт количества тайлов (режим cover, независимо от use_tiles)
            tiles_x = self._tiles_cover_count(int(W), int(tile_size), int(stride))
            tiles_y = self._tiles_cover_count(int(H), int(tile_size), int(stride))

            # Возврат исходного изображения и метрик
            return (
                Image,
                int(tile_size),
                int(stride),
                bool(use_tiles),
                int(tiles_x),
                int(tiles_y),
            )

        except RuntimeError:
            raise
        except Exception as e:
            msg = f"[Image Tile Planner] Unexpected error: {type(e).__name__}: {str(e)}"
            print(msg)
            raise RuntimeError(msg) from e


# Регистрация ноды в ComfyUI
NODE_CLASS_MAPPINGS = {"ImageIntTilePlanner": ImageIntTilePlanner}
NODE_DISPLAY_NAME_MAPPINGS = {"ImageIntTilePlanner": "🧩 Image Tile Planner"}
