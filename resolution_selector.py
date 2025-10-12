# File: resolution_selector.py

class ResolutionSelector:
    """
    Нода "Resolution Selector" — выбор разрешения из фиксированного списка с опцией
    принудительного горизонтального вида (swap ширины и высоты).
    Возвращает (Width, Height, HxW).

    Важно:
    - Preset задаётся как строка "ratio — W×H".
    - При Horizontal_View=True значения меняются местами (квадрат не меняется).
    - Строка HxW всегда формируется как "HEIGHTxWIDTH".
    """

    # Полная карта пресетов: "label" -> (W, H)
    PRESET_MAP = {
        # 9:16
        "9:16 — 144×256": (144, 256),
        "9:16 — 288×512": (288, 512),
        "9:16 — 432×768": (432, 768),
        "9:16 — 540×960": (540, 960),
        "9:16 — 720×1280": (720, 1280),
        "9:16 — 864×1536": (864, 1536),
        "9:16 — 1008×1792": (1008, 1792),
        "9:16 — 1152×2048": (1152, 2048),
        "9:16 — 1440×2560": (1440, 2560),
        "9:16 — 2160×3840": (2160, 3840),

        # 3:4
        "3:4 — 480×640": (480, 640),
        "3:4 — 576×768": (576, 768),
        "3:4 — 768×1024": (768, 1024),
        "3:4 — 960×1280": (960, 1280),
        "3:4 — 1056×1408": (1056, 1408),
        "3:4 — 1152×1536": (1152, 1536),
        "3:4 — 1200×1600": (1200, 1600),
        "3:4 — 1440×1920": (1440, 1920),
        "3:4 — 1536×2048": (1536, 2048),

        # 1:1
        "1:1 — 256×256": (256, 256),
        "1:1 — 384×384": (384, 384),
        "1:1 — 512×512": (512, 512),
        "1:1 — 640×640": (640, 640),
        "1:1 — 768×768": (768, 768),
        "1:1 — 896×896": (896, 896),
        "1:1 — 1024×1024": (1024, 1024),
        "1:1 — 1280×1280": (1280, 1280),
        "1:1 — 1536×1536": (1536, 1536),
        "1:1 — 2048×2048": (2048, 2048),

        # 4:5
        "4:5 — 640×800": (640, 800),
        "4:5 — 768×960": (768, 960),
        "4:5 — 896×1120": (896, 1120),
        "4:5 — 1024×1280": (1024, 1280),
        "4:5 — 1152×1440": (1152, 1440),
        "4:5 — 1280×1600": (1280, 1600),
        "4:5 — 1408×1760": (1408, 1760),
        "4:5 — 1536×1920": (1536, 1920),

        # 2:3
        "2:3 — 512×768": (512, 768),
        "2:3 — 640×960": (640, 960),
        "2:3 — 800×1200": (800, 1200),
        "2:3 — 1024×1536": (1024, 1536),
        "2:3 — 1280×1920": (1280, 1920),
        "2:3 — 1600×2400": (1600, 2400),
    }

    @classmethod
    def INPUT_TYPES(cls):
        options = list(cls.PRESET_MAP.keys())
        return {
            "required": {
                "Preset": (options, {"default": "3:4 — 576×768"}),
                "Horizontal_View": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES = ("INT", "INT", "STRING")
    RETURN_NAMES = ("Width", "Height", "HxW")
    FUNCTION = "execute"
    CATEGORY = "utils/image"
    OUTPUT_NODE = False

    def execute(self, Preset, Horizontal_View):
        """
        Выполняет выбор разрешения и ориентации.
        Возвращает (Width, Height, HxW).

        :param Preset: строка из списка пресетов
        :param Horizontal_View: bool, если True — поменять местами W и H
        """
        try:
            if Preset not in self.PRESET_MAP:
                raise ValueError(f"Unknown preset: '{Preset}'")

            w, h = self.PRESET_MAP[Preset]

            if Horizontal_View:
                width, height = h, w  # swap
            else:
                width, height = w, h

            hxw = f"{height}x{width}"
            return (int(width), int(height), hxw)

        except Exception as e:
            msg = f"[Resolution Selector] Error: {str(e)}"
            print(msg)
            raise RuntimeError(msg) from e


# Регистрация ноды в ComfyUI
NODE_CLASS_MAPPINGS = {"ResolutionSelector": ResolutionSelector}
NODE_DISPLAY_NAME_MAPPINGS = {"ResolutionSelector": "🧩 Resolution Selector"}
