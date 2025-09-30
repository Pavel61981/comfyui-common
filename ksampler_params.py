# filename: ksampler_lora_params.py

"""
Нода "KSampler + LoRA Params"
- Параметры для KSampler (Seed, Steps, CFG, Sampler, Scheduler, Denoise) + раздельные силы LoRA.
- Генерация строкового конфига по шаблону (Format).
- Типы Sampler/Scheduler — нативные списки из comfy.samplers (без прокси).
- Дефолты вычисляются динамически из актуальных списков при каждом вызове INPUT_TYPES,
  чтобы избежать ошибок вида "Value not in list".
- В execute мягкая проверка имён (предупреждение, а не исключение).

Формат чисел в строке:
- INT без точки; FLOAT — фиксировано 2 знака (напр. 1.00).
"""

from __future__ import annotations
import re
from typing import Any, Optional


# ---- Утилиты доступа к comfy.samplers ----
def _get_samplers_list():
    import comfy.samplers as cs  # type: ignore

    return cs.KSampler.SAMPLERS


def _get_schedulers_list():
    import comfy.samplers as cs  # type: ignore

    return cs.KSampler.SCHEDULERS


# --- Вспомогательные функции для имён/дефолтов/форматирования ---
def _extract_name(item: Any) -> str:
    """Возвращает человекочитаемое имя опции из строки/кортежа/словаря."""
    if isinstance(item, (list, tuple)) and item:
        return str(item[0])
    if isinstance(item, dict):
        return str(item.get("name") or item.get("label") or item.get("id") or "")
    return str(item)


def _to_name_list(raw) -> list[str]:
    try:
        items = list(raw) if raw is not None else []
    except Exception:
        return []
    return [_extract_name(x) for x in items]


def _find_raw_item_by_name(raw_list, name: str) -> Optional[Any]:
    """Ищет реальный элемент контейнера по имени (возвращает сам элемент контейнера)."""
    try:
        for it in raw_list:
            if _extract_name(it) == name:
                return it
    except Exception:
        pass
    return None


def _choose_default(raw_list, preferred_name: str) -> Any:
    """
    Возвращает дефолт, который ТОЧНО содержится в raw_list.
    Сначала пытается preferred_name, иначе — первый элемент.
    """
    try:
        items = list(raw_list)
    except Exception:
        items = []
    if not items:
        # Пустой список — вернём просто preferred_name (UI всё равно не даст выбрать).
        return preferred_name
    hit = _find_raw_item_by_name(items, preferred_name)
    return hit if hit is not None else items[0]


_ALLOWED_KEYS = {"steps", "cfg", "sam", "sch", "den", "lora1", "lora2"}
_PLACEHOLDER_RE = re.compile(r"\{([a-z0-9_]+)\}")


def _fmt_int(v: int) -> str:
    return str(int(v))


def _fmt_float(v: float) -> str:
    try:
        return f"{float(v):.2f}"
    except Exception:
        return str(v)


def _build_config_string(format_pattern: str, values: dict) -> str:
    if not format_pattern:
        return ""
    keys_in_tpl = set(m.group(1) for m in _PLACEHOLDER_RE.finditer(format_pattern))
    unknown = keys_in_tpl - _ALLOWED_KEYS
    if unknown:
        allowed = ", ".join(sorted(_ALLOWED_KEYS))
        raise RuntimeError(
            f"[ksampler_params] Неизвестные плейсхолдеры в Format: {', '.join(sorted(unknown))}. "
            f"Допустимые: {allowed}"
        )

    def repl(m):
        key = m.group(1)
        if key not in values:
            return ""
        return f"{key}-{values[key]}"

    return _PLACEHOLDER_RE.sub(repl, format_pattern)


# ---- Основной класс ноды ----
class KSamplerParams:
    """
    Выходы:
        Seed (INT)
        Steps (INT)
        CFG (FLOAT)
        Sampler (enum comfy.samplers.KSampler.SAMPLERS)
        Scheduler (enum comfy.samplers.KSampler.SCHEDULERS)
        Denoise (FLOAT)
        LoRA_Strength_Model (FLOAT)
        LoRA_Strength_CLIP (FLOAT)
        ConfigString (STRING)
    """

    @classmethod
    def INPUT_TYPES(cls):
        import comfy.samplers as cs

        # Используем ТОЧНО ТЕ ЖЕ объекты, что и KSampler
        samplers_obj = cs.KSampler.SAMPLERS
        schedulers_obj = cs.KSampler.SCHEDULERS

        # Дефолты выбираем так, чтобы они ТOЧНО были "в списке"
        sampler_default = _choose_default(samplers_obj, "res_multistep")
        scheduler_default = _choose_default(schedulers_obj, "karras")

        return {
            "required": {
                "Seed": ("INT", {"default": 0, "min": 0, "max": 2147483647, "step": 1}),
                "Steps": ("INT", {"default": 10, "min": 1, "max": 4096, "step": 1}),
                "CFG": (
                    "FLOAT",
                    {"default": 1.0, "min": 1.0, "max": 30.0, "step": 0.1},
                ),
                "Sampler": (samplers_obj, {"default": sampler_default}),
                "Scheduler": (schedulers_obj, {"default": scheduler_default}),
                "Denoise": (
                    "FLOAT",
                    {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01},
                ),
                "LoRA_Strength_Model": (
                    "FLOAT",
                    {"default": 0.5, "min": 0.0, "max": 2.0, "step": 0.05},
                ),
                "LoRA_Strength_CLIP": (
                    "FLOAT",
                    {"default": 0.5, "min": 0.0, "max": 2.0, "step": 0.05},
                ),
                "Format": (
                    "STRING",
                    {"default": "{lora1}__{sam}__{sch}__{cfg}__{steps}"},
                ),
            }
        }

    # Используем ТОЧНО ТЕ ЖЕ объекты, что и KSampler
    import comfy.samplers as cs

    _SAMPLERS_TYPE = cs.KSampler.SAMPLERS
    _SCHEDULERS_TYPE = cs.KSampler.SCHEDULERS

    RETURN_TYPES = (
        "INT",
        "INT",
        "FLOAT",
        _SAMPLERS_TYPE,
        _SCHEDULERS_TYPE,
        "FLOAT",
        "FLOAT",
        "FLOAT",
        "STRING",
    )
    RETURN_NAMES = (
        "Seed",
        "Steps",
        "CFG",
        "Sampler",
        "Scheduler",
        "Denoise",
        "LoRA_Strength_Model",
        "LoRA_Strength_CLIP",
        "ConfigString",
    )
    FUNCTION = "execute"
    CATEGORY = "utils/ksampler"
    OUTPUT_NODE = False

    def execute(
        self,
        Seed: int,
        Steps: int,
        CFG: float,
        Sampler: Any,
        Scheduler: Any,
        Denoise: float,
        LoRA_Strength_Model: float,
        LoRA_Strength_CLIP: float,
        Format: str,
    ):
        """
        Валидирует числа, мягко проверяет имена (без падения) и собирает ConfigString.
        """
        try:
            # Жёсткие числовые диапазоны
            if not (0 <= int(Seed) <= 2147483647):
                raise ValueError("Seed должен быть в диапазоне [0..2147483647].")
            if not (1 <= int(Steps) <= 4096):
                raise ValueError("Steps должен быть в диапазоне [1..4096].")
            if not (1.0 <= float(CFG) <= 30.0):
                raise ValueError("CFG должен быть в диапазоне [1.0..30.0].")
            if not (0.0 <= float(Denoise) <= 1.0):
                raise ValueError("Denoise должен быть в диапазоне [0.0..1.0].")
            if not (0.0 <= float(LoRA_Strength_Model) <= 2.0):
                raise ValueError(
                    "LoRA_Strength_Model должен быть в диапазоне [0.0..2.0]."
                )
            if not (0.0 <= float(LoRA_Strength_CLIP) <= 2.0):
                raise ValueError(
                    "LoRA_Strength_CLIP должен быть в диапазоне [0.0..2.0]."
                )

            # Мягкая проверка имён (если вообще удастся прочитать текущие списки)
            try:
                current_sampler_names = _to_name_list(self._SAMPLERS_TYPE)
                current_scheduler_names = _to_name_list(self._SCHEDULERS_TYPE)
            except Exception:
                current_sampler_names, current_scheduler_names = [], []

            sampler_name = _extract_name(Sampler)
            scheduler_name = _extract_name(Scheduler)

            if current_sampler_names and sampler_name not in current_sampler_names:
                print(
                    f"[ksampler_params][warn] Sampler '{sampler_name}' сейчас не найден в списке: "
                    f"{', '.join(current_sampler_names)}. Передаю значение как есть."
                )
            if (
                current_scheduler_names
                and scheduler_name not in current_scheduler_names
            ):
                print(
                    f"[ksampler_params][warn] Scheduler '{scheduler_name}' сейчас не найден в списке: "
                    f"{', '.join(current_scheduler_names)}. Передаю значение как есть."
                )

            # Строка-конфиг
            values = {
                "steps": _fmt_int(Steps),
                "cfg": _fmt_float(CFG),
                "sam": sampler_name,
                "sch": scheduler_name,
                "den": _fmt_float(Denoise),
                "lora1": _fmt_float(LoRA_Strength_Model),
                "lora2": _fmt_float(LoRA_Strength_CLIP),
            }
            config_str = _build_config_string(str(Format), values)

            # Возвращаем оригинальные объекты Sampler и Scheduler (не строки!)
            return (
                int(Seed),
                int(Steps),
                float(CFG),
                Sampler,  # ← оригинальный объект из enum
                Scheduler,  # ← оригинальный объект из enum
                float(Denoise),
                float(LoRA_Strength_Model),
                float(LoRA_Strength_CLIP),
                config_str,
            )

        except Exception as e:
            msg = f"[ksampler_params] {e}"
            print(msg)
            raise RuntimeError(msg) from e


# Регистрация ноды
NODE_CLASS_MAPPINGS = {"KSamplerParams": KSamplerParams}
NODE_DISPLAY_NAME_MAPPINGS = {"KSamplerParams": "🧩 KSampler + LoRA Params"}
