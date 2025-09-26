import os
import importlib
from pathlib import Path

# Получаем путь к текущей директории
current_dir = Path(__file__).parent

# Собираем все .py файлы в директории, кроме __init__.py
node_files = [
    f.stem
    for f in current_dir.glob("*.py")
    if f.name != "__init__.py" and not f.name.startswith(".")
]

NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}

# Динамически импортируем каждый файл и объединяем маппинги
for module_name in node_files:
    try:
        # Импортируем модуль
        module = importlib.import_module(f".{module_name}", package=__name__)

        # Проверяем наличие ОБОИХ обязательных атрибутов
        if hasattr(module, "NODE_CLASS_MAPPINGS") and hasattr(
            module, "NODE_DISPLAY_NAME_MAPPINGS"
        ):
            # Обновляем общие маппинги
            NODE_CLASS_MAPPINGS.update(module.NODE_CLASS_MAPPINGS)
            NODE_DISPLAY_NAME_MAPPINGS.update(module.NODE_DISPLAY_NAME_MAPPINGS)

            # Логируем только негативные случаи
            loaded_display_names = [
                NODE_DISPLAY_NAME_MAPPINGS.get(key, key)
                for key in module.NODE_CLASS_MAPPINGS.keys()
            ]
            if not loaded_display_names:
                print(f"⚠️  Модуль '{module_name}' не содержит нодов (пустые маппинги)")
        else:
            print(
                f"⚠️  {module_name} не содержит NODE_CLASS_MAPPINGS или NODE_DISPLAY_NAME_MAPPINGS — пропущен"
            )

    except Exception as e:
        print(f"❌ Ошибка при импорте {module_name}: {e}")

# Экспортируем для ComfyUI
__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
