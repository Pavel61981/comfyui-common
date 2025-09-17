import torch
import torch.nn.functional as F


class ImagesPack:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "left_image": ("IMAGE",),
                "right_image": ("IMAGE",),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "concat_images"
    CATEGORY = "image/postprocessing"

    def resize_image_proportionally(self, image, target_height):
        """Пропорциональное изменение размера изображения"""
        original_height = image.shape[1]
        original_width = image.shape[2]

        # Вычисляем коэффициент масштабирования
        scale = target_height / original_height
        new_width = int(original_width * scale)

        # Изменяем размер с сохранением пропорций
        resized = F.interpolate(
            image.permute(0, 3, 1, 2),
            size=(target_height, new_width),
            mode="bicubic",
            align_corners=False,
        ).permute(0, 2, 3, 1)

        return resized

    def concat_images(self, left_image, right_image):
        # Проверяем совпадение размеров по высоте
        if left_image.shape[1] != right_image.shape[1]:
            # Пропорционально изменяем размер изображения с меньшей высотой
            if left_image.shape[1] < right_image.shape[1]:
                left_image = self.resize_image_proportionally(
                    left_image, right_image.shape[1]
                )
            else:
                right_image = self.resize_image_proportionally(
                    right_image, left_image.shape[1]
                )

        # Склеиваем изображения по ширине (dim=2)
        concatenated = torch.cat([left_image, right_image], dim=2)
        return (concatenated,)


# Регистрация ноды
NODE_CLASS_MAPPINGS = {"ImagesPack": ImagesPack}

NODE_DISPLAY_NAME_MAPPINGS = {"ImagesPack": "Images pack"}
