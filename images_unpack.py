import torch


class ImagesUnpack:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
            }
        }

    RETURN_TYPES = ("IMAGE", "IMAGE")
    RETURN_NAMES = ("left_image", "right_image")
    FUNCTION = "split_image"
    CATEGORY = "image/postprocessing"

    def split_image(self, image):
        # Получаем ширину изображения
        width = image.shape[2]

        # Вычисляем середину изображения
        mid_point = width // 2

        # Разделяем изображение на две части
        left_image = image[:, :, :mid_point, :]
        right_image = image[:, :, mid_point:, :]

        return (left_image, right_image)


# Регистрация ноды
NODE_CLASS_MAPPINGS = {"ImageSplitHorizontal": ImagesUnpack}

NODE_DISPLAY_NAME_MAPPINGS = {"ImageSplitHorizontal": "Images unpack"}
