import torch.nn as nn
from torchvision import transforms


class UpscaleDefense:
    def __init__(self, mode='bilinear', upscale_factor=0.5):
        self.mode = mode
        self.upscale_factor = upscale_factor

    def __call__(self, image):
        new_size = [int(image.shape[i] * self.upscale_factor) for i in range(2, 4)]
        resized_image = transforms.Resize(list(new_size))(image)
        resized_image = nn.Upsample(size=(299, 299), mode=self.mode)(resized_image)
        return resized_image
