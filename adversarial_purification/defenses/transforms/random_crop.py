from torchvision import transforms


class RandomCropDefense:
    def __init__(self, size=256):
        self.size = size

    def __call__(self, image):
        return transforms.RandomCrop(self.size)(image)
