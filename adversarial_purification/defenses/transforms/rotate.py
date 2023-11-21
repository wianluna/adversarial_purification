from torchvision import transforms


class RotateDefense:
    def __init__(self, angle_limit):
        self.angle_limit = angle_limit

    def __call__(self, image):
        return transforms.RandomRotation(self.angle_limit)(image)
