import torch


class FlipDefense:
    def __init__(self, axes=[2, 3]):
        self.axes = axes

    def __call__(self, image):
        return torch.flip(image, self.axes)
