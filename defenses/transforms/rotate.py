import albumentations as A
import torch
from torchvision import transforms


class RotateDefense:
    def __init__(self, angle_limit):
        self.angle_limit = angle_limit

    def __call__(self, image):
        # image = image.squeeze(0).cpu().numpy()
        # transform = A.Compose([A.Rotate(limit=self.angle_limit, p=1)])
        # image = transform(image=image)['image']
        # image = torch.from_numpy(image).cuda()
        # image = image.unsqueeze(0)
        return transforms.RandomRotation(self.angle_limit)(image)
