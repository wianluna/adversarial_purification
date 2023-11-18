import albumentations as A
import torch


class MedianFilterDefense:
    def __init__(self, blur_limit=3):
        self.blur_limit = blur_limit

    def __call__(self, image):
        image = image.squeeze(0).cpu().permute(1, 2, 0).numpy()
        transform = A.Compose([A.MedianBlur(blur_limit=self.blur_limit, p=1)])
        image = transform(image=image)['image']
        image = torch.from_numpy(image).cuda()
        image = image.unsqueeze(0).permute(0, 3, 1, 2)
        return image
