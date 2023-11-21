from glob import glob
import os
from pathlib import Path

import cv2
from torch.utils.data import Dataset
import torchvision.transforms as transforms


all_attacks = [
    'adv-cf',
    'zhang-et-al-dists',
    'zhang-et-al-lpips',
    'zhang-et-al-ssim',
    'amifgsm',
    'ifgsm',
    'mifgsm',
    'korhonen-et-al',
    'madc',
    'ssah',
]


class PurificationDataset(Dataset):
    def __init__(
        self,
        attacked_dir: str,
        source_dir: str,
        mode: str = 'train',
        attacks=['adv-cf'],
    ):
        self.attacked_dir = attacked_dir
        self.source_dir = Path(source_dir)
        self.transform = transforms.ToTensor()

        self.images = sorted(self.source_dir.glob('*'), key=lambda item: item.name)

        self.images = []
        for attack in attacks:
            images = sorted(glob(os.path.join(self.attacked_dir, attack, '*')))
            if mode == 'train':
                self.images.extend(images[: int(0.8 * len(images))])
            else:
                self.images.extend(images[int(0.8 * len(images)) :])
        print(len(self.images))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image_name = os.path.basename(self.images[index])

        source_img = cv2.imread(str(self.source_dir / image_name))
        source_img = cv2.cvtColor(source_img, cv2.COLOR_BGR2RGB)
        source_img = self.transform(source_img)

        attacked_img = cv2.imread(self.images[index])
        attacked_img = cv2.cvtColor(attacked_img, cv2.COLOR_BGR2RGB)
        attacked_img = self.transform(attacked_img)
        return attacked_img, source_img
