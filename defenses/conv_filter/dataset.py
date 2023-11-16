from pathlib import Path

import cv2
from torch.utils.data import Dataset
import torchvision.transforms as transforms


class PurificationDataset(Dataset):
    def __init__(
        self,
        attacked_dir: str,
        source_dir: str,
        mode: str = 'train',
    ):
        self.attacked_dir = Path(attacked_dir)
        self.source_dir = Path(source_dir)
        self.transform = transforms.ToTensor()

        self.images = sorted(self.source_dir.glob('*'), key=lambda item: item.name)
        if mode == 'train':
            self.images = self.images[: int(0.8 * len(self.images))]
        else:
            self.images = self.images[int(0.8 * len(self.images)) :]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image_name = self.images[index].name

        source_img = cv2.imread(str(self.source_dir / image_name))
        source_img = cv2.cvtColor(source_img, cv2.COLOR_BGR2RGB)
        source_img = self.transform(source_img)

        attacked_img = cv2.imread(str(self.attacked_dir / image_name))
        attacked_img = cv2.cvtColor(attacked_img, cv2.COLOR_BGR2RGB)
        attacked_img = self.transform(attacked_img)
        return attacked_img, source_img
