import argparse
from glob import glob
import os
from pathlib import Path
import warnings

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from skimage.metrics import peak_signal_noise_ratio, structural_similarity, mean_squared_error
import torch
from torchvision import transforms
from torch.utils.data import Dataset
from tqdm import tqdm

from model import MetricModel

warnings.filterwarnings("ignore")


class AttackedDataset(Dataset):
    def __init__(
        self, data_dir: str, reference_dir: str, transform: bool = True, attacks: list = None
    ):
        """
        :param data_dir: Directory with all the images.
        :param reference_dir: Directory with reference (clear) images.
        :param transform: True for transformation to pass to model.
        False to return pristine image to display
        :param attack: Attack method. If None return images for all attacks.
        """
        self.data_dir = data_dir
        self.reference_dir = reference_dir

        if transform:
            self.transform = transforms.ToTensor()
        else:
            self.transform = None

        if attacks:
            self.attacked_images = []
            for attack in attacks:
                self.attacked_images.extend(sorted(glob(os.path.join(self.data_dir, attack, '*'))))
        else:
            self.attacked_images = sorted(glob(os.path.join(self.data_dir, '*', '*')))

    def __len__(self):
        return len(self.attacked_images)

    def __getitem__(self, idx):
        img_name = self.attacked_images[idx]
        image = cv2.imread(img_name)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        attack_name = img_name.split('/')[-2]

        reference_path = os.path.join(self.reference_dir, os.path.basename(img_name))
        reference_image = cv2.imread(reference_path)
        reference_image = cv2.cvtColor(reference_image, cv2.COLOR_BGR2RGB)

        if self.transform:
            image = self.transform(image).unsqueeze(0)
            reference_image = self.transform(reference_image).unsqueeze(0)

        sample = {
            'attacked_image': image,
            'reference_image': reference_image,
            'attack_name': attack_name,
        }

        return sample


def to_numpy(x):
    if torch.is_tensor(x):
        x = x.cpu().detach().permute(0, 2, 3, 1).numpy()
    return x if len(x.shape) == 4 else x[np.newaxis]


def PSNR(x, y):
    return peak_signal_noise_ratio(to_numpy(x), to_numpy(y), data_range=1)


def SSIM(x, y):
    ssim_sum = 0
    for img1, img2 in zip(to_numpy(x), to_numpy(y)):
        ssim = structural_similarity(img1, img2, channel_axis=2, data_range=1)
        ssim_sum += ssim
    return ssim_sum / len(x)


def tensors_to_images(img):
    img_dump = img.permute(0, 2, 3, 1).cpu().detach().type(torch.float32)  # [b,w,h,c]
    return img_dump.numpy()


def save_image(source_image, attacked_image, defended_image, save_path):
    fig, axs = plt.subplots(nrows=1, ncols=3, squeeze=False, figsize=(12, 8))

    axs[0, 0].set_title('Clean')
    axs[0, 0].imshow(tensors_to_images(source_image)[0][..., ::-1])
    axs[0, 1].set_title('Attacked')
    axs[0, 1].imshow(tensors_to_images(attacked_image)[0][..., ::-1])
    axs[0, 2].set_title('Purified')
    axs[0, 2].imshow(tensors_to_images(defended_image)[0][..., ::-1])

    fig.savefig(save_path)


def plot_res():
    results = pd.read_csv('results.csv')
    results['defense_type'] = results['defense_type'].fillna('without defense')
    results = pd.concat([results[results['info'].isna()]])

    for attack_type in results['attack_type'].unique():
        attack_res = results[results['attack_type'] == attack_type]

        for _, res in attack_res.iterrows():
            plt.scatter(res['quality_score'], res['gain_score'], label=res['defense_type'])

        plt.title(f'{attack_type}')
        plt.xlabel('quality score')
        plt.ylabel('gain score')
        plt.legend()
        plt.savefig(f'res_{attack_type}.png')
        plt.clf()


def plot_hist(dataset):
    attacked_hist = [[], [], []]
    source_hist = [[], [], []]

    for i, sample in tqdm(enumerate(dataset)):
        attacked_image = to_numpy(sample['attacked_image'])
        source_image = to_numpy(sample['reference_image'])

        for i in [0, 1, 2]:
            attacked_image_ch, _ = np.histogram(
                np.array(attacked_image)[:, :, :, i].flatten(), bins=255
            )
            source_image_ch, _ = np.histogram(
                np.array(source_image)[:, :, :, i].flatten(), bins=255
            )

            attacked_hist[i].append(attacked_image_ch)
            source_hist[i].append(source_image_ch)

    fig, axes = plt.subplots(2, 3, figsize=(12, 8))

    rows = [f'{label} images' for label in ['Clean', 'Attacked']]
    cols = ['R', 'G', 'B']

    for ax, col in zip(axes[0], cols):
        ax.set_title(col)

    for ax, row in zip(axes[:, 0], rows):
        ax.set_ylabel(row, rotation=90, size='large')

    plt.subplots_adjust(left=0.15, bottom=0.1, right=0.9, top=0.9, wspace=0.4, hspace=0.4)

    for i in [0, 1, 2]:
        attacked_hist_mean = np.array(attacked_hist[i]).mean(axis=0)
        source_hist_mean = np.array(source_hist[i]).mean(axis=0)
        plt.subplot(2, 3, 1 + i)
        plt.bar(list(range(0, 255)), source_hist_mean)
        plt.subplot(2, 3, 4 + i)
        plt.bar(list(range(0, 255)), attacked_hist_mean)

    fig.savefig('hist.png')


def get_score(model, attacked_image, source_image, save_path=None):
    defense_model_score = model(attacked_image, defense=True).item()
    attacked_model_score = model(attacked_image, defense=False).item()

    source_image_score = model(source_image, defense=False).item()

    cleared_image = model.defense(attacked_image)

    if cleared_image.shape != source_image.shape:
        cleared_image = transforms.Resize(list(source_image.shape[2:]))(cleared_image)

    psnr = PSNR(source_image, cleared_image)
    ssim = SSIM(source_image, cleared_image)

    mse_origin = mean_squared_error(to_numpy(source_image), to_numpy(attacked_image))
    if mse_origin == 0:
        psnr = 150

    quality_score = ssim + psnr / 80
    gain_score = abs(source_image_score - defense_model_score) / source_image_score

    if save_path:
        save_image(source_image, attacked_image, cleared_image, save_path)
    return (
        quality_score / 2,
        gain_score,
        [source_image_score, defense_model_score, attacked_model_score, psnr, ssim],
    )


def test(model, dataset, device, attack_type, defense_type, q=None):
    quality_score_arr = []
    gain_score_arr = []
    score_arr = []

    for i, sample in tqdm(enumerate(dataset)):
        attacked_image = sample['attacked_image'].to(device)
        source_image = sample['reference_image'].to(device)

        quality_score, gain_score, score = get_score(
            model,
            attacked_image,
            source_image,
            save_path=f'image_examples/{attack_type}_{defense_type}_{q}.png' if i == 2 else None,
        )

        quality_score_arr.append(quality_score)
        gain_score_arr.append(gain_score)
        score_arr.append(score)

    quality_score = np.mean(quality_score_arr)
    gain_score = np.mean(gain_score_arr)
    print(quality_score)
    print(gain_score)
    print(np.mean(score_arr, axis=0))

    return quality_score, gain_score


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--reference_dir",
        type=str,
        help="path to source image",
        default="/home/a.chistyakova/space-1/hackathon/dataset/test/reference",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="/home/a.chistyakova/space-1/hackathon/dataset/test/attacked",
    )
    parser.add_argument("--device", type=str, help="device to run model on", default="cuda")
    parser.add_argument("--plot-hist", action='store_true')
    parser.add_argument("--plot-res", action='store_true')
    parser.add_argument("--defense", type=str, default=None)
    args = parser.parse_args()

    if args.plot_res:
        plot_res()
        return

    attacks = {
        # 'color_attack': ['adv-cf'],
        # 'zhang': ['zhang-et-al-dists', 'zhang-et-al-lpips', 'zhang-et-al-ssim'],
        # 'fgsm': ['amifgsm', 'ifgsm', 'mifgsm'],
        # 'korhonen': ['korhonen-et-al'],
        # 'madc': ['madc'],
        # 'ssah': ['ssah'],
        'all_attacks': [
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
    }

    results = []

    for attack_type in attacks:
        print(attack_type)
        dataset = AttackedDataset(
            data_dir=args.data_dir, reference_dir=args.reference_dir, attacks=attacks[attack_type]
        )

        if args.plot_hist:
            plot_hist(dataset)
            return
        print(len(dataset))
        if args.defense == 'jpeg-all':
            qfs = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
            quality_score_arr = []
            gain_score_arr = []
            for q in qfs:
                model = MetricModel(
                    args.device, 'p1q2.pth', defense_type=args.defense, defense_params={'q': q}
                )
                model.eval()

                quality_score, gain_score = test(
                    model, dataset, args.device, attack_type, 'jpeg', q=q
                )
                quality_score_arr.append(quality_score)
                gain_score_arr.append(gain_score)
            pd.DataFrame(
                data={
                    'q': qfs,
                    'quality_score': quality_score_arr,
                    'gain_score': gain_score_arr,
                }
            ).to_csv(f'{attack_type}_jpeg_results.csv', index=False)
        elif args.defense == 'fcn_filter':
            # if attack_type != 'color_attack':
            #     continue

            model = MetricModel(args.device, 'p1q2.pth', defense_type=args.defense)
            model.eval()

            quality_score, gain_score = test(model, dataset, args.device, attack_type, args.defense)
            results.append(
                {
                    'attack_type': attack_type,
                    'defense_type': args.defense,
                    'quality_score': quality_score,
                    'gain_score': gain_score,
                }
            )
        else:
            model = MetricModel(args.device, 'p1q2.pth', defense_type=args.defense)
            model.eval()

            quality_score, gain_score = test(model, dataset, args.device, attack_type, args.defense)
            results.append(
                {
                    'attack_type': attack_type,
                    'defense_type': args.defense,
                    'quality_score': quality_score,
                    'gain_score': gain_score,
                }
            )

    if not Path('results.csv').exists():
        pd.DataFrame(columns=['attack_type', 'defense_type', 'quality_score', 'gain_score']).to_csv(
            'results.csv', index=False
        )

    pd.concat([pd.read_csv('results.csv'), pd.DataFrame(data=results)]).to_csv(
        'results.csv', index=False
    )


if __name__ == "__main__":
    main()
