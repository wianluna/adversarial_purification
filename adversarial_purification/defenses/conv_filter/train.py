import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from pytorch_msssim import ssim
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from adversarial_purification.defenses.conv_filter.dataset import PurificationDataset
from adversarial_purification.defenses.conv_filter.fcn import FCN
from adversarial_purification.defenses.conv_filter.model import MetricModel


def tensors_to_images(img):
    img_dump = img.permute(0, 2, 3, 1).cpu().detach().type(torch.float32)  # [b,w,h,c]
    return img_dump.numpy()


def save_image(source_image, attacked_image, defended_image, epoch):
    fig, axs = plt.subplots(nrows=1, ncols=3, squeeze=False, figsize=(12, 8))

    axs[0, 0].set_title('Clean')
    axs[0, 0].imshow(tensors_to_images(source_image)[0])
    axs[0, 1].set_title('Attacked')
    axs[0, 1].imshow(tensors_to_images(attacked_image)[0])
    axs[0, 2].set_title('Purified')
    axs[0, 2].imshow(tensors_to_images(defended_image)[0])

    fig.savefig(f'train_fcn_examples/epoch={epoch}.png')


def train(attacked_dir, source_dir, metric_checkpoints, n_epoch=200):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    Path('train_fcn_examples').mkdir(parents=True, exist_ok=True)

    model = FCN(3, 3, 64, norm_type='instance', act_type='relu').to(device)

    train_data = PurificationDataset(attacked_dir, source_dir, mode='train')
    train_loader = DataLoader(train_data, batch_size=4, shuffle=True)

    test_data = PurificationDataset(attacked_dir, source_dir, mode='test')
    test_loader = DataLoader(test_data, batch_size=4, shuffle=False)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = torch.nn.MSELoss()

    metric_model = MetricModel(device, metric_checkpoints)
    metric_model.eval()

    best_ssim = 0
    best_epoch = 0

    for epoch in range(n_epoch):
        loss_list = []
        model.train()

        for i, (attacked_images, clean_images) in tqdm(enumerate(train_loader)):
            clean_images = clean_images.to(device)
            attacked_images = attacked_images.to(device)
            purified_images = model(attacked_images)

            loss = criterion(purified_images, clean_images)

            loss_list.append(loss.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(epoch)
        print(f'Epoch {epoch}: train loss = {np.array(loss_list).mean()}')

        test_loss_list = []
        ssim_list = []
        model.eval()

        with torch.no_grad():
            for i, (attacked_images, clean_images) in tqdm(enumerate(test_loader)):
                clean_images = clean_images.to(device)
                attacked_images = attacked_images.to(device)

                purified_images = model(attacked_images)
                purified_images.clamp_(0.0, 1.0)

                if i == len(test_loader) - 1:
                    save_image(
                        clean_images[-1].unsqueeze(0),
                        attacked_images[-1].unsqueeze(0),
                        purified_images[-1].unsqueeze(0),
                        epoch,
                    )
                loss = criterion(purified_images, clean_images)
                test_loss_list.append(loss.item())
                ssim_val = ssim(purified_images, clean_images, data_range=1, size_average=False)
                ssim_list.extend([val.item() for val in ssim_val])

        ssim_mean = np.array(ssim_list).mean()
        if ssim_mean > best_ssim:
            best_ssim = ssim_mean
            best_epoch = epoch
            checkpoint = {'model': model.state_dict(), 'epoch': epoch}
            torch.save(checkpoint, Path(metric_checkpoints).parent / 'best_fcn.pth')
        print(f'Epoch {epoch}: test loss = {np.array(test_loss_list).mean()}')
        print(f'Epoch {epoch}: ssim = {ssim_mean}')

    print(f'Best SSIM = {best_ssim} ({best_epoch} epoch)')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--reference-dir',
        type=str,
        help='path to source images',
        default='../../space-1/hackathon/dataset/train/reference',
    )
    parser.add_argument(
        '--data-dir',
        type=str,
        help='path to attacked images',
        default='../../space-1/hackathon/dataset/train/attacked',
    )
    parser.add_argument(
        '--metric-checkpoints',
        type=str,
        help='path to linearity checkpoints',
        default='../checkpoints/p1q2.pth',
    )
    args = parser.parse_args()

    train(args.data_dir, args.reference_dir, args.metric_checkpoints)


if __name__ == '__main__':
    main()
