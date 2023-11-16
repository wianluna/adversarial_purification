import torch
from torch.autograd import Variable

from torch import optim


# image quantization
def quantization(x):
    x_quan = torch.round(x * 255) / 255
    return x_quan


# picecwise-linear color filter
def CF(img, param, pieces):
    param = param[:, :, None, None]
    color_curve_sum = torch.sum(param, 4) + 1e-30
    total_image = img * 0
    for i in range(pieces):
        total_image += torch.clamp(img - 1.0 * i / pieces, 0, 1.0 / pieces) * param[:, :, :, :, i]
    total_image *= pieces / color_curve_sum
    return total_image


def step(model, optimizer, inputs, Paras, pieces, ref_image, metric_range, device):
    const = 0.5
    batch_size = inputs.shape[0]
    Paras.data = torch.clamp(Paras.data, min=0)
    Paras_sum = torch.sum(Paras.view(batch_size, 3, -1), dim=2, keepdim=True)

    # regularization on the adjustment
    l2 = torch.sum((((Paras / Paras_sum - 1 / pieces) ** 2)).view(batch_size, -1), dim=1)

    adv = CF(inputs, Paras, pieces)
    adv[adv > 1] = 1
    adv[adv < 0] = 0
    score = model(adv.to(device)).mean()
    l2_loss = (const * l2).sum()
    sign = -1 if model.lower_better else 1
    loss = score.to(device) * sign / metric_range + l2_loss.to(device)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return quantization(adv).detach(), score.detach(), loss.detach()


def attack(compress_image, ref_image=None, model=None, metric_range=100, device='cpu'):
    pieces = 64
    compress_image = Variable(compress_image.clone().to(device), requires_grad=False)

    batch_size = compress_image.shape[0]

    best_score = 10000
    o_best_adversary = compress_image.clone()

    Paras = torch.ones(batch_size, 3, pieces).to(device) * 1 / pieces
    Paras.requires_grad = True
    optimizer = optim.Adam([Paras], lr=0.001, betas=(0.9, 0.999), eps=1e-8)

    for iteration in range(2):
        # print(iteration, best_score)
        # perform the adversary
        adv, score, loss = step(
            model, optimizer, compress_image, Paras, pieces, ref_image, metric_range, device
        )
        if score < best_score:
            best_score = score
            o_best_adversary = adv.clone()

    res_image = (o_best_adversary).data.clamp_(min=0, max=1)
    return res_image
