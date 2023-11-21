import torch

from defenses.conv_filter.fcn import FCN


class FCNDefense:
    def __init__(self, fcn_path, device):
        self.defence_model = FCN(3, 3, 64, norm_type='instance', act_type='relu').to(device)

        checkpoint = torch.load(fcn_path)
        self.defence_model.load_state_dict(checkpoint['model'])
        print(f'Epoch = {checkpoint["epoch"]}')
        self.defence_model.eval()

    def __call__(self, image):
        return self.defence_model(image).clamp(0.0, 1.0)
