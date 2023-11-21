import torch.nn as nn


class FCN(nn.Module):
    def __init__(self, input_nc, output_nc, ngf, norm_type='batch', act_type='selu'):
        super(FCN, self).__init__()

        self.conv1 = nn.Conv2d(input_nc, 64, 3, 1, 1)
        self.conv2 = nn.Conv2d(64, 16, 3, 1, 1)
        self.conv3 = nn.Conv2d(16, 3, 3, 1, 1)

        if norm_type == 'batch':
            self.norm = nn.BatchNorm2d(16)
            self.norm2 = nn.BatchNorm2d(8)
        elif norm_type == 'instance':
            self.norm = nn.InstanceNorm2d(16)
            self.norm2 = nn.InstanceNorm2d(8)
        self.leaky_relu = nn.LeakyReLU(0.2, True)

        if act_type == 'selu':
            self.act = nn.SELU(True)
        else:
            self.act = nn.ReLU(True)

        self.dropout = nn.Dropout(0.5)

        self.tanh = nn.Tanh()

    def forward(self, inputs):
        # Encoder
        # Convolution layers:
        # input is (nc) x 512 x 1024
        e1 = self.conv1(inputs)
        # print(f'e1 {e1.shape}')
        # state size is (ngf) x 256 x 512
        e2 = self.norm2(self.conv2(self.leaky_relu(e1)))
        # print(f'e2 {e2.shape}')
        # state size is (ngf x 2) x 128 x 256
        e3 = self.conv3(self.leaky_relu(e2))
        # print(f'e3 {e3.shape}')

        output = self.tanh(e3)
        return output
