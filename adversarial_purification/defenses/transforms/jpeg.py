import cv2
import numpy as np
import torch


def to_numpy(x: torch.Tensor):
    if torch.is_tensor(x):
        x = x.cpu().detach().permute(0, 2, 3, 1).numpy()
    return x if len(x.shape) == 4 else x[np.newaxis]


def to_torch(x, device='cuda'):
    if isinstance(x, np.ndarray):
        x = torch.from_numpy(x)
        if len(x.shape) == 3:
            x = x.permute(2, 0, 1).unsqueeze(0)
        else:
            x = x.permute(0, 3, 1, 2)
        x = x.type(torch.FloatTensor).to(device)
    return x


class JpegDefense:
    def __init__(self, q=50):
        """Compress images with JPEG

        :param q: Quality factor.
        """
        self.q = q

    def __call__(self, image):
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), self.q]
        np_batch = to_numpy(image)

        if len(np_batch.shape) == 3:
            np_batch = np_batch[np.newaxis]
        jpeg_batch = np.empty(np_batch.shape)

        for i in range(len(np_batch)):
            result, enc_img = cv2.imencode('.jpg', np_batch[i] * 255, encode_param)
            jpeg_batch[i] = cv2.imdecode(enc_img, 1) / 255

        return torch.nan_to_num(to_torch(jpeg_batch), nan=0)
