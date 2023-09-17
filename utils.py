import datetime
import torch
from torch.nn import Module


def format_time(elapsed):
    """
    :param elapsed: time in seconds
    :return: formatted time string in hh:mm:ss
    """
    # Round to the nearest second.
    elapsed_rounded = round(elapsed)

    # Format as hh:mm:ss
    return str(datetime.timedelta(seconds=elapsed_rounded))


class GaussianNoise(Module):
    """
    Add gaussian noise to a torch tensor

    Args:
        mean (float): mean of the gaussian distribution. Default: 0.0
        std (float): standard deviation of the gaussian distribution. Default: 1.0
    """
    def __init__(self, mean: float = 0.0, std: float = 0.1, *args, **kwargs):
        """
        :param mean: mean of the gaussian distribution
        :param std: standard deviation of the gaussian distribution
        """
        super().__init__(*args, **kwargs)
        self.mean = mean
        self.std = std

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        """
        :param img: torch tensor
        :return: torch tensor with added noise
        """
        if not isinstance(img, torch.Tensor):
            raise TypeError(f"img should be a torch tensor. Got {type(img)}")

        return img + torch.randn(img.size()) * self.std + self.mean

    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)
