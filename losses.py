import torch
import torch.nn.functional as F
from torch.nn import Module
from torch import Tensor
from torch.types import Device


def classification_loss(fake_logits: Tensor, labels: Tensor) -> Tensor:
    """
    Calculates classification loss
    :param fake_logits: fake logits
    :param labels: true class labels
    :return: classification loss
    """
    return F.cross_entropy(fake_logits, labels)


def discriminator_loss(real_output: Tensor, fake_output: Tensor, device: Device,
                       label_smoothing_val: float = 1) -> Tensor:
    """
    Calculates discriminator loss
    :param real_output: real image output of discriminator
    :param fake_output: fake image output of discriminator
    :param device: device to run on
    :param label_smoothing_val: label smoothing value for real images
    :return: discriminator loss
    """
    real_loss = F.binary_cross_entropy_with_logits(
        real_output, torch.ones_like(real_output, device=device) * label_smoothing_val)
    fake_loss = F.binary_cross_entropy_with_logits(
        fake_output, torch.zeros_like(fake_output, device=device))
    return (real_loss + fake_loss) / 2


def generator_loss(fake_output: Tensor, device: Device) -> Tensor:
    """
    Calculates generator loss
    :param fake_output: fake image output of discriminator
    :param device: device to run on
    :return: generator loss
    """
    return F.binary_cross_entropy_with_logits(fake_output,
                                              torch.ones_like(fake_output, device=device))


def L1_loss(real_image: Tensor, fake_image: Tensor) -> Tensor:
    """
    Calculates L1 loss
    :param real_image: real image
    :param fake_image: fake image from generator
    :return: L1 loss
    """
    return F.l1_loss(real_image, fake_image)


def L2_loss(real_image: Tensor, fake_image: Tensor) -> Tensor:
    """
    Calculates L2 loss
    :param real_image: real image
    :param fake_image: fake image from generator
    :return: L2 loss
    """
    return F.mse_loss(real_image, fake_image)


class DiscriminatorCriterion:

    def __init__(self, device: Device, wgan: bool = False, wgan_gp: bool = False,
                 gp_lambda: float = 10, gp_type: str = 'mixed', gp_constant: float = 1,
                 label_smoothing_value: float = 1):
        """
        Discriminator loss
        :param device: device to run on
        :param wgan: whether to use Wasserstein loss
        :param wgan_gp: whether to use gradient penalty
        :param gp_lambda: gradient penalty lambda
        :param gp_type: gradient penalty type, can be 'mixed', 'real', 'fake'
        :param gp_constant: gradient penalty constant
        :param label_smoothing_value: label smoothing value for real images
        """
        self.device = device
        self.wgan = wgan
        self.wgan_gp = wgan_gp
        assert (wgan if wgan_gp else True), 'WGAN-GP must be used with WGAN'
        self.gp_lambda = gp_lambda
        # real uses real image, fake uses fake image, mixed uses both
        assert gp_type in ['mixed', 'real', 'fake'], 'Invalid gradient penalty type: ' + gp_type
        self.gp_type = gp_type
        self.gp_constant = gp_constant
        self.label_smoothing_value = label_smoothing_value

    def __call__(self, input_images: Tensor, real_output: Tensor, fake_output: Tensor,
                 real_images: Tensor = None, fake_images: Tensor = None,
                 discriminator: Module = None, is_eval: bool = False):
        """
        Calculates discriminator loss
        :param input_images: input images
        :param real_output: real image output of discriminator
        :param fake_output: fake image output of discriminator
        :param real_images: real images
        :param fake_images: fake images
        :param discriminator: discriminator instance
        :param is_eval: pass is_eval=True to disable gradient penalty calculation when evaluating
        :return: disc gan loss, gradient penalty (none if not used)
        """
        grad_pen = None
        if self.wgan:
            loss = torch.mean(fake_output) - torch.mean(real_output)
            if self.wgan_gp:
                if not is_eval:
                    grad_pen = self._gradient_penalty(input_images, real_images, fake_images,
                                                      discriminator)
                else:
                    grad_pen = torch.tensor(0.0, device=self.device)
            return loss, grad_pen
        else:
            return discriminator_loss(real_output, fake_output, self.device,
                                      self.label_smoothing_value), grad_pen

    def _gradient_penalty(self, input_images: Tensor, real_images: Tensor, fake_images: Tensor,
                          discriminator: Module) -> Tensor:
        """
        Calculates gradient penalty
        :param input_images: input images
        :param real_images: real images
        :param fake_images: fake images
        :param discriminator: discriminator
        :return: gradient penalty
        """
        if self.gp_type == 'real':
            interpolates = real_images
        elif self.gp_type == 'fake':
            interpolates = fake_images
        elif self.gp_type == 'mixed':
            # Random weight term for interpolation between real and fake data
            alpha = torch.rand((real_images.size(0), 1, 1, 1), device=self.device)
            # Get random interpolation between real and fake data
            interpolates = (alpha * real_images + ((1 - alpha) * fake_images.detach()))
            interpolates = torch.autograd.Variable(interpolates, requires_grad=True)
            # interpolates.requires_grad_(True)
        else:
            raise ValueError("Invalid gradient penalty type")

        disc_interpolates = discriminator(torch.cat([input_images, interpolates], dim=1))
        grad_outs = torch.ones(disc_interpolates.size(), device=self.device)

        # Get gradient w.r.t. interpolates
        gradients = torch.autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                                        grad_outputs=grad_outs, create_graph=True,
                                        retain_graph=True, only_inputs=True)[0]
        gradients = gradients.view(gradients.size(0), -1)

        gradient_penalty = torch.mean(((gradients + 1e-16).norm(2, dim=1) - self.gp_constant) ** 2)

        return gradient_penalty * self.gp_lambda


class GeneratorCriterion:

    def __init__(self, device: Device, wgan: bool = False, l1_lambda: float = 0,
                 l2_lambda: float = 0, class_lambda: float = 0):
        """
        Generator loss
        :param device: device to run on
        :param wgan: whether to use Wasserstein loss
        :param l1_lambda: L1 loss lambda
        :param l2_lambda: L2 loss lambda
        :param class_lambda: class loss lambda
        """
        self.device = device
        self.wgan = wgan
        self.l1_lambda = l1_lambda
        self.l2_lambda = l2_lambda
        self.class_lambda = class_lambda

    def __call__(self, fake_output: Tensor, real_image: Tensor, fake_image: Tensor,
                 class_logits: Tensor, labels: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """
        :param fake_output: fake image output of discriminator
        :param real_image: real image
        :param fake_image: fake image from generator
        :param class_logits: class predictions (logits)
        :param labels: class target (logits)
        :return: generator loss
        """
        if self.wgan:
            gan_loss = -torch.mean(fake_output)
        else:
            gan_loss = generator_loss(fake_output, self.device)

        recon_loss = torch.zeros_like(gan_loss)

        if self.l1_lambda > 0:
            recon_loss += self.l1_lambda * L1_loss(real_image, fake_image)
        if self.l2_lambda > 0:
            recon_loss += self.l2_lambda * L2_loss(real_image, fake_image)

        if self.class_lambda > 0 and class_logits is not None:
            class_loss = self.class_lambda * classification_loss(class_logits, labels)
        else:
            class_loss = torch.zeros_like(gan_loss)

        return gan_loss, recon_loss, class_loss
