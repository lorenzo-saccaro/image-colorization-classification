import os
import time
from utils import format_time
from tqdm import tqdm
import torch
import matplotlib.pyplot as plt
from IPython.display import clear_output
from kornia.color import lab_to_rgb
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F


class Checkpoint:

    def __init__(self, generator, discriminator, gen_optimizer, disc_optimizer, gen_scheduler,
                 disc_scheduler, path):
        """
        :param generator: Generator model
        :param discriminator: Discriminator model
        :param gen_optimizer: Generator optimizer
        :param disc_optimizer: Discriminator optimizer
        :param gen_scheduler: Generator lr scheduler
        :param disc_scheduler: Discriminator lr scheduler
        :param path: checkpoint file path
        """
        self.generator = generator
        self.discriminator = discriminator
        self.gen_optimizer = gen_optimizer
        self.disc_optimizer = disc_optimizer
        self.gen_scheduler = gen_scheduler
        self.disc_scheduler = disc_scheduler
        self.path = path

    def load(self) -> dict or None:
        """
        Loads the generator, discriminator, optimizers and schedulers states and history from the checkpoint file.
        :return: history if checkpoint exists, else None
        """
        if os.path.exists(self.path):
            print("Loading checkpoint from {}".format(self.path))
            checkpoint = torch.load(self.path)
            self.generator.load_state_dict(checkpoint['generator'])
            print("Generator loaded")
            self.discriminator.load_state_dict(checkpoint['discriminator'])
            print("Discriminator loaded")
            self.gen_optimizer.load_state_dict(checkpoint['gen_optimizer'])
            print("Generator optimizer loaded")
            self.disc_optimizer.load_state_dict(checkpoint['disc_optimizer'])
            print("Discriminator optimizer loaded")
            if self.gen_scheduler is not None:
                self.gen_scheduler.load_state_dict(checkpoint['gen_scheduler'])
                print("Generator scheduler loaded")
            if self.disc_scheduler is not None:
                self.disc_scheduler.load_state_dict(checkpoint['disc_scheduler'])
                print("Discriminator scheduler loaded")
            return checkpoint['history']
        else:
            print("No checkpoint found at {}".format(self.path))
            print("Starting from scratch")
            return None

    def save(self, history) -> None:
        """
        Saves the generator, discriminator, optimizers and schedulers states and history to the checkpoint file.
        :param history: a dictionary containing the training and validation metrics
        :return:
        """
        print("Saving checkpoint to {}".format(self.path))
        torch.save({
            'generator': self.generator.state_dict(),
            'discriminator': self.discriminator.state_dict(),
            'gen_optimizer': self.gen_optimizer.state_dict(),
            'disc_optimizer': self.disc_optimizer.state_dict(),
            'gen_scheduler': self.gen_scheduler.state_dict() if self.gen_scheduler is not None else None,
            'disc_scheduler': self.disc_scheduler.state_dict() if self.disc_scheduler is not None else None,
            'history': history
        }, self.path)
        print("Checkpoint saved")

    def delete(self) -> None:
        """
        Deletes the checkpoint file.
        :return:
        """
        if os.path.exists(self.path):
            os.remove(self.path)
            print("Checkpoint deleted")


class Trainer:
    def __init__(self, generator, discriminator, gen_optimizer, disc_optimizer,
                 gen_scheduler, disc_scheduler, gen_criterion, disc_criterion, device, train_loader,
                 val_loader, metrics, options):
        """
        :param generator: Generator model
        :param discriminator: Discriminator model
        :param gen_optimizer: Generator optimizer
        :param disc_optimizer: Discriminator optimizer
        :param gen_scheduler: Generator lr scheduler
        :param disc_scheduler: Discriminator lr scheduler
        :param gen_criterion: Generator loss function
        :param disc_criterion: Discriminator loss function
        :param device: Device to use for training
        :param train_loader: Training data loader
        :param val_loader: Validation data loader
        :param metrics: list of instances of metrics objects from torchmetrics
        :param options: Training options, a dictionary
        """

        self.generator = generator
        self.discriminator = discriminator
        self.gen_optimizer = gen_optimizer
        self.disc_optimizer = disc_optimizer
        self.gen_scheduler = gen_scheduler
        self.disc_scheduler = disc_scheduler
        self.gen_criterion = gen_criterion
        self.disc_criterion = disc_criterion
        self.device = device
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.metrics = metrics
        self.options = options

        if self.options['tensorboard_path'] is not None:
            if self.options['tensorboard_path'] == 'default':
                self.tb_writer = SummaryWriter()
            else:
                self.tb_writer = SummaryWriter(self.options['tensorboard_path'])

            layout = {
                'Train': {
                    'gen_loss_tot': ["Multiline", ["gen_loss_tot"]],
                    'gen_loss_gan': ["Multiline", ["gen_loss_gan"]],
                    'gen_loss_recon': ["Multiline", ["gen_loss_recon"]],
                    'gen_loss_class': ["Multiline", ["gen_loss_class"]],
                    'disc_loss_tot': ["Multiline", ["disc_loss_tot"]],
                    'gp_loss': ["Multiline", ["gp_loss"]],
                    'disc_loss_gan': ["Multiline", ["disc_loss_gan"]]
                },
                'Epoch': {
                    'epoch_gen_loss_tot': ["Multiline", ["epoch_gen_loss_tot/train",
                                                         "epoch_gen_loss_tot/val"]],
                    'epoch_gen_loss_gan': ["Multiline", ["epoch_gen_loss_gan/train",
                                                         "epoch_gen_loss_gan/val"]],
                    'epoch_gen_loss_recon': ["Multiline", ["epoch_gen_loss_recon/train",
                                                           "epoch_gen_loss_recon/val"]],
                    'epoch_gen_loss_class': ["Multiline", ["epoch_gen_loss_class/train",
                                                           "epoch_gen_loss_class/val"]],
                    'epoch_disc_loss_tot': ["Multiline", ["epoch_disc_loss_tot/train",
                                                          "epoch_disc_loss_tot/val"]],
                    'epoch_gp_loss': ["Multiline", ["epoch_gp_loss/train",
                                                    "epoch_gp_loss/val"]],
                    'epoch_disc_loss_gan': ["Multiline", ["epoch_disc_loss_gan/train",
                                                          "epoch_disc_loss_gan/val"]]
                }
            }

            metric_keys = ['val_{}'.format(metric._get_name().lower()) for metric in self.metrics]
            for metric_key in metric_keys:
                layout['Epoch'][metric_key] = ["Multiline", [metric_key.replace('_', '/')]]

            self.tb_writer.add_custom_scalars(layout)

        if self.options['clip_weights']:
            self.clipper = ClipWeights(clip_value=self.options['clip_value'])
        else:
            self.clipper = None

        if options['checkpoint_path'] is not None:
            self.checkpoint = Checkpoint(generator, discriminator, gen_optimizer, disc_optimizer,
                                         gen_scheduler, disc_scheduler, options['checkpoint_path'])
            if options['reset_training']:
                self.checkpoint.delete()
            self.history = self.checkpoint.load()

        else:
            self.checkpoint = None
            self.history = None

    def _update_metrics(self, inputs, targets, outputs, class_logits, class_labels) -> None:
        """
        Updates the metrics for the current batch.
        :param inputs: input images
        :param targets: target images
        :param outputs: generated images
        :param class_logits: predicted class logits
        :param class_labels: class labels
        :return:
        """
        if self.options['use_lab_colorspace']:
            # map values in [0, 1] and combine channels, inputs, targets and outputs are in range [-1, 1]
            L = inputs
            L = (L + 1) * 0.5
            ab_true = targets * 0.5 + 0.5
            ab_pred = outputs * 0.5 + 0.5
            true_imgs = torch.cat((L, ab_true), dim=1)
            fake_imgs = torch.cat((L, ab_pred), dim=1)
        else:
            # map to [0, 1] range
            true_imgs = targets
            true_imgs = true_imgs * 0.5 + 0.5
            fake_imgs = outputs
            fake_imgs = fake_imgs * 0.5 + 0.5

        # compute the metrics
        for metric in self.metrics:
            if metric._get_name() == 'FrechetInceptionDistance':
                metric.update(true_imgs, real=True)
                metric.update(fake_imgs, real=False)
            elif metric._get_name() == "MulticlassAccuracy":
                metric.update(class_logits, class_labels)
            else:
                metric.update(fake_imgs, true_imgs)

    def _compute_metrics(self) -> None:
        """
        Computes the metrics for the current epoch.
        :return:
        """
        for metric in self.metrics:
            name = metric._get_name().lower()
            key = 'val_{}'.format(name)
            val = metric.compute()
            self.history.setdefault(key, []).append(val.item())
            metric.reset()

    def _plot_images(self, inputs, targets, outputs, split, epoch, step) -> None:
        """
        Plots the generated images, side by side with the input and target images.
        :param inputs: input images in range [-1, 1]
        :param targets: target images in range [-1, 1]
        :param outputs: generated images in range [-1, 1]
        :param split: 'train' or 'val'
        :param epoch: current epoch
        :param step: current step
        :return:
        """
        title = '{} images at epoch {} step {}'.format(split, epoch, step)
        # limit the number of images to plot
        n_plot = min(inputs.size(0), 8)
        inputs = inputs[:n_plot]
        targets = targets[:n_plot]
        outputs = outputs[:n_plot]

        with torch.no_grad():
            if self.options['use_lab_colorspace']:
                # prepare data for conversion to rgb
                L = (inputs + 1) * 50
                ab_true = targets
                ab_true[ab_true > 0] *= 127
                ab_true[ab_true < 0] *= 128
                ab_pred = outputs
                ab_pred[ab_pred > 0] *= 127
                ab_pred[ab_pred < 0] *= 128

                true_imgs = torch.cat((L, ab_true), dim=1)
                fake_imgs = torch.cat((L, ab_pred), dim=1)

                # convert to rgb
                true_imgs = lab_to_rgb(true_imgs)  # in range [0, 1]
                fake_imgs = lab_to_rgb(fake_imgs)  # in range [0, 1]
                input_imgs = inputs * 0.5 + 0.5  # in range [0, 1]
            else:
                true_imgs = targets * 0.5 + 0.5
                fake_imgs = outputs * 0.5 + 0.5
                input_imgs = inputs * 0.5 + 0.5

            true_imgs = true_imgs.permute(0, 2, 3, 1).cpu().detach().numpy()
            fake_imgs = fake_imgs.permute(0, 2, 3, 1).cpu().detach().numpy()
            input_imgs = input_imgs.permute(0, 2, 3, 1).cpu().detach().numpy()

        # Plot the first 8 input images, target images and generated images
        fig = plt.figure(figsize=(2 * n_plot, 9))
        for i in range(n_plot):
            ax = fig.add_subplot(3, n_plot, i + 1, xticks=[], yticks=[])
            ax.imshow(input_imgs[i], cmap='gray')
            ax = fig.add_subplot(3, n_plot, i + n_plot + 1, xticks=[], yticks=[])
            ax.imshow(fake_imgs[i])
            ax = fig.add_subplot(3, n_plot, i + 2 * n_plot + 1, xticks=[], yticks=[])
            ax.imshow(true_imgs[i])
        # set title
        fig.suptitle(title, fontsize=20)
        fig.tight_layout()
        # save figure
        fig.savefig(os.path.join(self.options['output_path'],
                                 '{}_epoch_{}_{}.png'.format(split, epoch, step)))
        if self.options['verbose']:
            plt.show()
        if self.tb_writer is not None:
            self.tb_writer.add_figure('{}/epoch_{}_{}'.format(split, epoch, step), fig)
            self.tb_writer.flush()

    def _print_summary(self, epoch: int) -> None:
        """
        Prints a summary of the current epoch.
        :param epoch: current epoch
        :return:
        """
        print("Epoch {} took {}".format(epoch + 1, format_time(self.history['epoch_times'][-1])))
        print("Train Generator Loss: {:.4f}".format(self.history['train_gen_loss_tot'][-1]))
        print("Train Generator Loss GAN: {:.4f}".format(self.history['train_gen_loss_gan'][-1]))
        print("Train Generator Loss Recon: {:.4f}".format(self.history['train_gen_loss_recon'][-1]))
        print("Train Generator Class Loss: {:.4f}".format(self.history['train_gen_loss_class'][-1]))
        print("Train Discriminator Loss: {:.4f}".format(self.history['train_disc_loss_tot'][-1]))
        if 'train_gp_loss' in self.history:
            print("Train Gradient Penalty Loss: {:.4f}".format(self.history['train_gp_loss'][-1]))
        print('\n')
        print("Valid Generator Loss: {:.4f}".format(self.history['val_gen_loss_tot'][-1]))
        print("Valid Generator Loss GAN: {:.4f}".format(self.history['val_gen_loss_gan'][-1]))
        print("Valid Generator Loss Recon: {:.4f}".format(self.history['val_gen_loss_recon'][-1]))
        print("Valid Generator Class Loss: {:.4f}".format(self.history['val_gen_loss_class'][-1]))
        print("Valid Discriminator Loss: {:.4f}".format(self.history['val_disc_loss_tot'][-1]))
        if 'val_gp_loss' in self.history:
            print("Valid Gradient Penalty Loss: {:.4f}".format(self.history['val_gp_loss'][-1]))
        print('\n')
        if self.metrics is not None:
            for metric in self.metrics:
                name = metric._get_name()
                print("Valid {}: {:.4f}".format(name, self.history['val_' + name.lower()][-1]))

    def _update_tensorboard(self) -> None:
        """
        Updates tensorboard with the epoch values
        :return:
        """
        for tag, value in self.history.items():
            if tag.startswith('val'):
                # check if corresponding train value exists
                train_tag = tag.replace('val', 'train')
                if train_tag in self.history:
                    train_plot_tag = train_tag.replace('train_', 'epoch_')
                    train_plot_tag += '/train'
                    self.tb_writer.add_scalar(train_plot_tag, self.history[train_tag][-1],
                                              self.history['elapsed_epochs'])
                    val_plot_tag = tag.replace('val_', 'epoch_')
                    val_plot_tag += '/val'
                    self.tb_writer.add_scalar(val_plot_tag, self.history[tag][-1],
                                              self.history['elapsed_epochs'])
                # if not, just add the validation value (e.g. for metrics)
                else:
                    self.tb_writer.add_scalar(tag.replace('_', '/'), self.history[tag][-1],
                                              self.history['elapsed_epochs'])
        self.tb_writer.flush()

    @staticmethod
    def _set_requires_grad(nets, requires_grad=False) -> None:
        """
        Set requires_grad=False for all the networks to avoid unnecessary computations
        :param nets: a list of networks
        :param requires_grad: whether the networks require gradients or not
        """
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad

    def _train_epoch(self) -> None:
        """
        Performs a single training epoch
        :return:
        """
        train_gen_loss = 0
        train_gen_loss_gan = 0
        train_gen_loss_recon = 0
        train_gen_loss_class = 0
        train_disc_loss = 0
        gp_loss = 0
        train_iter = tqdm(self.train_loader)

        # set models to train mode
        self.generator.train()
        self.discriminator.train()

        for i, (input_imgs, target_imgs, class_labels) in enumerate(train_iter):

            step_counter = self.history['elapsed_epochs'] * len(train_iter) + i

            # move images and labels to device
            input_imgs = input_imgs.to(self.device, non_blocking=True)
            target_imgs = target_imgs.to(self.device, non_blocking=True)
            class_labels = class_labels.to(self.device, non_blocking=True)

            fake_images = self.generator(input_imgs, return_class_out=False)

            # train discriminator
            self._set_requires_grad([self.discriminator], True)
            self.disc_optimizer.zero_grad()
            # combine input images and fake images
            fake_images_to_disc = torch.cat([input_imgs, fake_images], dim=1)
            # get discriminator predictions for fake images
            fake_preds = self.discriminator(fake_images_to_disc.detach())
            # combine input images and real images
            real_images_to_disc = torch.cat([input_imgs, target_imgs], dim=1)
            # get discriminator predictions for real images
            real_preds = self.discriminator(real_images_to_disc)
            # calculate discriminator loss
            disc_loss, gp = self.disc_criterion(input_imgs, real_preds, fake_preds, target_imgs,
                                                fake_images, self.discriminator)

            if gp is not None:
                disc_loss = disc_loss + gp
                gp_loss = gp_loss + gp.detach().cpu()

            # backpropagate disc loss
            disc_loss.backward()
            # update discriminator weights
            self.disc_optimizer.step()
            # clip discriminator weights
            if self.clipper is not None:
                self.discriminator.apply(self.clipper)

            train_disc_loss += disc_loss.detach().cpu()

            # train generator every n_critic iterations (for WGAN)
            if (step_counter + 1) % self.options['n_critic'] == 0:
                # set requires_grad to False for discriminator (to avoid unnecessary computations)
                self._set_requires_grad([self.discriminator], False)
                self.gen_optimizer.zero_grad()
                # generate fake images
                fake_images, class_logits = self.generator(input_imgs, return_class_out=True)
                # combine input images and fake images
                fake_images_to_disc = torch.cat([input_imgs, fake_images], dim=1)
                # get discriminator predictions for fake images
                fake_preds = self.discriminator(fake_images_to_disc)
                # calculate generator loss, gan loss + reconstruction loss (L1 or L2) + class loss
                gen_loss_gan, gen_loss_recon, class_loss = self.gen_criterion(fake_preds,
                                                                              target_imgs,
                                                                              fake_images,
                                                                              class_logits,
                                                                              class_labels)
                gen_loss = gen_loss_gan + gen_loss_recon + class_loss
                # backpropagate generator loss
                gen_loss.backward()
                # update generator weights
                self.gen_optimizer.step()

                # print training stats
                if self.options['verbose']:
                    if gp is not None:
                        display_string = "Gen loss: {:.4f}, Gen gan loss {:.4f}, Gen recon loss {:.4f}, Gen class loss {:.4f}, Disc loss: {:.4f}, Disc gan loss {:.4f}, GP :{:.4f}".format(
                            gen_loss, gen_loss_gan, gen_loss_recon, class_loss, disc_loss,
                            disc_loss - gp, gp)
                    else:
                        display_string = "Gen loss: {:.4f}, Gen gan loss {:.4f}, Gen recon loss {:.4f}, Gen class loss {:.4f}, Disc loss: {:.4f}".format(
                            gen_loss, gen_loss_gan, gen_loss_recon, class_loss, disc_loss)
                    train_iter.set_description(display_string)

                # update tensorboard
                if self.tb_writer is not None and step_counter % self.options['log_every_nstep'] == 0:
                    self.tb_writer.add_scalar('gen_loss_tot', gen_loss, step_counter)
                    self.tb_writer.add_scalar('gen_loss_gan', gen_loss_gan, step_counter)
                    self.tb_writer.add_scalar('gen_loss_recon', gen_loss_recon, step_counter)
                    self.tb_writer.add_scalar('gen_loss_class', class_loss, step_counter)
                    self.tb_writer.add_scalar('disc_loss_tot', disc_loss, step_counter)
                    if gp is not None:
                        self.tb_writer.add_scalar('gp_loss', gp, step_counter)
                        self.tb_writer.add_scalar('disc_loss_gan', disc_loss - gp,
                                                  step_counter)

                # update training losses
                train_gen_loss += gen_loss.detach().cpu()
                train_gen_loss_gan += gen_loss_gan.detach().cpu()
                train_gen_loss_recon += gen_loss_recon.detach().cpu()
                train_gen_loss_class += class_loss.detach().cpu()

            if (step_counter+1) % self.options['plot_every_nstep'] == 0:
                self._plot_images(input_imgs, target_imgs, fake_images, split='train',
                                  epoch=self.history['elapsed_epochs'] + 1, step=step_counter)

        train_gen_loss /= (len(self.train_loader) // self.options['n_critic'])
        train_gen_loss_gan /= (len(self.train_loader) // self.options['n_critic'])
        train_gen_loss_recon /= (len(self.train_loader) // self.options['n_critic'])
        train_gen_loss_class /= (len(self.train_loader) // self.options['n_critic'])
        train_disc_loss /= len(self.train_loader)
        gp_loss /= len(self.train_loader)
        self.history['train_gen_loss_tot'].append(train_gen_loss.item())
        self.history['train_gen_loss_gan'].append(train_gen_loss_gan.item())
        self.history['train_gen_loss_recon'].append(train_gen_loss_recon.item())
        self.history['train_gen_loss_class'].append(train_gen_loss_class.item())
        self.history['train_disc_loss_tot'].append(train_disc_loss.item())
        if isinstance(gp_loss, torch.Tensor):
            self.history.setdefault('train_gp_loss', []).append(gp_loss.item())
            self.history.setdefault('train_disc_loss_gan', []).append(train_disc_loss.item() - gp_loss.item())

    def _validate_epoch(self) -> None:
        """
        Performs a single validation epoch
        :return:
        """
        val_gen_loss = 0
        val_gen_loss_gan = 0
        val_gen_loss_recon = 0
        val_gen_class_loss = 0
        val_disc_loss = 0
        gp_loss = 0
        val_iter = tqdm(self.val_loader)

        # TODO: in the paper they use train mode also for validation
        #  (Dropout and BatchNorm in place of noise)
        self.generator.train()
        self.discriminator.train()

        for i, (input_imgs, target_imgs, class_labels) in enumerate(val_iter):
            # move images to device
            input_imgs = input_imgs.to(self.device, non_blocking=True)
            target_imgs = target_imgs.to(self.device, non_blocking=True)
            class_labels = class_labels.to(self.device, non_blocking=True)

            with torch.no_grad():
                # generate fake images
                fake_images, class_logits = self.generator(input_imgs, return_class_out=True)
                # combine input images and fake images
                fake_images_to_disc = torch.cat([input_imgs, fake_images], dim=1)
                # get discriminator predictions for fake images
                fake_preds = self.discriminator(fake_images_to_disc)
                # combine input images and real images
                real_images_to_disc = torch.cat([input_imgs, target_imgs], dim=1)
                # get discriminator predictions for real images
                real_preds = self.discriminator(real_images_to_disc)

                # calculate discriminator loss
                disc_loss, gp = self.disc_criterion(input_imgs, real_preds, fake_preds, target_imgs,
                                                    fake_images, self.discriminator, True)
                if gp is not None:
                    disc_loss += gp
                    gp_loss += gp.detach().cpu()

                val_disc_loss += disc_loss.detach().cpu()

                # calculate generator loss, gan loss + reconstruction loss (L1 or L2)
                gen_loss_gan, gen_loss_recon, class_loss = self.gen_criterion(fake_preds,
                                                                              target_imgs,
                                                                              fake_images,
                                                                              class_logits,
                                                                              class_labels)
                gen_loss = gen_loss_gan + gen_loss_recon + class_loss

                # update metric on batch
                if self.metrics is not None:
                    if class_logits is not None:
                        self._update_metrics(input_imgs.detach(), target_imgs.detach(),
                                             fake_images.detach(), class_logits.detach(),
                                             class_labels.detach())
                    else:
                        self._update_metrics(input_imgs.detach(), target_imgs.detach(),
                                             fake_images.detach(), None, None)

            if self.options['verbose']:
                if gp is not None:
                    display_string = "Gen loss: {:.4f}, Gen gan loss {:.4f}, Gen recon loss {:.4f}, Gen class loss {:.4f}, Disc loss: {:.4f}, Disc gan loss {:.4f}, GP :{:.4f}".format(
                        gen_loss, gen_loss_gan, gen_loss_recon, class_loss, disc_loss, disc_loss - gp,
                        gp)
                else:
                    display_string = "Gen loss: {:.4f}, Gen gan loss {:.4f}, Gen recon loss {:.4f}, Gen class loss {:.4f}, Disc loss: {:.4f}".format(
                        gen_loss, gen_loss_gan, gen_loss_recon, class_loss, disc_loss)
                val_iter.set_description(display_string)

            val_gen_loss += gen_loss.detach().cpu()
            val_gen_loss_gan += gen_loss_gan.detach().cpu()
            val_gen_loss_recon += gen_loss_recon.detach().cpu()
            val_gen_class_loss += class_loss.detach().cpu()

            if i == len(self.val_loader) - 1:
                self._plot_images(input_imgs, target_imgs, fake_images,
                                  split='val', epoch=self.history['elapsed_epochs'] + 1, step='end')

        val_gen_loss /= len(self.val_loader)
        val_gen_loss_gan /= len(self.val_loader)
        val_gen_loss_recon /= len(self.val_loader)
        val_gen_class_loss /= len(self.val_loader)
        val_disc_loss /= len(self.val_loader)
        gp_loss /= len(self.val_loader)
        self.history['val_gen_loss_tot'].append(val_gen_loss.item())
        self.history['val_gen_loss_gan'].append(val_gen_loss_gan.item())
        self.history['val_gen_loss_recon'].append(val_gen_loss_recon.item())
        self.history['val_gen_loss_class'].append(val_gen_class_loss.item())
        self.history['val_disc_loss_tot'].append(val_disc_loss.item())
        if isinstance(gp_loss, torch.Tensor):
            self.history.setdefault('val_gp_loss', []).append(gp_loss.item())
            self.history.setdefault('val_disc_loss_gan', []).append(val_disc_loss.item() - gp_loss.item())

    def train(self) -> dict:
        """
        Trains the generator and discriminator models
        :return: history, a dictionary containing the training and validation metrics
        """
        if self.history is None:
            self.history = {'train_gen_loss_tot': [], 'train_gen_loss_gan': [],
                            'train_gen_loss_recon': [], 'train_gen_loss_class': [],
                            'train_disc_loss_tot': [],
                            'val_gen_loss_tot': [], 'val_gen_loss_gan': [],
                            'val_gen_loss_recon': [], 'val_gen_loss_class': [],
                            'val_disc_loss_tot': [],
                            'elapsed_epochs': 0, 'epoch_times': []}

        current_epoch = self.history['elapsed_epochs']
        while current_epoch < self.options['num_epochs']:
            start_time = time.time()
            print("\n\n")
            print("Epoch {}/{}".format(current_epoch + 1, self.options['num_epochs']))

            # train one epoch
            print("Training...")
            self._train_epoch()

            # validate after one epoch
            print("Validation...")
            self._validate_epoch()
            # compute epoch metrics
            if self.metrics is not None:
                self._compute_metrics()

            # update lr schedulers
            if self.disc_scheduler is not None:
                self.disc_scheduler.step()
            if self.gen_scheduler is not None:
                self.gen_scheduler.step()

            end_time = time.time()
            self.history['epoch_times'].append(end_time - start_time)

            # print epoch summary
            if self.options['verbose']:
                self._print_summary(current_epoch)

            # update tensorboard
            if self.tb_writer is not None:
                self._update_tensorboard()

            # clear output every 10 epochs
            if (current_epoch + 1) % 10 == 0:
                clear_output(wait=True)

            # update epoch counter
            current_epoch += 1
            self.history['elapsed_epochs'] = current_epoch

            # save checkpoint
            if self.checkpoint is not None:
                self.checkpoint.save(self.history)

        return self.history


class ClipWeights(object):
    def __init__(self, clip_value=0.01):
        self.clip_value = clip_value

    def __call__(self, module):
        if hasattr(module, 'weight'):
            w = module.weight.data
            w = w.clamp(-self.clip_value, self.clip_value)
            module.weight.data = w
