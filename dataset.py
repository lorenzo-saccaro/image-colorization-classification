# create Dataset class to load coco dataset
import os.path
import pandas as pd
from torch.utils.data import Dataset
from PIL import Image
from torchvision.transforms import Compose
from kornia.color import rgb_to_grayscale, rgb_to_lab


class Places205Dataset(Dataset):

    def __init__(self, dataset_folder: str, dataset_info_file: str, dataset_split: str,
                 transforms: Compose | None = None, use_lab_colorspace: bool = False):
        """
        :param dataset_folder: path to the dataset root folder
        :param dataset_info_file: name of the csv file containing the dataset information
        :param dataset_split: train, val or test split
        :param transforms: transforms to apply to the input image
        :param use_lab_colorspace: whether to use LAB colorspace (default is True)
        """
        assert dataset_split in ['train', 'val',
                                 'test'], 'Invalid dataset split: ' + dataset_split + '. Must be train, val, or test.'

        # load csv file with the dataset information
        self.img_df = pd.read_csv(os.path.join(dataset_folder, dataset_info_file))

        # keep only image paths of the corresponding split
        self.img_df = self.img_df[self.img_df['split'] == dataset_split]

        self.path = dataset_folder
        self.transforms = transforms
        self.use_lab_colorspace = use_lab_colorspace

    def __len__(self):
        return len(self.img_df)

    def __getitem__(self, idx: int):
        # load image
        img = Image.open(os.path.join(self.path, self.img_df.iloc[idx]['file_name']))

        # convert to RGB if necessary
        if img.mode != 'RGB':
            img = img.convert('RGB')

        # apply transforms
        if self.transforms is not None:
            img = self.transforms(img)

        label = self.img_df.iloc[idx]['class']

        # convert and scale input to [-1, 1]
        if self.use_lab_colorspace:
            img = rgb_to_lab(img)
            L = (img[[0], :, :] / 100 - 0.5) / 0.5
            ab = img[[1, 2], :, :]
            ab[ab > 0] /= 127
            ab[ab < 0] /= 128
            return L, ab, label

        else:
            gray, color = rgb_to_grayscale(img), img
            gray = (gray - 0.5) / 0.5
            color = (color - 0.5) / 0.5
            return gray, color, label


if __name__ == '__main__':
    # test dataset
    from torchvision import transforms
    from torch.utils.data import DataLoader

    # compose resize and to tensor transforms
    transforms = transforms.Compose([transforms.ToTensor(), transforms.Resize((256, 256))])

    dataset = Places205Dataset(dataset_folder='C:\\Users\\loren\\Datasets\\Places205',
                               dataset_info_file='files_split.csv', dataset_split='val',
                               transforms=transforms, use_lab_colorspace=True)

    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

    for batch_x, batch_y, label in dataloader:
        print(batch_x.shape)
        print(batch_x[0])
        print(batch_y.shape)
        print(batch_y[0])
        print(label.shape)
        print(label[0])
        break
