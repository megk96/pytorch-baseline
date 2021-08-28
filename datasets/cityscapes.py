import os
import numpy as np
import torch
from PIL import Image
from torch.utils import data


num_classes = 19
ignore_label = 255
root = '/content/gdrive/My Drive/deep-metric-learning/datasets/cityscapes'


# The color encoding table for Cityscapes dataset referenced from the code at
# https://github.com/mcordts/cityscapesScripts/blob/master/cityscapesscripts/helpers/labels.py
palette = [
    128, 64, 128,
    244, 35, 232,
    70, 70, 70,
    102, 102, 156,
    190, 153, 153,
    153, 153, 153,
    250, 170, 30,
    220, 220, 0,
    107, 142, 35,
    152, 251, 152,
    0, 130, 180,
    220, 20, 60,
    255, 0, 0,
    0, 0, 142,
    0, 0, 70,
    0, 60, 100,
    0, 80, 100,
    0, 0, 230,
    119, 11, 32,
]
zero_pad = 256 * 3 - len(palette)
for i in range(zero_pad):
    palette.append(0)


def colorize_mask(mask):
    # mask: numpy array of the mask
    new_mask = Image.fromarray(mask.astype(np.uint8)).convert('P')
    new_mask.putpalette(palette)

    return new_mask


def make_dataset(quality, mode):
    # Sanity check for argument values
    assert (quality == 'fine' and mode in ['train', 'val']) or \
           (quality == 'coarse' and mode in ['train', 'train_extra', 'val'])
    # Cityscapes dataset has two markings - fine and coarse
    # For the purpose of this project, the fine annotations are used.
    # Respective paths and nomenclature of files is set for the selected mode
    if quality == 'coarse':
        img_dir_name = 'leftImg8bit_trainextra' if mode == 'train_extra' else 'leftImg8bit_trainvaltest'
        mask_path = os.path.join(root, 'gtCoarse', 'gtCoarse', mode)
        mask_postfix = '_gtCoarse_labelIds.png'
    else:
        img_dir_name = 'leftImg8bit_trainvaltest'
        mask_path = os.path.join(root, 'gtFine_trainvaltest', 'gtFine', mode)
        mask_postfix = '_gtFine_labelIds.png'
    img_path = os.path.join(root, img_dir_name, 'leftImg8bit', mode)
    #assert os.listdir(img_path) == os.listdir(mask_path)
    items = []
    categories = os.listdir(img_path)
    for c in categories:
        if c == 'ulm' or c == 'munster':
            c_items = [name.split('_leftImg8bit.png')[0] for name in os.listdir(os.path.join(img_path, c))]
            for it in c_items:
                # Creating the corresponding (rgb image, mask) set for filenames
                item = (os.path.join(img_path, c, it + '_leftImg8bit.png'), os.path.join(mask_path, c, it + mask_postfix))
                items.append(item)
    return items





class CityScapes(data.Dataset):
    def __init__(self, quality, mode, joint_transform=None, sliding_crop=None, transform=None, target_transform=None):
        # self.imgs contains (rgb image, mask) filenames
        self.imgs = make_dataset(quality, mode)

        # Store all the initial
        self.quality = quality
        self.mode = mode
        self.joint_transform = joint_transform
        self.sliding_crop = sliding_crop
        self.transform = transform
        self.target_transform = target_transform
        # Among the 33 class labels of Cityscapes, 19 are selected and the rest of the labels are set to a default
        # value to ignore
        # The rest of the labels are transformed to represent a contiguous value
        self.id_to_trainid = {-1: ignore_label, 0: ignore_label, 1: ignore_label, 2: ignore_label,
                              3: ignore_label, 4: ignore_label, 5: ignore_label, 6: ignore_label,
                              7: 0, 8: 1, 9: ignore_label, 10: ignore_label, 11: 2, 12: 3, 13: 4,
                              14: ignore_label, 15: ignore_label, 16: ignore_label, 17: 5,
                              18: ignore_label, 19: 6, 20: 7, 21: 8, 22: 9, 23: 10, 24: 11, 25: 12, 26: 13, 27: 14,
                              28: 15, 29: ignore_label, 30: ignore_label, 31: 16, 32: 17, 33: 18}
        self.pixels = self.get_pixels()

    def __getitem__(self, index):
        img, mask = self.pixels[index]
        return img, mask



    def __len__(self):
        return len(self.imgs)

    def get_pixels(self):
        pixels = []
        for image in self.imgs:
            img_path, mask_path = image
            img, mask = Image.open(img_path).convert('RGB'), Image.open(mask_path)

            # Convert mask values to the converted values with the reduced classes
            mask = np.array(mask)
            mask_copy = mask.copy()
            for k, v in self.id_to_trainid.items():
                mask_copy[mask == k] = v
            mask = Image.fromarray(mask_copy.astype(np.uint8))

            # Apply all the necessary transformation functions
            if self.joint_transform is not None:
                img, mask = self.joint_transform(img, mask)
            if self.sliding_crop is not None:
                img_slices, mask_slices, slices_info = self.sliding_crop(img, mask)
                if self.transform is not None:
                    img_slices = [self.transform(e) for e in img_slices]
                if self.target_transform is not None:
                    mask_slices = [self.target_transform(e) for e in mask_slices]
                img, mask = torch.stack(img_slices, 0), torch.stack(mask_slices, 0)
            else:
                if self.transform is not None:
                    img = self.transform(img)
                if self.target_transform is not None:
                    mask = self.target_transform(mask)
            print(img.shape)
            print(mask.shape)
            for i in range(img.shape[1]):
                for j in range(img.shape[2]):
                    pixel_map = (img[:, i, j], mask[i, j])
                    print(pixel_map[0].shape)
                    print(pixel_map[1].shape)
                    pixels.append(pixel_map)

        return pixels





