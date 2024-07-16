################################################################################
# Imports
################################################################################

import os
import json

import torch
from torch.utils.data import Dataset
from torchvision.io import read_image
from torchvision.transforms import v2

from source.constants import ALL_IMG_NORMS, DATASET_SPECIFIC_NORMALIZATION_CONSTANTS_PATH

################################################################################
# Transforms
################################################################################

STANDARD_INPUT_SIZE = 224


def get_norm_constants(img_norm: str = 'imagenet'):
    # Source: https://github.com/mahmoodlab/UNI/blob/main/uni/get_encoder/get_encoder.py
    constants_zoo = {
        'imagenet': {'mean': (0.485, 0.456, 0.406), 'std': (0.229, 0.224, 0.225)},
        'openai_clip': {'mean': (0.48145466, 0.4578275, 0.40821073), 'std': (0.26862954, 0.26130258, 0.27577711)},
        'uniform': {'mean': (0.5, 0.5, 0.5), 'std': (0.5, 0.5, 0.5)},
    }
    try:
        constants = constants_zoo[img_norm]
    except KeyError as e:
        print(f"Key {e} not found in constants_zoo of `data.get_norm_constants()`. Trying to load from dataset-specific constants.")
        with open(DATASET_SPECIFIC_NORMALIZATION_CONSTANTS_PATH, 'r') as f:
            dataset_specific_constants = json.load(f)
            constants = dataset_specific_constants[img_norm]
        print(f"Succesfully loaded constants for {img_norm} from dataset-specific constants.")
    return constants.get('mean'), constants.get('std')


def get_data_transform(img_norm: str = 'imagenet', mean=None, std=None):
    """
    Returns a torchvision transform for preprocessing input data.

    Args:
        img_norm (str): The type of image normalization to apply. Defaults to 'imagenet'.

    Returns:
        torchvision.transforms.Compose: A composition of image transformations.

    Raises:
        AssertionError: If an invalid normalization type is provided.

    """
    if img_norm == 'resize_only':
        transform = v2.Compose([
            # v2.ToImage(),  # Convert to tensor, only needed if you had a PIL image
            v2.Resize(STANDARD_INPUT_SIZE),
            # Normalize expects float input
            v2.ToDtype(torch.float32, scale=True),
        ])
    elif img_norm == 'manual':
        # used when mean and std are provided as arguments
        assert mean is not None and std is not None, "Mean and std must be provided for dataset-specific normalization."
        transform = v2.Compose([
            # v2.ToImage(),  # Convert to tensor, only needed if you had a PIL image
            v2.Resize(STANDARD_INPUT_SIZE),
            v2.ToDtype(torch.float32, scale=True),  # Normalize expects float input
            v2.Normalize(mean=mean, std=std),
        ])
    else:
        assert img_norm in ALL_IMG_NORMS, f"Invalid normalization type: {img_norm}. Should be one of {ALL_IMG_NORMS}."
        mean, std = get_norm_constants(img_norm)
        transform = v2.Compose([
            # v2.ToImage(),  # Convert to tensor, only needed if you had a PIL image
            v2.Resize(STANDARD_INPUT_SIZE),
            v2.ToDtype(torch.float32, scale=True),  # Normalize expects float input
            v2.Normalize(mean=mean, std=std),
        ])

    return transform

################################################################################
# Dataset Classes
################################################################################


class FeatureExtractionDataset(Dataset):
    """
    A custom dataset class for feature extraction.

    Args:
        img_dir (str): The directory path where the images are stored.
        img_ext (str): The file extension of the images.
        transform (callable, optional): A function/transform to apply to the images. Default is None.
        return_image_details (bool, optional): Whether to return additional image details. Default is False.

    Attributes:
        img_dir (str): The directory path where the images are stored.
        return_image_details (bool): Whether to return additional image details.
        transform (callable): A function/transform to apply to the images.
        img_file_name_list (list): A sorted list of image file names in the directory.

    Methods:
        __len__(): Returns the length of the dataset.
        __getitem__(idx): Returns the sample at the given index.

    """

    def __init__(self,
                 img_dir: str,
                 img_ext: str,
                 transform=None,
                 return_image_details: bool = False):

        # image details
        self.img_dir = img_dir
        self.return_image_details = return_image_details

        # image transforms
        self.transform = transform

        self.img_file_name_list = sorted([file_name
                                          for file_name in os.listdir(img_dir)
                                          if file_name.endswith(img_ext)])

    def __len__(self):
        """
        Returns the length of the dataset.

        Returns:
            int: The number of samples in the dataset.

        """
        return len(self.img_file_name_list)

    def __getitem__(self, idx):
        """
        Returns the sample at the given index.

        Args:
            idx (int): The index of the sample to retrieve.

        Returns:
            dict: A dictionary containing the image and optionally additional image details.

        """
        img_id = idx

        img_file_name = self.img_file_name_list[idx]
        img_name = img_file_name.split('.')[0]
        img_path = os.path.join(f"{self.img_dir}/{img_file_name}")

        # faster than PIL for JPEG images - uses libjpeg-turbo
        image = read_image(img_path)

        if self.transform:
            image = self.transform(image)

        if not self.return_image_details:
            sample = {"image": image}
        else:
            sample = {"image": image,
                      "image_id": img_id, "image_name": img_name, "image_path": img_path}
        return sample
