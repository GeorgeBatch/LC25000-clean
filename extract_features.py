import argparse
import os
import json

import numpy as np

import torch

from torch import nn

from torch.utils.data import DataLoader

from tqdm.auto import tqdm

from source.constants import ALL_CANCER_TYPES
from source.constants import ALL_IMG_NORMS, ALL_EXTRACTOR_MODELS
from source.constants import DATA_DIR, FEATURE_VECTORS_SAVE_DIR, DATASET_SPECIFIC_NORMALIZATION_CONSTANTS_PATH
from source.feature_extraction.data import (
    FeatureExtractionDataset,
    get_data_transform,
    get_original_image_transform,
)
from source.feature_extraction.get_model_with_transform import get_feature_extractor


def prepare_directories(all_img_dir_path, all_features_save_dir, cancer_type, extractor_name, img_norm):
    current_img_dir = f"{all_img_dir_path}/{cancer_type}"
    assert os.path.isdir(current_img_dir), f"Directory not found: {current_img_dir}"
    current_features_save_dir = f"{all_features_save_dir}/{cancer_type}/{extractor_name}/{img_norm}"
    os.makedirs(current_features_save_dir, exist_ok=True)
    return current_img_dir, current_features_save_dir


def make_pytorch_dataset(img_dir, data_transform):
    return FeatureExtractionDataset(
        img_dir=img_dir,
        transform=data_transform,
        img_ext='jpeg',
        return_image_details=True,
    )


def make_pytorch_dataloader(dataset, batch_size):
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False
    )
    return dataloader


def calculate_dataset_mean_std(img_dir, batch_size, decimals=4):
    """
    Calculate the mean and standard deviation of a dataset. 2-pass method:
    1. Calculate the mean during the first pass
    2. Calculate the standard deviation using the computed mean during the second pass

    A 1-pass method is possible, it uses the calculation of running mean and variance.
    Check online: https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance

    Args:
        img_dir (str): The directory containing the dataset images.
        batch_size (int): The batch size for the data loader.
        decimals (int, optional): The number of decimal places to round the mean and standard deviation to. Defaults to 4.

    Returns:
        tuple: A tuple containing the mean and standard deviation of the dataset.

    """
    dataset = make_pytorch_dataset(
        img_dir=img_dir, data_transform=get_data_transform(img_norm='resize_only'))
    dataloader = make_pytorch_dataloader(
        dataset=dataset, batch_size=batch_size)

    # Initialize variables to store sum and sum of squares of pixel values
    mean = torch.zeros(3)
    squared_diff_sum = torch.zeros(3)
    n_pixels = 0

    # ------------------------------------------------------------------------------
    print("Calculating mean for the dataset...")
    for batch in tqdm(dataloader):
        images = batch['image']

        # Reshape images: (batch_size, channels, height, width) -> (batch_size, channels, num_pixels)
        images = images.view(images.size(0), images.size(1), -1)

        # Update number of pixels
        n_pixels += images.size(0) * images.size(2)

        # Sum of pixel values
        mean += images.sum([0, 2])

    # Calculate the mean
    mean /= n_pixels

    # ------------------------------------------------------------------------------
    print("Calculating variance for the dataset (needed the mean)...")
    for batch in tqdm(dataloader):
        images = batch['image']

        # Reshape images: (batch_size, channels, height, width) -> (batch_size, channels, num_pixels)
        images = images.view(images.size(0), images.size(1), -1)

        # Sum of squared differences from the mean
        squared_diff_sum += ((images - mean.unsqueeze(1)) ** 2).sum([0, 2])

    # Calculate the variance and standard deviation
    variance = squared_diff_sum / (n_pixels - 1)
    std = torch.sqrt(variance)

    mean_tuple = tuple([round(mean[i].item(), decimals)
                       for i in range(len(mean))])
    std_tuple = tuple([round(std[i].item(), decimals)
                      for i in range(len(std))])

    return mean_tuple, std_tuple


def update_dataset_specific_mean_std(json_path, mean, std, img_norm):
    # Load existing data
    with open(json_path, 'r') as f:
        normalisation_constants = json.load(f)
    # Update data
    normalisation_constants[img_norm] = {'mean': mean, 'std': std}
    # Save updated data
    with open(json_path, 'w') as f:
        json.dump(normalisation_constants, f)
    print("Saved new normalization constants. You need to manually add them to `source.feature_extraction.data.get_norm_constants`")


def prepare_feature_extractor(extractor_name, device):
    feature_extractor = get_feature_extractor(extractor_name)

    # eval mode
    feature_extractor.eval()
    # move to device
    feature_extractor = feature_extractor.to(device)

    # hardware
    if device == 'cpu':
        print("CPU mode")
    else:
        if torch.cuda.device_count() == 1:
            print("Single GPU mode")
        elif (torch.cuda.device_count() > 1):
            if device == 'cuda':
                print("Multiple GPU mode")
                feature_extractor = nn.DataParallel(feature_extractor)
            else:
                print("Single GPU mode")
        else:
            raise NotImplementedError

    return feature_extractor


def extract_features(feature_extractor, dataloader, device):
    print(f'Using device: {device}')

    # check how img_dir is made in prepare_directories()
    img_dir = dataloader.dataset.img_dir
    all_img_dir_name = img_dir.split('/')[-2]

    current_ids_list = []
    current_features_list = []
    current_paths_list = []
    for batch_idx, batch in enumerate(tqdm(dataloader)):
        # print(batch)

        inputs = batch['image'].to(device)
        if dataloader.dataset.return_image_details:
            details = {
                detail: batch[detail]
                for detail in batch
                if detail.startswith('image_')
            }
        # print("inputs.shape", inputs.shape)
        # print(details)

        ids = details['image_id'].numpy()
        # print(ids.shape)
        current_ids_list.append(ids)

        paths = details['image_path']  # list of strings
        short_paths = [f"./{all_img_dir_name}/" + path.split(f'/{all_img_dir_name}/')[1]
                       for path in paths]
        current_paths_list.extend(short_paths)

        with torch.no_grad():
            features = feature_extractor(inputs).cpu().numpy()
        # print(features.shape)
        current_features_list.append(features)

    current_features_numpy_array = np.concatenate(current_features_list, axis=0)
    current_ids_numpy_array = np.concatenate(current_ids_list, axis=0)
    current_ids_2_img_paths = {str(i): current_paths_list[i] for i in current_ids_numpy_array}
    return {
        'features': current_features_numpy_array,
        'ids': current_ids_numpy_array,
        'ids_2_img_paths': current_ids_2_img_paths
    }


def save_features(contents, paths):
    if not all([os.path.exists(paths[key]) for key in paths]):
        print("No files at these paths. Saving files...")
        np.save(paths['ids'], contents['ids'])
        np.save(paths['features'], contents['features'])

        with open(paths['ids_2_img_paths'], "w") as f:
            json.dump(contents['ids_2_img_paths'], f, indent=4)
        print("Files saved.")

    else:
        print("Files already exist.")

        print("Checking if the contents are the same...")
        ids_old = np.load(paths['ids'])
        features_old = np.load(paths['features'])
        with open(paths['ids_2_img_paths'], "r") as f:
            ids_2_img_paths_old = json.load(f)

        if np.allclose(ids_old, contents['ids'], atol=1e-6) \
            and np.allclose(features_old, contents['features'], atol=1e-6) \
                and (ids_2_img_paths_old == contents['ids_2_img_paths']):
            print("Contents are the same.")
        else:
            # ask for confirmation
            print(
                "Files already exist. Contents are not the same. Do you want to overwrite them?")
            print(f"\t ids_file_path: {paths['ids']}")
            print(f"\t ids_2_img_paths_file_path: {paths['ids_2_img_paths']}")
            print(f"\t features_file_path: {paths['features']}")
            print("Enter 'y' to confirm and overwrite.")
            user_input = input()
            print("User input: ", user_input)
            if user_input == 'y':
                np.save(paths['ids'], contents['ids'])
                np.save(paths['features'], contents['features'])

                with open(paths['ids_2_img_paths'], "w") as f:
                    json.dump(contents['ids_2_img_paths'], f, indent=4)
            else:
                print("Exiting without saving.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--cancer_type", type=str, default='lung_aca',
                        help="Cancer type name.",
                        choices=ALL_CANCER_TYPES)
    parser.add_argument("--img_norm", type=str, default='resize_only',
                        help="Image normalization type. 'original' means using image normalization constants recommended by the model authors.",
                        choices=ALL_IMG_NORMS + ["original"])
    parser.add_argument("--extractor_name", type=str, default='UNI',
                        help="Feature extractor name.",
                        choices=ALL_EXTRACTOR_MODELS)
    parser.add_argument("--device", type=str, default='cuda',
                        help="Device to use. 'cpu' or 'cuda' or 'cuda:<INDEX>'.")
    parser.add_argument("--batch_size", type=int, default=256,
                        help="Batch size for feature extraction.")
    args = parser.parse_args()

    device = args.device if torch.cuda.is_available() else 'cpu'

    img_dir, features_save_dir = prepare_directories(
        all_img_dir_path=DATA_DIR,
        all_features_save_dir=FEATURE_VECTORS_SAVE_DIR,
        cancer_type=args.cancer_type,
        img_norm=args.img_norm,
        extractor_name=args.extractor_name,
    )
    features_save_paths = {
        'ids': f'{features_save_dir}/ids.npy',
        'ids_2_img_paths': f'{features_save_dir}/ids_2_img_paths.json',
        'features': f'{features_save_dir}/features.npy'
    }

    if args.img_norm == 'original':
        # provided by the model authors
        data_transform = get_original_image_transform(extractor_name=args.extractor_name)
    else:
        # specific data transform
        try:
            data_transform = get_data_transform(img_norm=args.img_norm)
        except KeyError as e:
            print(f"Key {e} not found in either constansts_zoo of `data.get_norm_constants()` or data-specific transforms in {DATASET_SPECIFIC_NORMALIZATION_CONSTANTS_PATH}")
            assert args.cancer_type in args.img_norm, f"Dataset-specific img_norm={args.img_norm} should include cancer_type={args.cancer_type} in its name."
            mean, std = calculate_dataset_mean_std(img_dir=img_dir, batch_size=args.batch_size)
            data_transform = get_data_transform(img_norm='manual', mean=mean, std=std)
            update_dataset_specific_mean_std(json_path=DATASET_SPECIFIC_NORMALIZATION_CONSTANTS_PATH, mean=mean, std=std, img_norm=args.img_norm)

    feature_extractor = prepare_feature_extractor(
        extractor_name=args.extractor_name,
        device=args.device
    )

    dataset = make_pytorch_dataset(
        img_dir=img_dir, data_transform=data_transform)
    dataloader = make_pytorch_dataloader(
        dataset=dataset, batch_size=args.batch_size)

    features_and_info = extract_features(
        feature_extractor=feature_extractor,
        dataloader=dataloader,
        device=device,
    )
    save_features(contents=features_and_info, paths=features_save_paths)
