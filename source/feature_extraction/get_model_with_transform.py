import torch
from torch import nn

import torchvision
from torchvision.models import ResNet18_Weights

import timm
from transformers import AutoImageProcessor, AutoModel

# # need to be installed in the environment
# import uni  # https://github.com/mahmoodlab/UNI/?tab=readme-ov-file#installation

# local imports
from source.feature_extraction.data import get_data_transform
from source.feature_extraction.models.resnet_clam import resnet50_baseline
from source.feature_extraction.models.resnet_dsmil import get_resnet18_dsmil
from source.feature_extraction.models.owkin_phikon import OwkinPhikonFeatureExtractor
from source.constants import EXTRACTOR_NAMES_2_WEIGHTS_PATHS


extractor_2_original_transform = {
    'imagenet_resnet18-last-layer': 'imagenet',
    'imagenet_resnet50-clam-extractor': 'imagenet',
    'dinov2_vits14': 'imagenet',
    'dinov2_vitb14': 'imagenet',
    'UNI': 'imagenet',
    'prov-gigapath': 'imagenet',
    'owkin-phikon': 'imagenet',
    'simclr-tcga-lung_resnet18-2.5x': 'resize_only',
    'simclr-tcga-lung_resnet18-10x': 'resize_only',
}


def get_feature_extractor(extractor_name):
    """
    Get the feature extractor based on the given extractor name.

    Args:
        extractor_name (str): The name of the feature extractor. Must be one of the implemented models:
            - 'imagenet_resnet18-last-layer'
            - 'imagenet_resnet50-clam-extractor'
            - 'dinov2_vits14'
            - 'dinov2_vitb14'
            - 'UNI'
            - 'prov-gigapath'
            - 'owkin-phikon'
            - 'simclr-tcga-lung_resnet18-2.5x'
            - 'simclr-tcga-lung_resnet18-10x'
        device (str): The device to use for computation. Defaults to 'cpu'.

    Returns:
        Feature extractor.

    Raises:
        NotImplementedError: If the given extractor name is not implemented.
    """

    # -----------------------------------------------------------------------------------------------
    # Natural Image models
    # -----------------------------------------------------------------------------------------------
    # ResNet-18: last layer turned to Identity function
    if extractor_name == 'imagenet_resnet18-last-layer':
        feature_extractor = torchvision.models.resnet18(
            weights=ResNet18_Weights.IMAGENET1K_V1)
        feature_extractor.fc = nn.Identity()  # cut the classifier

    # ResNet-50: CLAM style (first 3 blocks only)
    elif extractor_name == 'imagenet_resnet50-clam-extractor':
        feature_extractor = resnet50_baseline(pretrained=True)

    # DINOv2: ViT-Small
    elif extractor_name == 'dinov2_vits14':
        feature_extractor = torch.hub.load(
            'facebookresearch/dinov2', 'dinov2_vits14')

    # DINOv2: ViT-Base
    elif extractor_name == 'dinov2_vitb14':
        feature_extractor = torch.hub.load(
            'facebookresearch/dinov2', 'dinov2_vitb14')

    # -----------------------------------------------------------------------------------------------
    # Pathology-specific models
    # -----------------------------------------------------------------------------------------------
    # UNI: https://github.com/mahmoodlab/UNI
    elif extractor_name == 'UNI':

        # # option 1: most manual
        # local_dir = "../weights/UNI/ckpts/vit_large_patch16_224.dinov2.uni_mass100k/"
        # os.makedirs(local_dir, exist_ok=True)  # create directory if it does not exist
        # hf_hub_download("MahmoodLab/UNI",
        #                 filename="pytorch_model.bin",
        #                 local_dir=local_dir,
        #                 # force_download=True,
        #                 )
        # feature_extractor = timm.create_model(
        #     "vit_large_patch16_224", img_size=224, patch_size=16, init_values=1e-5, num_classes=0, dynamic_img_size=True
        # )
        # feature_extractor.load_state_dict(torch.load(os.path.join(
        #     local_dir, "pytorch_model.bin"), map_location=device), strict=True)

        # option 2
        feature_extractor = timm.create_model(
            "hf-hub:MahmoodLab/UNI", pretrained=True, init_values=1e-5, dynamic_img_size=True)

        # # option 3 - requires installation of the UNI package, which installs old timm version
        # import uni
        # feature_extractor, _ = uni.get_encoder(
        #     enc_name='uni', device=device)

    # ProViT-GigaPath tile encoder: https://huggingface.co/prov-gigapath/prov-gigapath
    # does not work with timm==0.9.8, needs timm==1.0.3: https://github.com/prov-gigapath/prov-gigapath/issues/2
    elif extractor_name == 'prov-gigapath':
        assert timm.__version__ > '1.0.0', "There is a bug in version `timm==0.9.8`. Tested to work from version `timm==1.0.3`"
        feature_extractor = timm.create_model(
            "hf_hub:prov-gigapath/prov-gigapath", pretrained=True)

    elif extractor_name == 'owkin-phikon':
        original_transform = AutoImageProcessor.from_pretrained("owkin/phikon")
        feature_extractor = OwkinPhikonFeatureExtractor(version="v1")
    
    elif extractor_name == 'owkin-phikon-v2':
        original_transform = AutoImageProcessor.from_pretrained("owkin/phikon-v2")
        feature_extractor = OwkinPhikonFeatureExtractor(version="v2")

    elif extractor_name == "H-optimus-0":
        # suggested use (HuggingFace) with normalization: mean=(0.707223, 0.578729, 0.703617), std=(0.211883, 0.230117, 0.177517)
        feature_extractor = timm.create_model("hf-hub:bioptimus/H-optimus-0", pretrained=True, init_values=1e-5, dynamic_img_size=False)

    # ResNet18 trained with SimCLR on TCGA-Lung images (2.5x magnification): https://github.com/binli123/dsmil-wsi/issues/41
    elif extractor_name == 'simclr-tcga-lung_resnet18-2.5x':
        feature_extractor = get_resnet18_dsmil(
            weights_path=EXTRACTOR_NAMES_2_WEIGHTS_PATHS[extractor_name])

    # ResNet18 trained with SimCLR on TCGA-Lung images (10x magnification): https://github.com/binli123/dsmil-wsi/issues/41
    elif extractor_name == 'simclr-tcga-lung_resnet18-10x':
        feature_extractor = get_resnet18_dsmil(
            weights_path=EXTRACTOR_NAMES_2_WEIGHTS_PATHS[extractor_name])

    # -----------------------------------------------------------------------------------------------
    else:
        raise NotImplementedError(
            f"Extractor {extractor_name} not implemented.")

    return feature_extractor


def get_feature_extractor_with_default_transform(extractor_name):
    """
    Get the feature extractor and data transform based on the given extractor name.

    Args:
        extractor_name (str): The name of the feature extractor. Must be one of the implemented models:
            - 'imagenet_resnet18-last-layer'
            - 'imagenet_resnet50-clam-extractor'
            - 'dinov2_vits14'
            - 'dinov2_vitb14'
            - 'UNI'
            - 'prov-gigapath'
            - 'owkin-phikon'
            - 'simclr-tcga-lung_resnet18-2.5x'
            - 'simclr-tcga-lung_resnet18-10x'

    Returns:
        tuple: A tuple containing the feature extractor with corresponding default data transform.

    Raises:
        NotImplementedError: If the given extractor name is not implemented.
    """

    feature_extractor = get_feature_extractor(extractor_name)
    data_transform = get_data_transform(extractor_2_original_transform[extractor_name])
    return feature_extractor, data_transform
