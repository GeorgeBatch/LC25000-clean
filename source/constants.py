from os import path as osp

PROJECT_PATH = osp.abspath(
    osp.join(osp.dirname(osp.realpath(__file__)), '../'))

RANDOM_SEED = 42

DATA_DIR_FOLDER_NAME = 'LC25000'
DATA_DIR = osp.join(PROJECT_PATH, DATA_DIR_FOLDER_NAME)

DATASET_SPECIFIC_NORMALIZATION_CONSTANTS_PATH = osp.join(PROJECT_PATH, 'source/feature_extraction/img_normalisation_constants.json')

FEATURE_VECTORS_SAVE_DIR = osp.join(PROJECT_PATH, 'feature_vectors')
ANNOTATIONS_SAVE_DIR = osp.join(PROJECT_PATH, 'annotations')


NUM_TOTAL_IMAGES = 25000
NUM_CLASS_IMAGES = 5000
NUM_CLASSES = 5
assert NUM_TOTAL_IMAGES == NUM_CLASS_IMAGES * NUM_CLASSES, 'NUM_TOTAL_IMAGES should be equal to NUM_CLASS_IMAGES * NUM_CLASSES'

NUM_CLASS_PROTOTYPES = 250  # every class has 250 prototypes, 1250 prototypes in total (5 classes * 250 prototypes per class)
AVG_NUM_DUPLICATES_PER_CLASS_PROTOTYPE = 20
assert NUM_CLASS_IMAGES == NUM_CLASS_PROTOTYPES * AVG_NUM_DUPLICATES_PER_CLASS_PROTOTYPE, 'NUM_CLASS_IMAGES should be equal to NUM_CLASS_PROTOTYPES * AVG_NUM_DUPLICATES_PER_CLASS_PROTOTYPE'


ALL_TISSUE_TYPES = (
    'lung',
    'colon',
)

ALL_CANCER_TYPES = (
    'colon_aca', 'colon_n',
    'lung_aca', 'lung_n', 'lung_scc',
)
assert len(ALL_CANCER_TYPES) == NUM_CLASSES, 'Number of cancer types should match NUM_CLASSES'

ALL_IMG_NORMS = (
    "resize_only",
    # risize to STANDARD_INPUT_SIZE + normalize using mean and std
    "imagenet",
    "openai_clip",
    "uniform",
    "H-optimus-0",
    "hibou",
    # dataset-specific
    "lc25k-lung_aca-resized",
    "lc25k-lung_scc-resized",
    "lc25k-lung_n-resized",
    "lc25k-colon_aca-resized",
    "lc25k-colon_n-resized",
)

ALL_EXTRACTOR_MODELS = (
    # Natural Images
    'imagenet_resnet18-last-layer',
    'imagenet_resnet50-clam-extractor',
    'dinov2_vits14',  # ViT sizes: small -> base -> large -> giant
    'dinov2_vitb14',
    # Pathology Images
    'UNI',
    'prov-gigapath',
    'owkin-phikon',
    'owkin-phikon-v2',
    'hibou-b',
    'hibou-L',
    'Virchow',
    'Virchow2',
    'H-optimus-0',
    'simclr-tcga-lung_resnet18-10x',
    'simclr-tcga-lung_resnet18-2.5x',
)

FEATURE_EXTRACTOR_2_ORIGINAL_TRANSFORM = {
    # risize to STANDARD_INPUT_SIZE + normalize using ImageNet mean and std
    "imagenet_resnet18-last-layer": "imagenet",
    "imagenet_resnet50-clam-extractor": "imagenet",
    "dinov2_vits14": "imagenet",
    "dinov2_vitb14": "imagenet",
    "UNI": "imagenet",
    "prov-gigapath": "imagenet",
    "owkin-phikon": "imagenet",
    "owkin-phikon-v2": "imagenet",
    "Virchow": "imagenet",
    "Virchow2": "imagenet",
    # risize to STANDARD_INPUT_SIZE + normalize using mean and std constants provided by the authors
    "hibou-b": "hibou",
    "hibou-L": "hibou",
    "H-optimus-0": "H-optimus-0",
    # resize only
    "simclr-tcga-lung_resnet18-2.5x": "resize_only",
    "simclr-tcga-lung_resnet18-10x": "resize_only",
}
assert set(FEATURE_EXTRACTOR_2_ORIGINAL_TRANSFORM.keys()) == set(
    ALL_EXTRACTOR_MODELS
), f"FEATURE_EXTRACTOR_2_ORIGINAL_TRANSFORM keys should match ALL_EXTRACTOR_MODELS. Difference: {set(FEATURE_EXTRACTOR_2_ORIGINAL_TRANSFORM.keys()) ^ set(ALL_EXTRACTOR_MODELS)}"

ORIGINAL_2_PRETTY_MODEL_NAMES = {
    'UNI': 'UNI',
    'prov-gigapath': 'Prov-GigaPath',
    'owkin-phikon': 'Phikon',
    'owkin-phikon-v2': 'Phikon-v2',
    'dinov2_vits14': 'DINOv2-ViT-S/14',
    'dinov2_vitb14': 'DINOv2-ViT-B/14',
    'simclr-tcga-lung_resnet18-10x': 'ResNet18-lung-10x',
    'simclr-tcga-lung_resnet18-2.5x': 'ResNet18-lung-2.5x',
    'imagenet_resnet50-clam-extractor': 'ResNet50-CLAM',
    'imagenet_resnet18-last-layer': 'ResNet18',
}

EXTRACTOR_NAMES_2_WEIGHTS_PATHS = {
    'simclr-tcga-lung_resnet18-10x': osp.join(
        PROJECT_PATH, 'weights/simclr-tcga-lung/weights-10x/model-v1.pth'),
    'simclr-tcga-lung_resnet18-2.5x': osp.join(
        PROJECT_PATH, 'weights/simclr-tcga-lung/weights-2.5x/model-v1.pth'),
}

ALL_DIMENSIONALITY_REDUCTION_METHODS = [
    'NoReduction',
    # PCA: Proportion of variance explained
    'PCA-0.9',
    'PCA-0.95',
    'PCA-0.99',
    # UMAP: accepts n_components values between 2 and 100
    'UMAP-2',
    'UMAP-8',
    'UMAP-32',
]

ALL_DISTANCE_METRICS = (
    'euclidean',
    'cosine',
)

ALL_CLUSTERING_ALGORITHMS = (
    'kmeans',
    'agglomerative-single',
    'agglomerative-average',
    'agglomerative-complete',
)
