
```shell
conda create -n lc25k-cleaning python=3.11
conda activate lc25k-cleaning

# pytorch - should be installed first
conda install pytorch torchvision pytorch-cuda=12.1 -c pytorch -c nvidia

# creating UNI; Prov-GigaPath models (Prov-GigaPath model is created with bugs when used with timm==0.9.8)
conda install -c conda-forge timm>1.0.0

# reading annotations
conda install -c conda-forge numpy pandas

# reading images
conda install -c conda-forge opencv imageio scikit-image

# machine learning
conda install -c conda-forge scikit-learn scipy statsmodels umap-learn

# progress bar and debugger
conda install -c conda-forge tqdm ipdb

# plotting
conda install -c conda-forge matplotlib seaborn

# jupyter notebooks
conda install -c conda-forge jupyter

# precision_at_1 checking agianst pytorch-metric-learning library
# conda install -c pytorch faiss-cpu=1.8.0
# conda install -c conda-forge pytorch-metric-learning

# tests and linter
conda install -c conda-forge pytest flake8
```
