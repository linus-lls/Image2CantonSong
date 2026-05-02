# Environment Installation
Multiple conda environments needed, because they have different dependencies.

## `yue_project_clean`
Main environment for streamlit app, and image-to-lyrics part. Install with:
```{bash}
# Paths relative to this README's directory
conda env create -f ./yue_project_clean/yue_project_clean.yml
conda activate yue_project_clean
pip install -r ./yue_project_clean/yue_project_clean-requirements.txt --extra-index-url https://download.pytorch.org/whl/cu121
conda deactivate
```

## `yue_official`
Environment for model lyrics-to-music. Please refer to official README for installation: https://github.com/multimodal-art-projection/YuE or $PROJECT_ROOT/YuE/README.md

## `clip-e`
Environment for image-to-mood CLIP model. This uses an old version of TensorFlow. This is required for calculating image-emotion similarity. A subprocess is called to activate this environment in the python scripts.
```{bash}
# Paths relative to this README's directory
conda env create -n clip-e python=3.11
conda activate clip-e
conda install -c nvidia/label/cuda-11.8.0 cuda-toolkit
conda install -c conda-forge cudnn=8.8 cuda-version=11.8

pip install -r ./clip-e/clip-e-requirements.txt

mkdir -p $CONDA_PREFIX/etc/conda/activate.d
mkdir -p $CONDA_PREFIX/etc/conda/deactivate.d
cp ./clip-e/activate.d/cuda_paths.sh $CONDA_PREFIX/etc/conda/activate.d/
cp ./clip-e/deactivate.d/cuda_paths.sh $CONDA_PREFIX/etc/conda/deactivate.d/
conda deactivate
```