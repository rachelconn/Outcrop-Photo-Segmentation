# Outcrop Photo Segmentation
Note: This repo is based off of EMANet by Li et al.
The original repo can be found here:
https://github.com/XiaLiPKU/EMANet

## Installation
Steps are adapted from EMANet, but updated to be more descriptive and support the outcrop dataset.
1. (Recommended): Create a python virtual environment for this project by running `python -m venv env`
    - Make sure to activate the environment before proceeding to the next steps
2. Run `pip install -r requirements.txt` to install prerequisite packages
3. Install pytorch with CUDA support. This project is tested with version 1.8.2, but the latest release should work. Directions can be found [here](https://pytorch.org/get-started/locally/).
4. Download the pretrained [ResNet50](https://drive.google.com/file/d/1ibhxxzrc-DpoHbHv4tYrqRPC1Ui7RZ-0/view?usp=sharing) and [ResNet101](https://drive.google.com/file/d/1de2AyWSTHsZQRB_MI-VcOfeP8NAs3Wat/view?usp=sharing), unzip them, and put into the 'models' folder from the root directory of this project (you will have to create it).
5. **IMPORTANT:** Follow the steps in one of the sections below corresponding to the dataset you want to use.
6. Run `sh clean.sh` to clear the models and logs from the last experiment.
7. Run `python train.py` for training and `sh tensorboard.sh` for visualization on your browser.
8. Or you can download the [pretraind model](https://drive.google.com/file/d/11GbUBfpWnt000Hy6FI32tppHc7QxczPO/view?usp=sharing), put into the 'models' folder, and skip step 6.
9. Run `python eval.py` for validation

## Running with PASCAL VOC dataset
1. Download [images](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar) and [labels](https://drive.google.com/file/d/1OqX6s07rFqtu-JZCjdjnJDv1QfDz9uG7/view?usp=sharing) of PASCAL VOC and SBD, extracting them to the same folder.
2. Change the following settings in `settings.py`:
  - `DATA_ROOT` to the location where you placed the dataset
  - `N_CLASSES` to 21
3. Swap the files located in the `datalist` folder with the ones in the `pascal` folder to fetch the correct files when training and testing.

## Running with the outcrop dataset
1. Download the dataset (note: this dataset is currently private, so no links are provided).
2. Change the following settings in `settings.py`:
  - `DATA_ROOT` to the location where you placed the dataset
  - `N_CLASSES` to 6
3. Swap the files located in the `datalist` folder with the ones in the `outcrop` folder to fetch the correct files when training and testing.
