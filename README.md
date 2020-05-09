# image-relighting

CS1430 Final Project: Image Relighting

# Team
Aalia Habib, Adam Pikielny, Jack Dermer, Ben Givertz

Monsters.inc

# Project

## Overview
Traditional methods for relighting faces requires knowledge of the subject's reflectance, lighting, and structure. We sought out to implement a deep learning algorithm to solve this task and relight portraits given only a single image as input. We implemented an Hourglass-shaped CNN from research by Zhou et al., [Deep Single Image Portrait Relighting](https://zhhoper.github.io/dpr.html), in order to relight portrait images. The model first separates the input image into facial and lighting features, from which a specialized lighting network predicts the direction of light. Then, the facial features are combined with the desired new lighting. Using a synthesized data set of portrait images under various artificial lighting conditions for training and ground truth, we were able to achieve realistic results, outputting images at a resolution of 256*256.

## Dataset
Due to computational and storage limitations, we used a scaled down version of the
DPR dataset created by the original paper. The images were scaled down to both
128x128 and 256x255. Both datasets are available for download on [Google Drive](https://drive.google.com/open?id=1v-8FebXQPk5YqlWYYDe7frwy9OkJ24yq).

# Usage
The dependencies for this project can be found in the `requirements.txt` file and
can be installed with `pip install -r requirements.txt`.

## Model Training
The `train.py` file can be found in the model directory along with files for data loading,
the loss function, and the model itself. The image folders from the dataset must be moved into the `data/train/` directory.

Train a new model `python train.py [-h] [--epochs EPOCHS] [--batch BATCH] [--lr LR] [--data DATA] [--model MODEL] [--verbose] [--debug]`

## Model Testing
The `test_network.py` file...

The `relight.py` file...

The `live_lighting_transfer.py` file...

The `gui.py` file...
- Use any image that contains faces for lighting reference--no cropping necessary
- For the image you would like to apply lighting to, please crop close to the face prior to input.

