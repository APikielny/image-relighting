# Portrait Image Relighting

Brown CS1430 Final Project: Portrait Image Relighting

# Team
Aalia Habib, Adam Pikielny, Jack Dermer, Ben Givertz

Monsters.inc

# Project

![Obama image](https://github.com/APikielny/image-relighting/blob/master/README/obamaResults.png)
![Face to face relighting](https://github.com/APikielny/image-relighting/blob/master/README/adamaaliamix%20crop.png)

## Overview
Traditional methods for relighting faces require knowledge of the subject's reflectance, lighting, and structure. We sought out to implement a deep learning algorithm to solve this task and relight portraits given only a single image as input. We implemented an Hourglass-shaped CNN from research by Zhou et al., [Deep Single Image Portrait Relighting](https://zhhoper.github.io/dpr.html), in order to relight portrait images. The model first separates the input image into facial and lighting features, from which a specialized lighting network predicts the direction of light. Then, the facial features are combined with the desired new lighting. Using a synthesized data set of portrait images under various artificial lighting conditions for training and ground truth, we were able to achieve realistic results, outputting images at a resolution of 256*256.

## Dataset
Due to computational and storage limitations, we used a scaled down version of the
DPR dataset created by the original paper. The images were scaled down to both
128x128 and 256x256. Both datasets are available for download on [Google Drive](https://drive.google.com/open?id=1v-8FebXQPk5YqlWYYDe7frwy9OkJ24yq).

# Usage
The dependencies for this project can be found in the `requirements.txt` file and
can be installed with `pip install -r requirements.txt`.

## Model Training
The `train.py` file can be found in the model directory along with files for data loading,
the loss function, and the model itself. To train the model, the image folders from the dataset must be moved into the `data/train/` directory.

Train a new model using:
- `python train.py [-h] [--epochs EPOCHS] [--batch BATCH] [--lr learning_rate] [--data DATA] [--model MODEL] [--verbose] [--debug]`

## Model Testing
There are multiple ways to test our model, detailed below. For each test, we allow specification of input images, the model to use, and whether or not to use a GPU.

The image(s) should be stored in the folder `data/test/images/`. The model should be stored in `trained_models/`. Use the `--gpu` flag if you'd like to run on a CUDA GPU (such as on Google Cloud Platform). 

1. To **relight a face from several angles**, use `test_network.py`. The `test_network.py` file can be run using: 
- `python test_network.py [-h] [--image IMAGE)] [--model MODEL] [--gpu]`

2. To **relight based on lighting from another face**, use:
- `python relight.py [-h] [--source_image SOURCE_IMAGE] [--light_image LIGHT_IMAGE] [--model MODEL] [--gpu] [--face_detect FACE_DETECT]`

- The `[--face_detect]` flag can be passed "both" or "light". "Light" will only run face detection on the lighting input, which is recommended. Running "both" will crop both faces, so the output face will also be cropped.

3. The `live_lighting_transfer.py` file can be run to **see a live webcam view with dynamic relighting**:

- `python live_lighting_transfer.py [-h] [--light_image LIGHT_IMAGE] [--light_text LIGHT_TEXT]`

- `[--light_text]` is the target lighting as an array. 

4. For a more **user friendly** approach, the `gui.py` file can be run using `python gui.py` in the `GUI` folder.

- Use any image that contains faces for lighting reference--no cropping necessary
- For the image you would like to apply lighting to, please crop close to the face prior to input.

