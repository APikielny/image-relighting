# image-relighting

CS1430 Final Project: Image Relighting

# Team

Monsters.inc

# Project

## Overview
We will be implementing single image portrait relighting using the findings and architecture
of the paper, [Deep Single Image Portrait Relighting](https://zhhoper.github.io/dpr.html).
Using their Celeb-HQ dataset, we will train our own neural network and GAN to produced high quality
portrait images under novel lighting conditions. Once implemented we hope to extend the project
to allow for easy input for new lighting conditions for a given image.

## Resources for Network Design
* [Supplemental paper](https://zhhoper.github.io/paper/zhou_ICCV_2019_DPR_sup.pdf) describing a more
detailed architecture.
* LS-GAN [Github](https://zhhoper.github.io/paper/zhou_ICCV2019_D) [2].
* Patch GAN [Github](https://github.com/phillipi/pix2pix) and [paper](https://arxiv.org/pdf/1611.07004.pdf) [10].
* FFHQ images [GitHub](https://github.com/NVlabs/stylegan), [dataset](https://github.com/NVlabs/ffhq-dataset)
and [paper](https://arxiv.org/pdf/1812.04948.pdf) [13].