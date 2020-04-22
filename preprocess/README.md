# Data Portrait Religihting (DPR) dataset
## Input Images
1. Get input images from [CelebA-HQ dataset](https://arxiv.org/pdf/1710.10196.pdf).
    - [GitHub](https://github.com/tkarras/progressive_growing_of_gans)
2. Run through a [landmark detector](https://www.cv-foundation.org/openaccess/content_cvpr_2014/papers/Kazemi_One_Millisecond_Face_2014_CVPR_paper.pdf) and discard images for which no landmarks are detectd.
    - Unofficial [GitHub](https://github.com/JiaShun-Xiao/face-alignment-ert-2D)
3. Randomly select 5 lighting conditions from a [lighting dataset](https://www.researchgate.net/publication/322841593_Occlusion-Aware_3D_Morphable_Models_and_an_Illumination_Prior_for_Face_Image_Analysis).

## Processing

