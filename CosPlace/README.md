
# Rethinking Visual Geo-localization for Large-Scale Applications

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/rethinking-visual-geo-localization-for-large/visual-place-recognition-on-pittsburgh-250k)](https://paperswithcode.com/sota/visual-place-recognition-on-pittsburgh-250k?p=rethinking-visual-geo-localization-for-large)[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/rethinking-visual-geo-localization-for-large/visual-place-recognition-on-pittsburgh-30k)](https://paperswithcode.com/sota/visual-place-recognition-on-pittsburgh-30k?p=rethinking-visual-geo-localization-for-large)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/rethinking-visual-geo-localization-for-large/visual-place-recognition-on-tokyo247)](https://paperswithcode.com/sota/visual-place-recognition-on-tokyo247?p=rethinking-visual-geo-localization-for-large)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/rethinking-visual-geo-localization-for-large/visual-place-recognition-on-mapillary-val)](https://paperswithcode.com/sota/visual-place-recognition-on-mapillary-val?p=rethinking-visual-geo-localization-for-large)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/rethinking-visual-geo-localization-for-large/visual-place-recognition-on-st-lucia)](https://paperswithcode.com/sota/visual-place-recognition-on-st-lucia?p=rethinking-visual-geo-localization-for-large)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/rethinking-visual-geo-localization-for-large/visual-place-recognition-on-sf-xl-test-v1)](https://paperswithcode.com/sota/visual-place-recognition-on-sf-xl-test-v1?p=rethinking-visual-geo-localization-for-large)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/rethinking-visual-geo-localization-for-large/visual-place-recognition-on-sf-xl-test-v2)](https://paperswithcode.com/sota/visual-place-recognition-on-sf-xl-test-v2?p=rethinking-visual-geo-localization-for-large)

This is the official pyTorch implementation of the CVPR 2022 paper "Rethinking Visual Geo-localization for Large-Scale Applications".
The paper presents a new dataset called San Francisco eXtra Large (SF-XL, go [_here_](https://forms.gle/wpyDzhDyoWLQygAT9) to download it), and a highly scalable training method (called CosPlace), which allows to reach SOTA results with compact descriptors.


[[CVPR OpenAccess](https://openaccess.thecvf.com/content/CVPR2022/html/Berton_Rethinking_Visual_Geo-Localization_for_Large-Scale_Applications_CVPR_2022_paper.html)] [[ArXiv](https://arxiv.org/abs/2204.02287)] [[Video](https://www.youtube.com/watch?v=oDyL6oVNN3I)] [[BibTex](https://github.com/gmberton/CosPlace#cite)]



The images below represent respectively:
1) the map of San Francisco eXtra Large
2) a visualization of how CosPlace Groups (read datasets) are formed
3) results with CosPlace vs other methods on Pitts250k (CosPlace trained on SF-XL, others on Pitts30k)
<p float="left">
  <img src="https://github.com/gmberton/gmberton.github.io/blob/main/images/SF-XL%20map.jpg" height="150" />
  <img src="https://github.com/gmberton/gmberton.github.io/blob/main/images/map_groups.png" height="150" /> 
  <img src="https://github.com/gmberton/gmberton.github.io/blob/main/images/backbones_pitts250k_main.png" height="150" />
</p>



## Train
After downloading the SF-XL dataset, simply run 

`$ python3 train.py --dataset_folder path/to/sf-xl/processed`

the script automatically splits SF-XL in CosPlace Groups, and saves the resulting object in the folder `cache`.
By default training is performed with a ResNet-18 with descriptors dimensionality 512 is used, which fits in less than 4GB of VRAM.

To change the backbone or the output descriptors dimensionality simply run 

`$ python3 train.py --dataset_folder path/to/sf-xl/processed --backbone ResNet50 --fc_output_dim 128`

You can also speed up your training with Automatic Mixed Precision (note that all results/statistics from the paper did not use AMP)

`$ python3 train.py --dataset_folder path/to/sf-xl/processed --use_amp16`

Run `$ python3 train.py -h` to have a look at all the hyperparameters that you can change. You will find all hyperparameters mentioned in the paper.

#### Dataset size and lightweight version

The SF-XL dataset is about 1 TB.
For training only a subset of the images is used, and you can use this subset for training, which is only 360 GB.
If this is still too heavy for you (e.g. if you're using Colab), but you would like to run CosPlace, we also created a small version of SF-XL, which is only 5 GB.
Obviously, using the small version will lead to lower results, and it should be used only for debugging / exploration purposes.
More information on the dataset and lightweight version are on the README that you can find on the dataset download page (go [_here_](https://forms.gle/wpyDzhDyoWLQygAT9) to find it).

#### Reproducibility
Results from the paper are fully reproducible, and we followed deep learning's best practices (average over multiple runs for the main results, validation/early stopping and hyperparameter search on the val set).
If you are a researcher comparing your work against ours, please make sure to follow these best practices and avoid picking the best model on the test set.


## Test
You can test a trained model as such

`$ python3 eval.py --dataset_folder path/to/sf-xl/processed --backbone ResNet50 --fc_output_dim 128 --resume_model path/to/best_model.pth`

You can download plenty of trained models below.

## Trained Models

We now have all our trained models on [PyTorch Hub](https://pytorch.org/docs/stable/hub.html), so that you can use them in any codebase without cloning this repository simply like this
```
import torch
model = torch.hub.load("gmberton/cosplace", "get_trained_model", backbone="ResNet50", fc_output_dim=2048)
```

As an alternative, you can download the trained models from the table below, which provides links to models with different backbones and dimensionality of descriptors, trained on SF-XL.

<table>
  <tr>
    <th rowspan=2>Model</th>
    <th colspan=7>Dimension of Descriptors</th>
  </tr>
  <tr>
    <td>32</td>
    <td>64</td>
    <td>128</td>
    <td>256</td>
    <td>512</td>
    <td>1024</td>
    <td>2048</td>
  </tr>
  <tr>
    <td>ResNet-18</td>
    <td><a href="https://drive.google.com/file/d/1tfT8r2fBeMVAEHg2bVfCql5pV9YzK620/view?usp=sharing">link</a></td>
    <td><a href="https://drive.google.com/file/d/1-d_Yi3ly3bY6hUW1F9w144FFKsZtYBL4/view?usp=sharing">link</a></td>
    <td><a href="https://drive.google.com/file/d/1HaQjGY5x--Ok0RcspVVjZ0bwrAVmBvrZ/view?usp=sharing">link</a></td>
    <td><a href="https://drive.google.com/file/d/1hjkogugTsHTQ6GTuW3MHqx-t4cXqx0uo/view?usp=sharing">link</a></td>
    <td><a href="https://drive.google.com/file/d/1rQAC2ZddDjzwB2OVqAcNgCFEf3gLNa9U/view?usp=sharing">link</a></td>
    <td>-</td>
    <td>-</td>
  </tr>
  <tr>
    <td>ResNet-50</td>
    <td><a href="https://drive.google.com/file/d/18AxbLO66CO0kG05-1YrRb1YwqN7Wgp6Z/view?usp=sharing">link</a></td>
    <td><a href="https://drive.google.com/file/d/1F2WMt7vMUqXBjsZDIwSga3N0l0r9NP2s/view?usp=sharing">link</a></td>
    <td><a href="https://drive.google.com/file/d/14U3jsoNEWC-QsINoVCWZaHFUGE20fIgZ/view?usp=sharing">link</a></td>
    <td><a href="https://drive.google.com/file/d/1Q2sZPEJfHAe19JaZkdgeFotUYwKbV_x2/view?usp=sharing">link</a></td>
    <td><a href="https://drive.google.com/file/d/1LgDaxCjbQqQWuk5qrPogfg7oN8Ksl1jh/view?usp=sharing">link</a></td>
    <td><a href="https://drive.google.com/file/d/1VBLUiQJfmnZ4kVQIrXBW-AE1dZ3EnMv2/view?usp=sharing">link</a></td>
    <td><a href="https://drive.google.com/file/d/1yNzxsMg34KO04UJ49ncANdCIWlB3aUGA/view?usp=sharing">link</a></td>
  </tr>
  <tr>
    <td>ResNet-101</td>
    <td><a href="https://drive.google.com/file/d/1a5FqhujOn0Pr6duKrRknoOgz8L8ckDSE/view?usp=sharing">link</a></td>
    <td><a href="https://drive.google.com/file/d/17C8jBQluxsbI9d8Bzf67b5OsauOJAIuX/view?usp=sharing">link</a></td>
    <td><a href="https://drive.google.com/file/d/1w37AztnIyGVklBMtm-lwkajb0DWbYhhc/view?usp=sharing">link</a></td>
    <td><a href="https://drive.google.com/file/d/1G5_I4vX4s4_oiAC3EWbrCyXrCOkV8Bbs/view?usp=sharing">link</a></td>
    <td><a href="https://drive.google.com/file/d/1uBKpNfMBt6sLIjCGfH6Orx9eQdQgN-8Z/view?usp=sharing">link</a></td>
    <td><a href="https://drive.google.com/file/d/12BU8BgfqFYzGLXXNaKLpaAzTHuN5I9gQ/view?usp=sharing">link</a></td>
    <td><a href="https://drive.google.com/file/d/1PF7lsSw1sFMh-Bl_xwO74fM1InyYy1t8/view?usp=sharing">link</a></td>
  </tr>
  <tr>
    <td>ResNet-152</td>
    <td><a href="https://drive.google.com/file/d/12pI1FToqKKt8I6-802CHWXDP-JmHEFSW/view?usp=sharing">link</a></td>
    <td><a href="https://drive.google.com/file/d/1rTjlv_pNtXgxY8VELiGYvLcgXiRa2zqB/view?usp=sharing">link</a></td>
    <td><a href="https://drive.google.com/file/d/1q5-szPBn4zL8evWmYT04wFaKjen66mrk/view?usp=sharing">link</a></td>
    <td><a href="https://drive.google.com/file/d/1sCQMA_rsIjmD-f381I0f2yDf0At4TnSx/view?usp=sharing">link</a></td>
    <td><a href="https://drive.google.com/file/d/1ggNYQfGSfE-dciKCS_6SKeQT76O0OXPX/view?usp=sharing">link</a></td>
    <td><a href="https://drive.google.com/file/d/15vBWuHVqEMxkAWWrc7IrkGsQroC65tPc/view?usp=sharing">link</a></td>
    <td><a href="https://drive.google.com/file/d/1AlF7xPSswDLA1TdhZ9yTVBkfRnJm0Hn8/view?usp=sharing">link</a></td>
  </tr>
  <tr>
    <td>VGG-16</td>
    <td>-</td>
    <td><a href="https://drive.google.com/file/d/1YJTBwagC0v50oPydpKtsTnGZnaYOV0z-/view?usp=sharing">link</a></td>
    <td><a href="https://drive.google.com/file/d/1vgw509lGBfJR46cGDJGkFcdBTGhIeyAH/view?usp=sharing">link</a></td>
    <td><a href="https://drive.google.com/file/d/1-4JtACE47rkXXSAlRBFIbydimfKemdo7/view?usp=sharing">link</a></td>
    <td><a href="https://drive.google.com/file/d/1F6CT-rnAGTTexdpLoQYncn-ooqzJe6wf/view?usp=sharing">link</a></td>
    <td>-</td>
    <td>-</td>
  </tr>
</table>

Or you can download all models at once at [this link](https://drive.google.com/drive/folders/1WzSLnv05FLm-XqP5DxR5nXaaixH23uvV?usp=sharing)

## Issues
If you have any questions regarding our code or dataset, feel free to open an issue or send an email to berton.gabri@gmail.com

## Acknowledgements
Parts of this repo are inspired by the following repositories:
- [CosFace implementation in PyTorch](https://github.com/MuggleWang/CosFace_pytorch/blob/master/layer.py)
- [CNN Image Retrieval in PyTorch](https://github.com/filipradenovic/cnnimageretrieval-pytorch) (for the GeM layer)
- [Visual Geo-localization benchmark](https://github.com/gmberton/deep-visual-geo-localization-benchmark) (for the evaluation / test code)

## Cite
Here is the bibtex to cite our paper
```
@InProceedings{Berton_CVPR_2022_CosPlace,
    author    = {Berton, Gabriele and Masone, Carlo and Caputo, Barbara},
    title     = {Rethinking Visual Geo-Localization for Large-Scale Applications},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2022},
    pages     = {4878-4888}
}
```
