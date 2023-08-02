# Deep Visual Geo-localization Benchmark

This is the official repository for the CVPR 2022 (Oral) paper [Deep Visual Geo-localization Benchmark](https://arxiv.org/abs/2204.03444).
It can be used to reproduce results from the paper, and to compute a wide range of experiments, by changing the components of a Visual Geo-localization pipeline.

<img src="https://raw.githubusercontent.com/gmberton/gmberton.github.io/main/images/vg_system.png" width="90%">

## Table of contents

- [Deep Visual Geo-localization Benchmark](#deep-visual-geo-localization-benchmark)
    - [Table of contents](#table-of-contents)
    - [Setup](#setup)
    - [Running experiments](#running-experiments)
        - [Basic experiment](#basic-experiment)
        - [Architectures and mining](#architectures-and-mining)
            - [Add a fully connected layer](#add-a-fully-connected-layer)
            - [Add PCA](#add-pca)
            - [Evaluate trained models](#evaluate-trained-models)
            - [Reproduce the results](#reproduce-the-results)
        - [Resize](#resize)
        - [Query  pre/post-processing  and  predictions  refinement](#query--prepost-processing--and--predictions--refinement)
        - [Data augmentation](#data-augmentation)
        - [Off-the-shelf models trained on Landmark Recognition datasets](#off-the-shelf-models-trained-on-landmark-recognition-datasets)
        - [Using pretrained networks on other datasets](#using-pretrained-networks-on-other-datasets)
        - [Changing the threshold distance](#changing-the-threshold-distance)
        - [Changing the recall values (R@N)](#changing-the-recall-values-rn)
        - [Model Zoo](#model-zoo)
    - [Acknowledgements](#acknowledgements)

## Setup

Before you begin experimenting with this toolbox, your dataset should be organized in a directory tree as such:

```txt
.
├── benchmarking_vg
└── datasets_vg
    └── datasets
        └── pitts30k
            └── images
                ├── train
                │   ├── database
                │   └── queries
                ├── val
                │   ├── database
                │   └── queries
                └── test
                    ├── database
                    └── queries
```

The [datasets_vg](https://github.com/gmberton/datasets_vg) repo can be used to download a number of datasets. Detailed instructions on how to download datasets are in the repo. Note that many datasets are available, and _pitts30k_ is just an example.

## Running experiments

### Basic experiment

For a basic experiment run

`$ python3 train.py --dataset_name=pitts30k`

this will train a ResNet-18 + NetVLAD on Pitts30k.
The experiment creates a folder named `./logs/default/YYYY-MM-DD_HH-mm-ss`, where checkpoints are saved, as well as an `info.log` file with training logs and other information, such as model size, FLOPs and descriptors dimensionality.

### Architectures and mining

You can replace the backbone and the aggregation as such

`$ python3 train.py --dataset_name=pitts30k --backbone=resnet50conv4 --aggregation=gem`

you can easily use ResNets cropped at conv4 or conv5.

#### Add a fully connected layer

To add a fully connected layer of dimension 2048 to GeM pooling:

`$ python3 train.py --dataset_name=pitts30k --backbone=resnet50conv4 --aggregation=gem --fc_output_dim=2048`

#### Add PCA

To add PCA to a NetVLAD layer just do:

`$ python3 eval.py --dataset_name=pitts30k --backbone=resnet50conv4 --aggregation=netvlad --pca_dim=2048 --pca_dataset_folder=pitts30k/images/train`

where _pca_dataset_folder_ points to the folder with the images used to compute PCA. In the paper we compute PCA's principal components on the train set as it showed best results. PCA is used only at test time.

#### Evaluate trained models

To evaluate the trained model on other datasets (this example is with the St Lucia dataset), simply run

`$ python3 eval.py --backbone=resnet50conv4 --aggregation=gem --resume=logs/default/YYYY-MM-DD_HH-mm-ss/best_model.pth --dataset_name=st_lucia`

#### Reproduce the results

Finally, to reproduce our results, use the appropriate mining method: _full_ for _pitts30k_ and _partial_ for _msls_ as such:

`$ python3 train.py --dataset_name=pitts30k --mining=full`

As simple as this, you can replicate all results from tables 3, 4, 5 of the main paper, as well as tables 2, 3, 4 of the supplementary.

### Resize

To resize the images simply pass the parameters _resize_ with the target resolution. For example, 80% of resolution to the full _pitts30k_ images, would be 384, 512, because the full images are 480, 640:

`$ python3 train.py --dataset_name=pitts30k --resize=384 512`

### Query  pre/post-processing  and  predictions  refinement

We gather all such methods under the _test_method_ parameter. The available methods are _hard_resize_, _single_query_, _central_crop_, _five_crops_mean_, _nearest_crop_ and _majority_voting_.
Although _hard_resize_ is the default, in most datasets it doesn't apply any transformation at all (see the paper for more information), because all images have the same resolution.

`$ python3 eval.py --resume=logs/default/YYYY-MM-DD_HH-mm-ss/best_model.pth --dataset_name=tokyo247 --test_method=nearest_crop`

### Data augmentation

You can reproduce all data augmentation techniques from the paper with simple commands, for example:

`$ python3 train.py --dataset_name=pitts30k --horizontal_flipping --saturation 2 --brightness 1`

### Off-the-shelf models trained on Landmark Recognition datasets

The code allows to automatically download and use models trained on Landmark Recognition datasets from popular repositories: [radenovic](https://github.com/filipradenovic/cnnimageretrieval-pytorch) and [naver](https://github.com/naver/deep-image-retrieval).
These repos offer ResNets-50/101 with GeM and FC 2048 trained on such datasets, and can be used as such:

`$ python eval.py --off_the_shelf=radenovic_gldv1 --l2=after_pool --backbone=r101l4 --aggregation=gem --fc_output_dim=2048`

`$ python eval.py --dataset_name=pitts30k --off_the_shelf=naver --l2=none --backbone=r101l4 --aggregation=gem --fc_output_dim=2048`

### Using pretrained networks on other datasets

Check out our [pretrain_vg](https://github.com/rm-wu/pretrain_vg) repo which we use to train such models.
You can automatically download and train on those models as such

`$ python train.py --dataset_name=pitts30k --pretrained=places`

### Changing the threshold distance

You can use a different distance than the default 25 meters as simply as this (for example to 100 meters):

`$ python3 eval.py --resume=logs/default/YYYY-MM-DD_HH-mm-ss/best_model.pth --val_positive_dist_threshold=100`

### Changing the recall values (R@N)

By default the toolbox computes recalls@ 1, 5, 10, 20, but you can compute other recalls as such:

`$ python3 eval.py --resume=logs/default/YYYY-MM-DD_HH-mm-ss/best_model.pth --recall_values 1 5 10 15 20 50 100`

### Model Zoo

We are currently exploring hosting options, so this is a partial list of models. More models will be added soon!!

<details>
     <summary><b>Pretrained models with different backbones</b></summary></br>
    Pretained networks employing different backbones.</br></br>
    <table>
        <tr>
            <th rowspan=2>Model</th>
            <th colspan="3">Training on Pitts30k</th>
            <th colspan="3">Training on MSLS</th>
         </tr>
         <tr>
              <td>Pitts30k (R@1)</td>
               <td>MSLS (R@1)</td>
               <td>Download</td>
            <td>Pitts30k (R@1)</td>
               <td>MSLS (R@1)</td>
               <td>Download</td>
         </tr>
        <tr>
            <td>vgg16-gem</td>
            <td>78.5</td> <td>43.4</td>
            <td><a href="https://drive.google.com/file/d/1-e9v_mynIX5XBsdtN_mG9tz5-nA5PWiq/view?usp=sharing">[Link]</a></td>
            <td>70.2</td> <td>66.7</td>
            <td><a href="https://drive.google.com/file/d/1GqgO-qG-WNJXWty43KgvDtW0OpG0Wrq-/view?usp=sharing">[Link]</a></td>
         </tr>
         <tr>
             <td>resnet18-gem</td>
            <td>77.8</td> <td>35.3</td>
            <td><a href="https://drive.google.com/file/d/1R66NYeLlxBIqLviUVL9XPZkrtmyMn_tU/view?usp=sharing">[Link]</a></td>
            <td>71.6</td> <td>65.3</td>
            <td><a href="https://drive.google.com/file/d/1IH0d_ME2kU3pagsKhx5ZfRfyWriErajn/view?usp=sharing">[Link]</a></td>
         </tr>
         <tr>
            <td> resnet50-gem </td>
            <td>82.0</td> <td>38.0</td>
            <td><a href="https://drive.google.com/file/d/1esgXzRFvDFHrMnwwR3GlTnErXjFNrYV7/view?usp=sharing">[Link]</a></td>
            <td>77.4</td> <td>72.0</td>
            <td><a href="https://drive.google.com/file/d/1uuIYJN4N7lQqqsN32pbZwjhz5Xvv3zr-/view?usp=sharing">[Link]</a></td>
         </tr>
         <tr>
            <td> resnet101-gem </td>
            <td>82.4</td> <td>39.6</td>
            <td><a href="https://drive.google.com/file/d/1Sd-sezmbzOGbZcy3eqRnWH07eoJ7CM0X/view?usp=sharing">[Link]</a></td>
            <td>77.2</td> <td>72.5</td>
            <td><a href="https://drive.google.com/file/d/1Iondvd8P3vb3piHFTA-RUgTFpqh0I31M/view?usp=sharing">[Link]</a></td>
         </tr>
         <tr>
            <td>vgg16-netvlad</td>
            <td>83.2</td> <td>50.9</td>
            <td><a href="https://drive.google.com/file/d/14s7OZor6wrlGBKeXr0vKbPfTzlW9preM/view?usp=sharing">[Link]</a></td>
            <td>79.0</td> <td>74.6</td>
            <td><a href="https://drive.google.com/file/d/1dwai3uNudjvns58JIyaf5CBRg4ojcWIW/view?usp=sharing">[Link]</a</td>
         </tr>
         <tr>
            <td>resnet18-netvlad</td>
            <td>86.4</td> <td>47.4</td>
            <td><a href="https://drive.google.com/file/d/1KFwonDQYdvzTAIILsOMjmLRUR76jXXvB/view?usp=sharing">[Link]</a></td>
            <td>81.6</td> <td>75.8</td>
            <td><a href="https://drive.google.com/file/d/1_Ozq2TdvwLAJUwy7YH9l69GsfOU-MlFZ/view?usp=sharing">[Link]</a></td>
         </tr>
         <tr>
            <td>resnet50-netvlad</td>
            <td>86.0</td> <td>50.7</td>
            <td><a href="https://drive.google.com/file/d/1KL8HoAApOjJFETin7Q7u7IcsOvroKlSj/view?usp=sharing">[Link]</a></td>
            <td>80.9</td> <td>76.9</td>
            <td><a href="https://drive.google.com/file/d/1krf0A6CeW8GqLqHWZ7dlSNJ9aTJ4dotF/view?usp=sharing">[Link]</a></td>
         </tr>
         <tr>
            <td>resnet101-netvlad</td>
            <td>86.5</td> <td>51.8</td>
            <td><a href="https://drive.google.com/file/d/1064kDJ0LPyWoU7J4bMvAa0lTNEhAEi8v/view?usp=sharing">[Link]</a></td>
            <td>80.8</td> <td>77.7</td>
            <td><a href="https://drive.google.com/file/d/1rtPfsgfJ2Zoxs5uu7Ph1_qc7q-hIxJek/view?usp=sharing">[Link]</a></td>
         </tr>
        <tr>
            <td>cct384-netvlad</td>
            <td>85.0</td> <td>52.5</td>
            <td><a href="https://drive.google.com/file/d/1Rx0oG4PG9bEraIg4y7e6Z24Q6b_TGr5u/view?usp=sharing">[Link]</a></td>
            <td>80.3</td> <td>85.1</td>
            <td><a href="https://drive.google.com/file/d/1wDZ6XRVYz6bcGe_p3Iiz2NfIe9MmZZMN/view?usp=sharing">[Link]</a></td>
         </tr>
    </table>
</details>

<details>
     <summary><b>Pretrained models with different aggregation methods</b></summary></br>
     Pretrained networks trained using different aggregation methods.</br></br>
    <table>
        <tr>
            <th rowspan=2>Model</th>
             <th colspan="3">Training on Pitts30k (R@1)</th>
             <th colspan="3">Training on MSLS (R@1)</th>
         </tr>
         <tr>
              <td>Pitts30k (R@1)</td>
               <td>MSLS (R@1)</td>
               <td>Download</td>
            <td>Pitts30k (R@1)</td>
               <td>MSLS (R@1)</td>
               <td>Download</td>
         </tr>
        <tr>
            <td>resnet50-gem</td>
            <td>82.0</td> <td>38.0</td>
            <td><a href="https://drive.google.com/file/d/1esgXzRFvDFHrMnwwR3GlTnErXjFNrYV7/view?usp=sharing">[Link]</a></td>
            <td>77.4</td> <td>72.0</td>
            <td><a href="https://drive.google.com/file/d/1uuIYJN4N7lQqqsN32pbZwjhz5Xvv3zr-/view?usp=sharing">[Link]</a></td>
         </tr>
         <tr>
            <td>resnet50-gem-fc2048</td>
            <td>80.1</td> <td>33.7</td>
            <td><a href="https://drive.google.com/file/d/1GCbE4gzcRXMH8ETD2YCPo0I3suAXDr-y/view?usp=sharing">[Link]</a></td>
            <td>79.2</td> <td>73.5</td>
            <td><a href="https://drive.google.com/file/d/1oSf11wAxaoEbjLnjfX0EWZ65dgccwdDD/view?usp=sharing">[Link]</a></td>
         </tr>
         <tr>
            <td>resnet50-gem-fc65536</td>
            <td>80.8</td> <td>35.8</td>
            <td><a href="https://drive.google.com/file/d/19GjodUuAGKpac6WhIcfuy3tiPV1J-ikn/view?usp=sharing">[Link]</a></td>
            <td>79.0</td> <td>74.4</td>
            <td><a href="https://drive.google.com/file/d/1OGwt651loL2vXnQYyABqitL39IEiXhag/view?usp=sharing">[Link]</a></td>
         </tr>
         <tr>
            <td>resnet50-netvlad</td>
            <td>86.0</td> <td>50.7</td>
            <td><a href="https://drive.google.com/file/d/1KL8HoAApOjJFETin7Q7u7IcsOvroKlSj/view?usp=sharing">[Link]</a></td>
            <td>80.9</td> <td>76.9</td>
            <td><a href="https://drive.google.com/file/d/1krf0A6CeW8GqLqHWZ7dlSNJ9aTJ4dotF/view?usp=sharing">[Link]</a></td>
         </tr>
         <tr>
            <td>resnet50-crn</td>
            <td>85.8</td> <td>54.0</td>
            <td><a href="https://drive.google.com/file/d/1mLOkILfIf8Wegi3tva9390TRIbWDxRor/view?usp=sharing">[Link]</a></td>
            <td>80.8</td> <td>77.8</td>
            <td><a href="https://drive.google.com/file/d/1KJzXwCsbyT0uNDl925H2J0QKXhKaeEgW/view?usp=sharing">[Link]</a></td>
         </tr>
    </table>
</details>

<details>
     <summary><b>Pretrained models with different mining methods</b></summary><br/>
    Pretained networks trained using three different mining methods (random, full database mining and partial database mining):</br></br>
    <table>
        <tr>
            <th rowspan=2>Model</th>
             <th colspan="3">Training on Pitts30k (R@1)</th>
             <th colspan="3">Training on MSLS (R@1)</th>
         </tr>
         <tr>
              <td>Pitts30k (R@1)</td>
               <td>MSLS (R@1)</td>
               <td>Download</td>
            <td>Pitts30k (R@1)</td>
               <td>MSLS (R@1)</td>
               <td>Download</td>
         </tr>
        <tr>
            <td> resnet18-gem-random</td>
            <td>73.7</td> <td>30.5</td>
            <td><a href="https://drive.google.com/file/d/12Ds-LcvFcA609bZVBTLNjAZIzV-g8UGK/view?usp=sharing">[Link]</a></td>
            <td>62.2</td> <td>50.6</td>
            <td><a href="https://drive.google.com/file/d/1oNZyfjTaulVTFX4wRrj0YISqxLuNRyhy/view?usp=sharing">[Link]</a></td>
         </tr>
        <tr>
            <td> resnet18-gem-full</td>
            <td>77.8</td> <td>35.3</td>
            <td><a href="https://drive.google.com/file/d/1bHVsnb6Km2npBsGK9ylI1vuOuc3WLKJb/view?usp=sharing">[Link]</a></td>
            <td>70.1</td><td>61.8</td>
            <td><a href="https://drive.google.com/file/d/1BbANLPVPxWDau2RP0cWTSS3FybbyUPL1/view?usp=sharing">[Link]</a></td>
         </tr>
        <tr>
            <td> resnet18-gem-partial</td>
            <td>76.5</td> <td>34.2</td>
            <td><a href="https://drive.google.com/file/d/1R66NYeLlxBIqLviUVL9XPZkrtmyMn_tU/view?usp=sharing">[Link]</a></td>
            <td>71.6</td> <td>65.3</td>
            <td><a href="https://drive.google.com/file/d/1IH0d_ME2kU3pagsKhx5ZfRfyWriErajn/view?usp=sharing">[Link]</a></td>
         </tr>
        <tr>
            <td> resnet18-netvlad-random</td>
            <td>83.9</td> <td>43.6</td>
            <td><a href="https://drive.google.com/file/d/19OcEe2ckk-D8drrmxpKkkarT_5mCkjnt/view?usp=sharing">[Link]</a></td>
            <td>73.3</td> <td>61.5</td>
             <td><a href="https://drive.google.com/file/d/1JlEbKbnWyCbR4zP1ZYDct3pYtuJrUmVp/view?usp=sharing">[Link]</a></td>
         </tr>
         <tr>
            <td> resnet18-netvlad-full</td>
            <td>86.4</td> <td>47.4</td>
            <td><a href="https://drive.google.com/file/d/1kwgyDEfRYtdaOEimQQlmj77rIR2tH3st/view?usp=sharing">[Link]</a></td>
            <td>-</td><td>-</td>
            <td>-</td>
         </tr>
         <tr>
            <td> resnet18-netvlad-partial</td>
            <td>86.2</td> <td>47.3</td>
            <td><a href="https://drive.google.com/file/d/1KFwonDQYdvzTAIILsOMjmLRUR76jXXvB/view?usp=sharing">[Link]</a></td>
            <td>81.6</td> <td>75.8</td>
            <td><a href="https://drive.google.com/file/d/1_Ozq2TdvwLAJUwy7YH9l69GsfOU-MlFZ/view?usp=sharing">[Link]</a></td>
         </tr>
         <tr>
            <td> resnet50-gem-random</td>
            <td>77.9</td> <td>34.3</td>
            <td><a href="https://drive.google.com/file/d/1f9be75EaG0fFLeNF0bufSre_efKH_ObU/view?usp=sharing">[Link]</a></td>
            <td>69.5</td> <td>57.4</td>
            <td><a href="https://drive.google.com/file/d/1h9-av6qMn-LVapI5KA4cZhT5BKaZ79C6/view?usp=sharing">[Link]</a></td>
        </tr>
        <tr>
            <td> resnet50-gem-full</td>
            <td>82.0</td> <td>38.0</td>
            <td><a href="https://drive.google.com/file/d/1quS9ZjOrXBqNDBhQzlSj8aeh3dBfP1GY/view?usp=sharing">[Link]</a></td>
            <td>77.3</td> <td>69.7</td>
            <td><a href="https://drive.google.com/file/d/1pxU881eTcz_YdQthKz5yohU7WoLpXt8J/view?usp=sharing">[Link]</a></td>
        </tr>
        <tr>
            <td> resnet50-gem-partial</td>
            <td>82.3</td> <td>39.0</td>
            <td><a href="https://drive.google.com/file/d/1esgXzRFvDFHrMnwwR3GlTnErXjFNrYV7/view?usp=sharing">[Link]</a></td>
            <td>77.4</td> <td>72.0</td>
            <td><a href="https://drive.google.com/file/d/1uuIYJN4N7lQqqsN32pbZwjhz5Xvv3zr-/view?usp=sharing">[Link]</a></td>
        </tr>
        <tr>
            <td> resnet50-netvlad-random</td>
            <td>83.4</td> <td>45.0</td>
            <td><a href="https://drive.google.com/file/d/1TkzlO-ZS42u6e783y2O3JZhcIoI7CEVj/view?usp=sharing">[Link]</a></td>
            <td>74.9</td> <td>63.6</td>
            <td><a href="https://drive.google.com/file/d/1E_X2nrnLxBqvLfVfNKtorOGW_VmwOqSu/view?usp=sharing">[Link]</a></td>
        </tr>
        <tr>
            <td> resnet50-netvlad-full</td>
            <td>86.0</td> <td>50.7</td>
            <td><a href="https://drive.google.com/file/d/133uxEJZ0gK6XL1myhSAFC7wibZtWnugK/view?usp=sharing">[Link]</a></td>
            <td>-</td><td>-</td>
            <td>-</td>
        </tr>
        <tr>
            <td> resnet50-netvlad-partial</td>
            <td>85.5</td> <td>48.6</td>
            <td><a href="https://drive.google.com/file/d/1GCbE4gzcRXMH8ETD2YCPo0I3suAXDr-y/view?usp=sharing">[Link]</a></td>
            <td>80.9</td> <td>76.9</td>
            <td><a href="https://drive.google.com/file/d/1krf0A6CeW8GqLqHWZ7dlSNJ9aTJ4dotF/view?usp=sharing">[Link]</a></td>
        </tr>
    </table>
</details>

If you find our work useful in your research please consider citing our paper:

```bib
@inProceedings{Berton_CVPR_2022_benchmark,
    author    = {Berton, Gabriele and Mereu, Riccardo and Trivigno, Gabriele and Masone, Carlo and
                 Csurka, Gabriela and Sattler, Torsten and Caputo, Barbara},
    title     = {Deep Visual Geo-localization Benchmark},
    booktitle = {CVPR},
    month     = {June},
    year      = {2022},
}
```

## Acknowledgements

Parts of this repo are inspired by the following great repositories:

- [NetVLAD's original code](https://github.com/Relja/netvlad) (in MATLAB)
- [NetVLAD layer in PyTorch](https://github.com/lyakaap/NetVLAD-pytorch)
- [NetVLAD training in PyTorch](https://github.com/Nanne/pytorch-NetVlad/)
- [GeM layer](https://github.com/filipradenovic/cnnimageretrieval-pytorch)
- [Deep Image Retrieval](https://github.com/naver/deep-image-retrieval)
- [Mapillary Street-level Sequences](https://github.com/mapillary/mapillary_sls)
- [Compact Convolutional Transformers](https://github.com/SHI-Labs/Compact-Transformers)

Check out also our other repo [_CosPlace_](https://github.com/gmberton/CosPlace), from the CVPR 2022 paper "Rethinking Visual Geo-localization for Large-Scale Applications", which provides a new SOTA in visual geo-localization / visual place recognition.
