# Visual GeoLocalization Datasets

Using these scripts you can download a number of visual geolocalization datasets. This code was part of the CVPR 2022 [Deep Visual Geo-localization Benchmark](https://arxiv.org/abs/2204.03444), so if you use these scripts make sure to cite:
```
@inProceedings{Berton_CVPR_2022_benchmark,
    author    = {Berton, Gabriele and Mereu, Riccardo and Trivigno, Gabriele and Masone, Carlo and
                 Csurka, Gabriela and Sattler, Torsten and Caputo, Barbara},
    title     = {Deep Visual Geo-localization Benchmark},
    booktitle = {CVPR},
    month     = {June},
    year      = {2022},
}
```

The datasets are downloaded and formatted using a standard format, for which the metadata of the images are contained within the images filename.
We adopt a convention so that the names of the files with the images are:

```txt
@ UTM_easting @ UTM_northing @ UTM_zone_number @ UTM_zone_letter @ latitude @ longitude @ pano_id @ tile_num @ heading @ pitch @ roll @ height @ timestamp @ note @ extension
```

Note that for many datasets some of these values are empty, and however the only required values are
UTM coordinates (obtained from latitude and longitude).

The reason for using the character "@" as a separator, is that commonly used characters such as dash "-" or underscore "\_" might be used in the fields, for example in the _pano\_id_ field.

The directory tree that is generated is as follows:

```txt
.
└── datasets
    └── dataset_name
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

You might want to link a scratch for dataset download

```bash
mkdir /scratch/$USER/vl-vpr/datasets
ln -s /scratch/$USER/vl-vpr/datasets ./
# For the benchmarking scripts, etc.
export DATASETS_FOLDER=$(realpath $(pwd)/datasets)
```

For training throughout our benchmark we used Pitts and MSLS as dataset, and the others, listed below, only as test set to evaluate the generalization capability of the models.
This is for many reasons, like the absence of a time machine that is necessary to train robust models.

The list of datasets that you can download with this code is the following:

- Pitts30k*
- Pitts250k*
- Mapillary SLS**
- Eysham - as test set only
- San Francisco - as test set only
- Tokyo 24/7* - as test set only
- St Lucia - as test set only
- SVOX - as test set only
- Nordland - as test set only

To download each dataset, simply run the corresponding python script, that will download,
unpack and format the file according to the structure above.

*: for Pitts30k, Pitts250k and Tokyo 24/7 the images should be downloaded by asking permission to the respective authors. Then they can be formatted with this codebase

*\*: for Mapillary SLS, you need to first log in into their website, download it [here](https://www.mapillary.com/dataset/places),
 then extract the zip files and run `$ python format_mapillary.py`

## Pitts30k

For Pitts30k, first download the data under datasets/pitts30k/raw_data, then simply run `$ python format_pitts30k.py`

## Pitts250k

For Pitts250k, first download the data under datasets/pitts250k/raw_data, then simply run `$ python format_pitts250k.py`

## Mapillary SLS

For Mapillary SLS, you need to first log in into their website, download it [here](https://www.mapillary.com/dataset/places),
 then extract the zip files, and place it in a folder `datasets` inside the repository root and name it
`mapillary_sls`.
Then you can run:

 `$ python format_mapillary.py`

## Eynsham

To download Eynsham, simply run `$ python download_eynsham.py`

## San Francisco

To download San Francisco, simply run `$ python download_san_francisco.py`

## St Lucia

To download St Lucia, simply run `$ python download_st_lucia.py`

## SVOX

To download SVOX, simply run `$ python download_svox.py`

## Nordland

To download Nordland, simply run `$ python download_nordland.py`

The images will be arranged to have GPS/UTM labels compatible with the benchmarking code. More info on it are in the comment on top of the `download_nordland.py` script. We used the splits used by the [Patch-NetVLAD paper](https://arxiv.org/abs/2103.01486).

## Tokyo 24/7

For Tokyo 24/7, first download the data under datasets/tokyo247/raw_data, then simply run `$ python format_tokyo247.py`. Queries are automatically downloaded.
