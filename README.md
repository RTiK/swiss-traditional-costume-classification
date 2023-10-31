# Swiss Traditional Costume Classification

## How it works

We use [OpenPose](https://github.com/CMU-Perceptual-Computing-Lab/openpose) to approimate the pose of the person in the image. Having a poselet, we cut out patches containing forearms, upper arms, shoulders, torso, upper and lower legs from the image. Afterwards we compute feature descriptors from the patches of the same costume and the same body part. 

A new image goes through the same process of patch extraction. Then the new patches are compared to the descriptors of all known classes and the costume is assigned to the class of the most similar decriptor. For more details check out our [paper](https://www.researchgate.net/publication/340042658_Image-based_Classification_of_Swiss_Traditional_Costumes_using_Contextual_Features).

## How traditional costumes work

Each region in Switzerland has unique traditional costumes for different purposes and occasions e.g. celebrations, work clothes, church visits. The appearance of each costume is rigorously defined. These guidelines must be followed strictly, however, they allow some variations in color, choice of accessoires etc. for many costumes. In this project we call these variations _subtypes of a costume_. These subtypes keep a reference to the _costume supertype_. In this README we will only talk of _supertypes_ to keep things simple.

In total there are over 400 different costumes. We managed to gather 1540 full-body pictures of people wearing 274 different costumes. Unfortunately, we cannot share this dataset for privacy reasons. But we do provide an archive with the extracted patches and the descriptors computed from them.

## Contents of the repository

### Inference pipeline

This repository contains everything needed to run the inference pipeline. The frontend is a Flask server hosting a website for uploading images and displaying the results. After upload the image is stored in the upload directory (default is `data/uploads`) under a unique file name. This file name is placed in a Celery task queue with Redis as backend. A worker consumes the task, performs classification and returns similarity scores for each class. These results are returned to the frontend and displayed to the user.

### Jupyter notebooks

Additionally, the repository includes two Jupyter notebooks:

* `patch_extraction.ipynb` contains code for extracting patches from full-body pictures

* `descriptor_construction.ipynb` shows a typical procedure for constructing descriptors from patches

## Database

The data used in this project is organized as shown in diagram below. The tables in the upper row are used to organize the costumes. The tables `samples`, `patches` and `descriptors` contain the names of the files. We provide the files for the patches and the descriptors constructed from them but as already mentioned above we cannot share the samples.


```
    cantons <------- regions <------- costumes -------> purposes
                                       ^    ^
                                       |    '-----------,
                            ,----------'             samples
                            |                           ^
                       descriptors       patches -------'
```

## Setup

* Install [OpenPose (v1.7)](https://github.com/CMU-Perceptual-Computing-Lab/openpose/releases)

* Install Redis

* Clone this repository

* Install the modules in `requirements.txt` (better use a virtual environment)

```bash
pip install -r requirements.txt
```

* Import the database backup into a MySQL database (or a comparable relational database). The schema `stcdb` will be created automatically

```bash
mysql --user user --password < data/database/stcdb_backup.sql
```

* Extract the archives `patches.7z` and `descriptors.7z` in the Release 0.1 (default location for these directories is `data/patches` and `data/descriptors`)

```bash
7z x patches.7z -o data/patches
7z x descriptors.7z -o data/descriptors
```

* Make a copy of the configuration template

```bash
cp config_template.ini config.ini
```

* Modify the configuration file `config.ini` according to your setup

## Startup

* Deploy Redis

```bash
redis
```

* Deploy Celery

```bash
celery -A classifier_worker worker -c 1 -P solo
```

> The argument `solo` starts only one worker. This is OK because we don't want OpenPose to be available on multiple threads in GPU.

* Deploy the web server

```bash
python backend.py
```
