# Swiss Traditional Costume Classification

This repository contains the code for my master's project. The goal was to preform a classification of a Swiss traditional costume based on a photo.

Each region in Switzerland has unique costumes for different purposes and occasions e.g. celebrations, work clothes, church visits. The appearance of each cosutme defined in written guidelines. These guidelines must be strictly followed, however, these guidelines allow variations in color or the choice of accessoires (for unmarried persons) for many costumes. In this project we call these variations _subtypes of a costume_ and they have a reference to the _costume supertype_. In this README we will only talk of _supertypes_ to keep things simple.

In total there are ower 400 different costumes. We managed to gather 1540 full body shots of persons in 274 different costumes. Unfortunately, we cannot share this dataset because it contains prvivate pictures. But we do provide
The scarcity of the dataset is one of the biggest challenges we faced in this project. 

We used [OpenPose](https://github.com/CMU-Perceptual-Computing-Lab/openpose) for pose recognition and used body parts as features. Then we cut out patches from the image containing these body parts and compute feature descriptors from the patches from the same costume and the same body part. Patches from new image are compared to the descriptors and thus the most similar costume is determined. For more details check out our [paper](https://www.researchgate.net/publication/340042658_Image-based_Classification_of_Swiss_Traditional_Costumes_using_Contextual_Features).

## Contents

### Inference pipeline

This repo contains everything needed for running the inference pipeline. The frontend is a Flask server prividing a website for upload of images and displaying of results. After upload the image is stored in the upload directory (default is `data/uploads`) under a unique file name. This file name is placed in a Celery task queue with Redis as backend. A worker performs the classification and returns similarity scores for each class. These results are returned to the frontend and displayed to the user.

### Jupyter notebooks

Additionally, the repository includes two Jupyter notebooks:

* `patch_extraction.ipynb` contains code for extracting patches from full body photos

* `descriptor_construction.ipynb` shows a typical procedure for constructing descriptors from patches

## Database

The data used in this project is organized in tables below. The tables in the upper row are used to organize the costumes. The tables `samples`, `patches` and `descriptors` contain the names of the files. We provide the files for the patches and the descriptors constructed from them but as already mentioned above we cannot share the samples. For the same reason the column `email` in the tables `samples` remains empty too.


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

* Import the database into a MySQL database (or a comparable relational database), the schema `stcdb` will be created automatically

```bash
mysql --user user --password < data/database/stcdb_backup.sql
```

* Extract the archives `patches.7z` and `descriptors.7z` (default location for these directories is `data/patches` and `data/descriptors`)

```bash
7z x patches.7z -o data/patches
7z x descriptors.7z -o data/descriptors
```

* Make a copy of the configuration template

```bash
cp config_template.ini config.ini
```

* Modify the configuration file according to your setup

## Startup

* Deploy Redis

```bash
Redis
```

* Deploy Celery

```bash
celery -A classifier_worker worker -c 1 -P solo
```

> The argument `solo` starts only one process which is OK because poselet recognition is a bottleneck

* Deploy the web server

```bash
python backend.py
```