from celery import Celery
from skimage.io import imread
from skimage.transform import rescale
from skimage.color import rgb2lab, rgb2yiq
from poselets import *
from patches import *
from classifiers import *
from configparser import ConfigParser
from db_operations import DBOps
import logging


log = logging.getLogger()

config = ConfigParser()
config.read_file(open('config.ini'))

classifier_config = config['classifier']
color_model = classifier_config.get('color_model')
similarity_metric = classifier_config.get('similarity_metric')
rejection_score = classifier_config.getfloat('rejection_score')

db_config = config['database']
host_name = db_config.get('host')
db_name = db_config.get('name')
username = db_config.get('username')
password = db_config.get('password')

db_ops = DBOps(host_name, db_name, username, password)

log.info('''Init parameters:
Color space: %s
Similarity metric: %s
Default rejection score: %s
''' % (color_model, similarity_metric, rejection_score))

classes_supported = db_ops.feature_sets_available(color_model, similarity_metric)

log.info('Reading descriptors: ' + str(classes_supported))

descriptors = {cl: db_ops.load_feature_set(cl, color_model, similarity_metric)
               for cl in classes_supported}

classifier = MseClassifier(descriptors) \
             if 'mse' in similarity_metric else \
             CcClassifier(descriptors)


def dummy_convert(image):
    return image


if color_model == 'yiq':
    convert = rgb2yiq
elif color_model == 'lab':
    convert = rgb2lab
else:
    convert = dummy_convert


app = Celery('classifier', broker='redis://localhost:6379', backend='redis://localhost:6379')


@app.task
def classify(filename, ext):

    image = imread('%s.%s' % (filename, ext))
    log.info('%s: Image read' % filename)
    psts = extract_poselets(image)

    if not len(psts):
        log.warning('%s: No poselets found in image' % filename)
        raise Exception('No poselet was found')

    pst = biggest_poselet(psts)

    log.info('%s: poselet retrieved: %s' % (filename, str(pst)))

    patches = []
    for i, patch in enumerate(patches_from_poselet(image, pst)):
        if patch is None:
            patches += [None]
        else:
            p = rescale(np.fliplr(patch), 0.5, mode='constant', order=0, multichannel=True, anti_aliasing=True) \
                 if i % 2 else \
                 rescale(patch, 0.5, mode='constant', order=0, multichannel=True, anti_aliasing=True)
            patches += [np.array(np.dstack([convert(p[:, :, :3]), p[:, :, 3]]), np.float32)]

    log.info('%s: cutouts extracted' % filename)
    number_of_patches = sum([1 for p in patches if p is not None])

    scores = classifier.predict(patches)
    scores[-1] = rejection_score * number_of_patches  # append rejection class (sum of scores for each descriptor)

    log.info('%s: prediction completed: %s' % (filename, str(scores.items())))

    return scores
