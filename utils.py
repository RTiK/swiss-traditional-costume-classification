from configparser import ConfigParser
import numpy as np
from skimage.color import yiq2rgb, lab2rgb, rgb2yiq, rgb2lab
from skimage.transform import rescale
from skimage.io import imsave, imread
from skimage import img_as_float32
import json
import random
from datetime import datetime
from glob import glob


config = ConfigParser()
config.read_file(open('config.ini'))
data_config = config['data']

IMAGES_PATH = data_config.get('images_path')
PATCHES_PATH = data_config.get('patches_path')
DESCRIPTORS_PATH = data_config.get('descriptors_path')
PREDICTIONS_PATH = data_config.get('predictions_path')


# descriptor I/O
def save_descriptors_as_rgb(descriptors, metric_name, color_model_name, folder_name=DESCRIPTORS_PATH):
    for cls, des in descriptors.items():
        for i, desc in enumerate(des):
            # merging operation can produce values outside allowed range
            if color_model_name.lower() == 'rgb':
                im = np.clip(desc, a_min=0.0, a_max=1.0)
            elif color_model_name.lower() == 'yiq':
                im = np.clip(np.dstack([yiq2rgb(desc[:, :, :3]), desc[:, :, 3]]), a_min=0.0, a_max=1.0)
            elif color_model_name.lower() == 'lab':
                im = np.clip(np.dstack([lab2rgb(desc[:, :, :3]), desc[:, :, 3]]), a_min=0.0, a_max=1.0)
            else:
                raise Exception('Unknown color model %s' % color_model_name)
            imsave('%s/%s-%s-%s-%s.png' % (folder_name, cls, metric_name, color_model_name, i), im)


def save_descriptors_as_npy(descriptors, metric_name, color_model_name, folder_name=DESCRIPTORS_PATH):
    for cls, des in descriptors.items():
        for i, desc in enumerate(des):
            np.save('%s/%s-%s-%s-%s.npy' % (folder_name, cls, metric_name, color_model_name, i), desc)


def load_descriptors_npy(metric_name, color_model_name, folder_name=DESCRIPTORS_PATH):
    classes = set([int(path.split('/')[-1].split('-', maxsplit=1)[0])
                   for path in glob('%s/*-%s-%s-*.npy' % (folder_name, metric_name, color_model_name))])
    descriptors = {cls: [] for cls in classes}
    for cls in descriptors.keys():
        feature_set = [np.load('%s/%s-%s-%s-%d.npy' % (folder_name, cls, metric_name, color_model_name, idx))
                       for idx in range(6)]
        descriptors[cls] = feature_set
    return descriptors


# predictions I/O
def append_predictions_to_file(predictions, y_true, metric_name, color_model_name, sample_set_name, random_seed='', folder_name=PREDICTIONS_PATH):
    dump_pred = []
    for pred in predictions:
        cls_dict = {}
        for cls, scores in pred.items():
            cls_dict[cls] = [score.tolist() for score in scores]
        dump_pred += [cls_dict]
    with open('%s/%s-%s-%s.json' % (folder_name, metric_name, color_model_name, sample_set_name), 'a') as f:
        json.dump({
            'meta': {'random_seed': random_seed, 'datetime': str(datetime.now())},
            'true': y_true,
            'predictions': dump_pred
        }, f)
        f.write('\n')


def read_predictions_from_file(metric_name, color_model_name, sample_set_name, folder_name=PREDICTIONS_PATH):
    predictions, trues, metas = [], [], []
    with open('%s/%s-%s-%s.json' % (folder_name, metric_name, color_model_name, sample_set_name)) as f:
        for line in f.readlines():
            raw = json.loads(line)
            processed = []
            for pred in raw['predictions']:
                cls_dict = {}
                for cls, scores in pred.items():
                    cls_dict[int(cls)] = [np.array(score, dtype=np.float32) for score in scores]
                processed += [cls_dict]
            metas += [raw['meta']]
            predictions += [processed]
            trues += [raw['true']]
    return predictions, trues, metas


# img I/O
to_flip = [12, 34, 192, 1536, 4098, 24576]  # these patches from the right will be flipped
patch_map = {6: 2, 12: 1, 24: 0, 34: 2, 96: 1, 192: 0, 514: 3, 1536: 4, 3072: 5, 4098: 3, 12288: 4, 24576: 5}  # feature code to feature index

'''
In the following methods the `data` argument is expected to provide 
{class0: {
    feature_code: [file_name, file_name, ...],
    feature_code: [file_name, file_name, ...],
    feature_code: [file_name, file_name, ...],
    ...
}, ...}
'''


def read_patches_rgb(data, scale=0.5):
    img_data = {}
    for cls, limbs in data.items():
        img_data[cls] = [[], [], [], [], [], []]
        for limb, files in limbs.items():
            dest_i = patch_map[limb]
            for f in files:
                if limb not in to_flip:
                    img_data[cls][dest_i] += [
                        img_as_float32(rescale(imread('%s/%s.png' % (PATCHES_PATH, f)), scale, multichannel=True))]
                else:
                    img_data[cls][dest_i] += [
                        img_as_float32(rescale(np.fliplr(imread('%s/%s.png' % (PATCHES_PATH, f))), scale, multichannel=True))]
    return img_data


def read_patches_yiq(data, scale=0.5):
    img_data = {}
    for cls, limbs in data.items():
        img_data[cls] = [[], [], [], [], [], []]
        for limb, files in limbs.items():
            dest_i = patch_map[limb]
            for f in files:
                im = rescale(imread('%s/%s.png' % (PATCHES_PATH, f)), scale, multichannel=True) \
                        if limb not in to_flip \
                        else rescale(np.fliplr(imread('%s/%s.png' % (PATCHES_PATH, f))), scale, multichannel=True)
                conv_im = np.array(np.dstack([rgb2yiq(im[:, :, :3]), im[:, :, 3]]), dtype=np.float32)
                img_data[cls][dest_i] += [conv_im]
    return img_data


def read_patches_0iq(data, scale=0.5):
    img_data = {}
    for cls, limbs in data.items():
        img_data[cls] = [[], [], [], [], [], []]
        for limb, files in limbs.items():
            dest_i = patch_map[limb]
            for f in files:
                im = rescale(imread('%s/%s.png' % (PATCHES_PATH, f)), scale, multichannel=True) \
                        if limb not in to_flip \
                        else rescale(np.fliplr(imread('%s/%s.png' % (PATCHES_PATH, f))), scale, multichannel=True)
                conv_im = np.array(np.dstack([rgb2yiq(im[:, :, :3]), im[:, :, 3]]), dtype=np.float32)
                conv_im[:, :, 0] = 0
                img_data[cls][dest_i] += [conv_im]
    return img_data


def read_patches_lab(data, scale=0.5):
    img_data = {}
    for cls, limbs in data.items():
        img_data[cls] = [[], [], [], [], [], []]
        for limb, files in limbs.items():
            dest_i = patch_map[limb]
            for f in files:
                im = rescale(imread('%s/%s.png' % (PATCHES_PATH, f)), scale, multichannel=True) \
                        if limb not in to_flip \
                        else rescale(np.fliplr(imread('%s/%s.png' % (PATCHES_PATH, f))), scale, multichannel=True)
                conv_im = np.array(np.dstack([rgb2lab(im[:, :, :3]), im[:, :, 3]]), dtype=np.float32)
                img_data[cls][dest_i] += [conv_im]
    return img_data


def read_patches_0ab(data, scale=0.5):
    img_data = {}
    for cls, limbs in data.items():
        img_data[cls] = [[], [], [], [], [], []]
        for limb, files in limbs.items():
            dest_i = patch_map[limb]
            for f in files:
                im = rescale(imread('%s/%s.png' % (PATCHES_PATH, f)), scale, multichannel=True) \
                        if limb not in to_flip \
                        else rescale(np.fliplr(imread('%s/%s.png' % (PATCHES_PATH, f))), scale, multichannel=True)
                conv_im = np.array(np.dstack([rgb2lab(im[:, :, :3]), im[:, :, 3]]), dtype=np.float32)
                conv_im[:, :, 0] = 0
                img_data[cls][dest_i] += [conv_im]
    return img_data


# dataset operations
def split_sets_by_patch_count(data, min_count=6):
    large_img_data = {}
    remaining_img_data = {}
    for cls, patches in data.items():
        mean_patch_count = sum([len(c) for c in patches]) / 6
        if mean_patch_count >= min_count and min([len(c) for c in patches]) > 1:
            large_img_data[cls] = patches
        else:
            remaining_img_data[cls] = patches
    return large_img_data, remaining_img_data


def split_sets_in_train_and_test(data, split_ratio=3, seed=None):  # test set will contain 1/split_ratio sets
    X_train = {}
    X_test = {}
    if seed:
        random.seed(seed)
    for cls, limbs in data.items():
        min_count = min([len(s) for s in limbs])
        train_data = []
        test_data = []
        for i, ims in enumerate(limbs):
            idx = list(range(len(ims)))
            random.shuffle(idx)
            test_idx = idx[:min_count//split_ratio]
            train_idx = idx[min_count//split_ratio:]
            train_data += [[ims[t_i] for t_i in train_idx]]
            test_data += [[ims[t_i] for t_i in test_idx]]
        X_train[cls] = train_data
        X_test[cls] = test_data
    return X_train, X_test


def prepare_whole_sample_sets(data):
    """
    Returns as many complete sample sets from data as possible.
    """
    X_sets = []
    y_true = []
    for cls, limbs in data.items():
        min_limbs = min([len(l) for l in limbs])
        for i in range(min_limbs):
            X_sets += [[limbs[0][i], limbs[1][i], limbs[2][i], limbs[3][i], limbs[4][i], limbs[5][i]]]
            y_true += [cls]
    return X_sets, y_true


# default class values, for when DB is unavailable
class_lookup = {4: 1, 24: 1, 5: 2, 23: 2, 14: 13, 167: 13, 168: 13, 151: 18, 20: 19, 22: 19, 28: 26,
                107: 32, 108: 32, 109: 33, 110: 33, 111: 33, 175: 33, 176: 33, 187: 33, 177: 36, 178: 37,
                41: 39, 42: 39, 179: 44, 180: 45, 181: 45, 182: 45, 183: 45, 53: 57, 54: 57, 55: 57,
                56: 57, 92: 59, 93: 59, 171: 61, 188: 61, 189: 61, 190: 61, 191: 61, 94: 62, 95: 62,
                186: 62, 63: 65, 64: 65, 67: 69, 68: 69, 70: 72, 71: 72, 73: 75, 74: 75, 184: 76, 88: 77,
                89: 77, 90: 77, 172: 77, 185: 77, 91: 78, 173: 96, 98: 97, 174: 97, 100: 99, 101: 99,
                102: 99, 103: 99, 104: 99, 105: 99, 106: 99, 113: 112, 114: 112, 116: 115, 117: 115,
                119: 118, 120: 118, 121: 118, 123: 122, 124: 122, 125: 122, 139: 138, 140: 138, 144: 143,
                145: 143, 153: 152, 154: 152, 158: 157, 159: 157, 160: 157, 162: 161, 163: 161, 164: 161,
                51: 169, 170: 169}


def top_classes_from(y):
    return [class_lookup[_y] if _y in class_lookup else _y for _y in y]
