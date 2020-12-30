import math
import numpy as np
from scipy.ndimage import affine_transform
from skimage import img_as_float


"""
The body 25 model has 25 joints. Each patch is extracted between 2 of these joints. The values in these tuples 
hold the indices of the joints from which a patch is taken.
"""
BODY_25_PATCH_INDICES = [
    (3, 4),  (6, 7),  # (24, 192) right & left elbow
    (2, 3),  (5, 6),  # (12, 96) right & left arm
    (1, 2),  (1, 5),  # (6, 34) right & left shoulder
    (1, 9),  (1, 12), # (514, 4098) right & left torso
    (9, 10), (12,13), # (1536, 12288) right & left hip
    (10,11), (13,14), # (3072, 24576) right & left thigh
]

"""
The patches will be scaled to these lengths so they all have the same dimensions and depict the same area of the body.
"""
PATCH_LENGTHS = [
    140, 140, # right & left elbow 
    140, 140, # right & left arm
    80,  80,  # right & left shoulder
    260, 260, # right & left torso
    200, 200, # right & left hip
    200, 200  # right & left thigh
]


def patches_from_poselet(image, poselet):
    return [_patch_from_coords(image, poselet[indices[0]], poselet[indices[1]], limb_length=length)
            if not np.any([np.isnan(poselet[indices[0]]), np.isnan(poselet[indices[1]])]) else None
            for indices, length in zip(BODY_25_PATCH_INDICES, PATCH_LENGTHS)]


def _patch_from_coords(image, p1, p2, pad=(5, 5, 5, 5), limb_length=None):
    """
    :param pad: Padding around the limb line (north, east, south, west)
    :param limb_length: The patch will be scaled so that the distance between the coords is equal
                        to this length. If None is set the patch will be returned in original scale.
    """
    a = p2[1] - p1[1]
    b = p2[0] - p1[0]
    c = math.hypot(b, a)
    rotation = math.atan2(b, a)
    middle = ((p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2)

    scale = c / limb_length if limb_length else 1.0

    translation_to_origin = np.array([[1, 0, middle[1]],
                                      [0, 1, middle[0]],
                                      [0, 0, 1]])
    scaling_around_origin = np.array([[scale, 0, 0],
                                      [0, scale, 0],
                                      [0, 0, 1]])
    rotation_around_origin = np.array([[math.cos(rotation), -math.sin(rotation), 0],
                                       [math.sin(rotation), math.cos(rotation), 0],
                                       [0, 0, 1]])
    translation_to_patch = np.array([[1, 0, -(limb_length / 2 + pad[0])],
                                      [0, 1, -(limb_length * 0.2 + pad[3])],
                                      [0, 0, 1]])
    transform = translation_to_origin.dot(scaling_around_origin) \
                                     .dot(rotation_around_origin) \
                                     .dot(translation_to_patch)
    # TODO rewrite in the same manner as MergeData._transform_image
    image_mat = img_as_float(image)
    ch1 = affine_transform(image_mat[:, :, 0], transform, mode='constant', order=0)
    ch2 = affine_transform(image_mat[:, :, 1], transform, mode='constant', order=0)
    ch3 = affine_transform(image_mat[:, :, 2], transform, mode='constant', order=0)
    mask = affine_transform(np.ones((image.shape[0], image.shape[1]), dtype=np.float), transform, mode='constant', order=0)

    out_image = np.dstack([ch1, ch2, ch3, mask])
    patch_height, patch_width = int(limb_length + pad[0] + pad[2]), int(limb_length * 0.2 * 2 + pad[1] + pad[3])

    return np.clip(np.array(out_image[:patch_height, :patch_width], dtype=np.float32), a_min=0.0, a_max=1.0)
