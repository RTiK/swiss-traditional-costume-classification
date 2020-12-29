import sys
import numpy as np

from configparser import ConfigParser

config = ConfigParser()
config.read_file(open('config.ini'))
openpose_config = config['openpose']
sys.path.append(openpose_config.get('openpose_python_path'))

from openpose import pyopenpose as op


# these are default OpenPose parameters from the openpose (v1.4) example
params = {
    "logging_level": 3,
    "output_resolution": "-1x-1",
    "net_resolution": "-1x368",
    "model_pose": "BODY_25",
    "alpha_pose": 0.6,
    "scale_gap": 0.3,
    "scale_number": 1,
    "render_threshold": 0.05,
    "num_gpu_start": 0,
    "disable_blending": False,
    "model_folder": "resources/models/"
}

# taken from the openpose python example
opWrapper = op.WrapperPython()
opWrapper.configure(params)
opWrapper.start()


def extract_poselets(image):
    datum = op.Datum()
    datum.cvInputData = image
    opWrapper.emplaceAndPop(op.VectorDatum([datum]))
    return datum.poseKeypoints


def biggest_poselet(kps, provide_area=False):
    if provide_area:
        return sorted([(kp, _bounding_box_area(kp)) for kp in kps], key=lambda x: x[1], reverse=True)[0]
    else:
        return sorted([(kp, _bounding_box_area(kp)) for kp in kps], key=lambda x: x[1], reverse=True)[0][0]


def _bounding_box_area(coords):
    return (np.nanmax(coords[:, 0]) - np.nanmin(coords[:, 0])) * (np.nanmax(coords[:, 1]) - np.nanmin(coords[:, 1]))
