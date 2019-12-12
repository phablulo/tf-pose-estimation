import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import cv2
import numpy as np
from tf_pose.estimator import TfPoseEstimator
from tf_pose.networks import get_graph_path, model_wh

def predict_2d(image, model='mobilenet_thin', resize='432x368'):
  w, h = model_wh(resize)
  if w == 0 or h == 0:
    e = TfPoseEstimator(get_graph_path(model), target_size=(432, 368))
  else:
    e = TfPoseEstimator(get_graph_path(model), target_size=(w, h))

  humans = e.inference(image, resize_to_default=(w > 0 and h > 0), upsample_size=4.0)
  poses = TfPoseEstimator.get_poses(image, humans, imgcopy=False)
  return poses

