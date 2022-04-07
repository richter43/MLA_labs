import os
import errno
import shutil
import random

import cv2
from matplotlib import pyplot as plt

import numpy as np
import pandas as pd
import seaborn as sb
import albumentations as A

import utils


def main():
    cur_path = os.path.join(os.path.abspath(os.curdir))
    old_images_root = os.path.join(cur_path, "Vane_Roughness", "data")

    if not os.path.isdir(old_images_root):

    new_images_root = os.path.join(cur_path, "data")

    image_classes = ["ok", "nok", "doubt"]

    path_dict = {'ok': os.path.join(new_images_root, "ok"), 'nok': os.path.join(new_images_root, "nok"),
                 'doubt': os.path.join(new_images_root, "doubt")}

    for new_path in path_dict.values():
        utils.makedirs(new_path)

    image_names = os.listdir(old_images_root)

    for image_name in image_names:
        image_path = os.path.join(old_images_root, image_name)
        if image_name.startswith("OK"):
            shutil.move(image_path, path_dict['ok'])
        elif image_name.startswith("NOK"):
            shutil.move(image_path, path_dict['nok'])
        else:
            shutil.move(image_path, path_dict['doubt'])


if __name__ == "__main__":
    # main()
    print(os.path.join(os.path.abspath(os.curdir)))
