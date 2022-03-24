import os
import numpy as np
import cv2

TITLE = (
    "Arbitrary Style Transfer in Real-time"
    "with Adaptive Instance Normalization"
)

DESC = "Unofficial pytorch-lightning implement"

AUTHOR = "YSLEE(rapidrabbit76)"

URL = (
    "https://github.com/rapidrabbit76/"
    "Arbitrary-Style-Transfer-in-Real-Time-"
    "With-Adaptive-Instance-Normalization-pytorch-lightning"
)

ENCODER_PATH = os.environ.get("ENCODER_PATH")
DECODER_PATH = os.environ.get("DECODER_PATH")

EXT = ["png", "jpeg", "jpg"]


def image_load(file):
    image = cv2.imdecode(np.asarray(bytearray(file.read()), dtype=np.uint8), 1)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image
