import tempfile
import pytest
from easydict import EasyDict
import torch
import torch.nn as nn
import numpy as np

from models import Decoder, Encoder


@pytest.fixture(scope="session")
def args():
    return EasyDict(
        {
            "image_size": 256,
            "batch_size": 2,
            "encoder": "weights/vgg.pht",
            "onnx_opset_version": 11,
        }
    )


def tensor(batch_size, size):
    return torch.zeros([batch_size, 3, size, size])


def build_image(size):
    return np.zeros([size, size, 3], dtype=np.uint8)


@pytest.fixture(scope="session")
def batch(args):
    return {
        "content": tensor(args.batch_size, args.image_size),
        "style": tensor(args.batch_size, args.image_size),
    }


@pytest.fixture(scope="session")
def image_batch(args):
    return build_image(args.image_size)


@pytest.fixture(scope="session")
def decoder():
    return Decoder()


@pytest.fixture(scope="session")
def encoder():
    encoder = Encoder()
    encoder = nn.Sequential(*list(encoder.children())[:31])
    return encoder


@pytest.fixture(scope="session")
def temp_dir_f():
    return tempfile.TemporaryDirectory()
