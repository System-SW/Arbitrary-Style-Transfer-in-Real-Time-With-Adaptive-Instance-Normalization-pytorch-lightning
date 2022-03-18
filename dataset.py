import os
from glob import glob

import albumentations as A
import numpy as np
import torch.utils.data as data
from PIL import Image, ImageFile
from torch.utils.data import DataLoader
from torchvision import transforms

included_ext = ["jpg", "jpeg", "bmp", "png"]
Image.MAX_IMAGE_PIXELS = None
ImageFile.LOAD_TRUNCATED_IMAGES = True


class FlatFolderDataset(data.Dataset):
    def __init__(self, image_paths, transform):
        super().__init__()
        self.paths = image_paths
        self.transform = transform

    def __getitem__(self, index):
        path = self.paths[index]
        image = Image.open(path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image

    def __len__(self):
        return len(self.paths)


def InfiniteSampler(n):
    # i = 0
    i = n - 1
    order = np.random.permutation(n)
    while True:
        yield order[i]
        i += 1
        if i >= n:
            np.random.seed()
            order = np.random.permutation(n)
            i = 0


class InfiniteSamplerWrapper(data.sampler.Sampler):
    def __init__(self, data_source):
        self.num_samples = len(data_source)

    def __iter__(self):
        return iter(InfiniteSampler(self.num_samples))

    def __len__(self):
        return 2 ** 31


def build_dataloader(args, root_dir: str) -> DataLoader:
    transform = transforms.Compose(
        [
            transforms.Resize(size=(512, 512)),
            transforms.RandomCrop(256),
            transforms.ToTensor(),
        ]
    )
    paths = glob(os.path.join(root_dir, "**"), recursive=True)
    paths = [
        path
        for path in paths
        if any(path.endswith(ext) for ext in included_ext)
    ]

    assert len(paths) != 0
    dataset = FlatFolderDataset(paths, transform)
    dataloader = DataLoader(
        dataset,
        args.batch_size,
        sampler=InfiniteSamplerWrapper(dataset),
        drop_last=True,
        num_workers=args.num_workers,
    )
    return dataloader
