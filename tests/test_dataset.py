from dataset import Transforms


def test_transforms(args, image_batch):
    T = Transforms(args.image_size)
    batch = T(image_batch)
    assert list(batch.shape) == [3, args.image_size, args.image_size]
