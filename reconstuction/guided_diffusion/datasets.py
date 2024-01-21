
import random
import os

from PIL import Image
import blobfile as bf
import numpy as np
from torch.utils.data import DataLoader, Dataset

def load_data_yield(loader):
    while True:
        yield from loader

def load_data_inpa(
    *,
    gt_path=None,
    mask_path=None,
    batch_size,
    data_size,
    class_cond=False,
    deterministic=False,
    random_crop=False,
    random_flip=True,
    return_dataloader=False,
    return_dict=False,
    max_len=None,
    drop_last=True,
    conf=None,
    offset=0,
    ** kwargs
):
    """
    For a dataset, create a generator over (images, kwargs) pairs.

    Each images is an NCHW float tensor, and the kwargs dict contains zero or
    more keys, each of which map to a batched Tensor of their own.
    The kwargs dict can be used for class labels, in which case the key is "y"
    and the values are integer tensors of class labels.

    :param data_dir: a dataset directory.
    :param batch_size: the batch size of each returned pair.
    :param data_size: the size to which images are resized.
    :param class_cond: if True, include a "y" key in returned dicts for class
                       label. If classes are not available and this is true, an
                       exception will be raised.
    :param deterministic: if True, yield results in a deterministic order.
    :param random_crop: if True, randomly crop the images for augmentation.
    :param random_flip: if True, randomly flip the images for augmentation.
    """

    gt_dir = os.path.expanduser(gt_path)
    mask_dir = os.path.expanduser(mask_path)

    gt_paths = _list_image_files_recursively(gt_dir)
    mask_paths = _list_image_files_recursively(mask_dir)

    assert len(gt_paths) == len(mask_paths)

    classes = None
    if class_cond:
        raise NotImplementedError()

    dataset = ImageDatasetInpa(
        data_size,
        gt_paths=gt_paths,
        mask_paths=mask_paths,
        classes=classes,
        shard=0,
        num_shards=1,
        random_crop=random_crop,
        random_flip=random_flip,
        return_dict=return_dict,
        max_len=max_len,
        conf=conf,
        offset=offset
    )

    if deterministic:
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=False, num_workers=1, drop_last=drop_last
        )

    else:
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=True, num_workers=1, drop_last=drop_last
        )

    if return_dataloader:
        return loader
    else:
        return load_data_yield(loader)


def _list_image_files_recursively(data_dir):
    results = []
    for entry in sorted(bf.listdir(data_dir)):
        full_path = bf.join(data_dir, entry)
        ext = entry.split(".")[-1]
        if "." in entry and ext.lower() in ["npy"]:
            results.append(full_path)
        elif bf.isdir(full_path):
            results.extend(_list_image_files_recursively(full_path))
    return results


class ImageDatasetInpa(Dataset):
    def __init__(
        self,
        resolution,
        gt_paths,
        mask_paths,
        classes=None,
        shard=0,
        num_shards=1,
        random_crop=False,
        random_flip=True,
        return_dict=False,
        max_len=None,
        conf=None,
        offset=0
    ):
        super().__init__()
        self.resolution = resolution

        gt_paths = sorted(gt_paths)[offset:]
        mask_paths = sorted(mask_paths)[offset:]

        self.local_gts = gt_paths[shard:][::num_shards]
        self.local_masks = mask_paths[shard:][::num_shards]

        self.local_classes = None if classes is None else classes[shard:][::num_shards]

        self.random_crop = random_crop
        self.random_flip = random_flip
        self.return_dict = return_dict
        self.max_len = max_len

    def __len__(self):
        if self.max_len is not None:
            return self.max_len

        return len(self.local_gts)

    def __getitem__(self, idx):
        gt_path = self.local_gts[idx]
        arr_gt = self.npl(gt_path)

        mask_path = self.local_masks[idx]
        arr_mask = self.npl(mask_path)

        if self.random_flip and random.random() < 0.5:
            arr_gt = arr_gt[:, ::-1]
            arr_mask = arr_mask[:, ::-1]

        zheng = np.max(arr_gt)
        fu = np.min(arr_gt)
        guiyi = max(zheng, -fu)
        arr_gt = arr_gt.astype(np.float32) / guiyi
        arr_mask = arr_mask.astype(np.float32) / 255.0

        out_dict = {}
        if self.local_classes is not None:
            out_dict["y"] = np.array(self.local_classes[idx], dtype=np.int64)

        if self.return_dict:
            name = os.path.basename(gt_path)
            return {
                'GT': arr_gt[np.newaxis,:],
                'GT_name': name,
                'gt_keep_mask': arr_mask[np.newaxis,:],
                'guiyi': guiyi
            }
        else:
            raise NotImplementedError()

    def npl(self, path):
        with bf.BlobFile(path, "rb") as f:
            pil_image = np.load(f)
        return pil_image


def center_crop_arr(pil_image, data_size):
    # We are not on a new enough PIL to support the `reducing_gap`
    # argument, which uses BOX downsampling at powers of two first.
    # Thus, we do it by hand to improve downsample quality.
    while min(*pil_image.shape) >= 2 * data_size:
        pil_image.resize(
            tuple(x // 2 for x in pil_image.shape), refcheck=False
        )

    scale = data_size / min(*pil_image.shape)
    pil_image.resize(
        tuple(round(x * scale) for x in pil_image.shape), refcheck=False
    )

    arr = pil_image
    crop_y = (arr.shape[0] - data_size) // 2
    crop_x = (arr.shape[1] - data_size) // 2
    return arr[crop_y: crop_y + data_size, crop_x: crop_x + data_size]
