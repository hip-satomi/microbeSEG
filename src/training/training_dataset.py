import numpy as np
import tifffile as tiff
from torch.utils.data import Dataset


class TrainingDataset(Dataset):
    """ Pytorch data set for instance cell nuclei segmentation """

    def __init__(self, root_dir, label_type, mode='train', transform=lambda x: x):
        """

        :param root_dir: Directory containing all created training/validation data sets.
            :type root_dir: pathlib Path object.
        :param mode: 'train' or 'val'.
            :type mode: str
        :param transform: transforms.
            :type transform:
        :return: Dict (image, cell_label, border_label, id).
        """

        self.img_ids = sorted((root_dir / mode).glob('img*.tif'))
        self.mode = mode
        self.root_dir = root_dir
        self.transform = transform
        self.label_type = label_type

    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self, idx):

        img_id = self.img_ids[idx]

        img = tiff.imread(str(img_id))

        img = img[..., None]  # Channel dimension needed later (for pytorch)

        if self.label_type == 'distance':

            dist_label_id = img_id.parent / "cell_dist{}".format(img_id.name.split('img')[-1])
            dist_neighbor_label_id = img_id.parent / "neighbor_dist{}".format(img_id.name.split('img')[-1])

            dist_label = tiff.imread(str(dist_label_id)).astype(np.float32)
            dist_neighbor_label = tiff.imread(str(dist_neighbor_label_id)).astype(np.float32)

            dist_label = dist_label[..., None]
            dist_neighbor_label = dist_neighbor_label[..., None]

            sample = {'image': img,
                      'cell_label': dist_label,
                      'border_label': dist_neighbor_label,
                      'id': img_id.stem}

        elif self.label_type == 'boundary':

            label_id = img_id.parent / "{}{}".format(self.label_type, img_id.name.split('img')[-1])
            label = tiff.imread(str(label_id)).astype(np.uint8)
            label = label[..., None]
            sample = {'image': img, 'label': label, 'id': img_id.stem}

        sample = self.transform(sample)

        return sample
