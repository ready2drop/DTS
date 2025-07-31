from typing import Optional

import nibabel
from tqdm import tqdm

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from monai import transforms


class BTCVDataset(Dataset):
    def __init__(self,
                 data_list: list,
                 image_size: int = 256,
                 spatial_size: int = 96,
                 pad: int = 2,
                 padding: bool = True,
                 transform: transforms = None,
                 data_path: Optional[str] = None,
                 mode: Optional[str] = "train",
                 use_cache: Optional[bool] = True) -> None:
        super().__init__()

        self.transform = transform
        self.data_list = data_list
        self.image_size = image_size
        self.spatial_size = spatial_size
        self.padding = padding
        self.data_path = data_path
        self.mode = mode
        self.use_cache = use_cache

        self.pad = (pad, pad)

        assert mode != "train" or  mode != "val" or  mode != "test", \
            "Key must be one of these keywords : train / val / test"

        self.key = "Tr" if mode == "train" else "Va"
        self.resize = transforms.Compose([transforms.Resize((spatial_size, image_size, image_size))])
        self.cache = {}

        if use_cache:
            print("Caching....")
            self.save_cache()

    def save_cache(self):
        for d in tqdm(self.data_list):
            _ = self.read_data(d)

    def read_data(self, data_path):
        if data_path[0] in self.cache.keys():
            return self.cache[data_path[0]]
        else:
            image_path = data_path[0]
            label_path = data_path[1]

            image = nibabel.load(image_path).get_fdata()
            label = nibabel.load(label_path).get_fdata()
            raw_label = nibabel.load(label_path).get_fdata()

            image = torch.tensor(image)
            label = torch.tensor(label)
            raw_label = torch.tensor(raw_label)

            image = F.pad(image, self.pad, "constant", 0)
            label = F.pad(label, self.pad, "constant", 0)

            # (H, W, D) -> (D, W, H)
            image = torch.transpose(image, 0, 2).contiguous()
            label = torch.transpose(label, 0, 2).contiguous()
            raw_label = torch.transpose(raw_label, 0, 2).contiguous()

            image = image.unsqueeze(0)
            label = label.unsqueeze(0)
            raw_label = raw_label.unsqueeze(0)

            self.cache[data_path[0]] = {
                "image": image,
                "label": label
            }

            if self.mode == "test": self.cache[data_path[0]]["raw_label"] = raw_label

            return self.cache[data_path[0]]

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, i):
        data = self.read_data(self.data_list[i])

        if self.transform is not None:
            data = self.transform(data)

        return data, self.data_list[i][0]



