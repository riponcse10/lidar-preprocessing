import os

import torch.utils.data
from torch.utils.data import Dataset
from torchvision.transforms import transforms

from utils import read_calib, read_label, read_points


class Dataset3D(Dataset):
    def __init__(self, root, split="training"):
        self.root = root
        self.split = split
        self.point_files = []
        self.label_files = []
        self.calib_files = []
        self.transform = transforms.Compose([transforms.ToTensor()])

        velo_files = os.listdir(os.path.join(self.root, self.split, "velodyne"))
        calibs = os.listdir(os.path.join(self.root, self.split, "calib"))
        labels = os.listdir(os.path.join(self.root, self.split, "label_2"))

        for i in range(len(velo_files)):
            if not velo_files[i].__contains__("bin"):
                print(velo_files[i])
            self.point_files.append(os.path.join(self.root, self.split, "velodyne", velo_files[i]))
            self.label_files.append(os.path.join(self.root, self.split, "label_2", labels[i]))
            self.calib_files.append(os.path.join(self.root, self.split, "calib", calibs[i]))

    def __getitem__(self, item):
        label = read_label(self.label_files[item])
        point = self.transform(read_points(self.point_files[item]))
        calib = read_calib(self.calib_files[item])

        datapoint = {"point": point, "label": label, "calib": calib}

        return datapoint

    def __len__(self):
        return len(self.point_files)


root = "/media/ripon/Windows4/Users/ahrip/Documents/linux-soft/Kitti"
split = "training"

dataset = Dataset3D(root, split)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True)

for data in dataloader:
    print(data)
    break