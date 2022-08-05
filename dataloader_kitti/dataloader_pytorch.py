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
        point = (read_points(self.point_files[item]))
        calib = read_calib(self.calib_files[item])

        datapoint = {"point": point, "label": label, "calib": calib}

        return datapoint

    def __len__(self):
        return len(self.point_files)


def collate(list_data):
    batched_pts_list, batched_gt_bboxes_list = [], []
    batched_labels_list, batched_names_list = [], []
    batched_difficulty_list = []
    batched_img_list, batched_calib_list = [], []

    for data_dict in list_data:
        point = data_dict['point']
        label = data_dict['label']
        calib = data_dict['calib']

        batched_pts_list.append(torch.from_numpy(point))
        batched_gt_bboxes_list.append(torch.from_numpy(label['bbox']))
        batched_labels_list.append(label['name'])
        # batched_names_list.append(gt_names)  # List(str)
        # batched_difficulty_list.append(torch.from_numpy(difficulty))
        # batched_img_list.append(image_info)
        # batched_calib_list.append(calbi_info)

    rt_data_dict = dict(
        batched_pts=batched_pts_list,
        batched_gt_bboxes=batched_gt_bboxes_list,
        batched_labels=batched_labels_list
        # batched_names=batched_names_list,
        # batched_difficulty=batched_difficulty_list,
        # batched_img_info=batched_img_list,
        # batched_calib_info=batched_calib_list
    )

    return rt_data_dict

root = "/media/ripon/Windows4/Users/ahrip/Documents/linux-soft/Kitti"
split = "training"

dataset = Dataset3D(root, split)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=2, shuffle=True, collate_fn=collate)

for data in dataloader:
    print(data)
    break