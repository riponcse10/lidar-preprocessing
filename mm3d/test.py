import pickle
import open3d.ml as ml3d

dataset = ml3d.datasets.KITTI("/media/ripon/Windows4/Users/ahrip/Documents/linux-soft/Kitti")

train_data = dataset.get_split("train")
item = train_data.get_data(0)

print(item.keys())

print(item['calib'])

