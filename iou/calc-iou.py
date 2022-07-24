from pytorch3d.ops.iou_box3d import box3d_overlap

# from pytorch3d.ops.sample_farthest_points import

# Assume inputs: boxes1 (M, 8, 3) and boxes2 (N, 8, 3)
def calc_iou(boxes1, boxes2):
    intersection_vol, iou_3d = box3d_overlap(boxes1, boxes2)

boxes1 = []
boxes2 = []