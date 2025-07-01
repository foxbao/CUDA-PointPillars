import os
import numpy as np
import pyvista as pv
from glob import glob

class_names = {
    0: "Pedestrian",
    1: "Car",
    2: "IGV-Full",
    3: "Truck",
    4: "Trailer-Empty",
    5: "Trailer-Full",
    6: "IGV-Empty",
    7: "Crane",
    8: "OtherVehicle",
    9: "Cone",
    10: "ContainerForklift",
    11: "Forklift",
    12: "Lorry",
    13: "ConstructionVehicle",
    14: "WheelCrane"
}


def read_bin_file(bin_path):
    """读取点云 bin 文件"""
    point_cloud = np.fromfile(bin_path, dtype=np.float32).reshape(-1, 4)
    return point_cloud[:, :3]

def read_bboxes_from_txt(txt_path):
    """读取检测框：返回 Nx9 数组"""
    bboxes = []
    with open(txt_path, 'r') as f:
        for line in f:
            parts = list(map(float, line.strip().split()))
            bboxes.append(parts)
    return np.array(bboxes)

def create_bbox_lines(bbox):
    """构建立方体线框的 12 条边"""
    x, y, z = bbox[0:3]
    l, w, h = bbox[3:6]
    ry = bbox[6]

    # 构建局部坐标系下的 8 个角点
    corners = np.array([
        [ l/2,  w/2,  h/2],
        [ l/2, -w/2,  h/2],
        [-l/2, -w/2,  h/2],
        [-l/2,  w/2,  h/2],
        [ l/2,  w/2, -h/2],
        [ l/2, -w/2, -h/2],
        [-l/2, -w/2, -h/2],
        [-l/2,  w/2, -h/2],
    ])

    # 绕 Z 轴旋转
    R = np.array([
        [np.cos(ry), -np.sin(ry), 0],
        [np.sin(ry),  np.cos(ry), 0],
        [0, 0, 1]
    ])
    rotated = (R @ corners.T).T + np.array([x, y, z])

    # 定义 12 条边的索引
    edges = [
        [0,1], [1,2], [2,3], [3,0],  # 顶部
        [4,5], [5,6], [6,7], [7,4],  # 底部
        [0,4], [1,5], [2,6], [3,7]   # 竖线
    ]

    return rotated, edges

def visualize_point_cloud_and_boxes(bin_file, txt_file):
    # 加载点云
    points = read_bin_file(bin_file)
    pcd = pv.PolyData(points)

    plotter = pv.Plotter(window_size=(1600, 1000))
    plotter.set_background("black")
    plotter.add_points(pcd, color="white", point_size=2)

    # 加载检测框
    bboxes = read_bboxes_from_txt(txt_file)
    for bbox in bboxes:
        label = int(bbox[7])
        score = bbox[8]
        corners, edges = create_bbox_lines(bbox)
        for edge in edges:
            p0, p1 = corners[edge[0]], corners[edge[1]]
            plotter.add_lines(np.array([p0, p1]), color='green', width=2)

        # 添加文字
        text_pos = corners[0] + np.array([0, 0, 0.5])  # 抬高文字
        class_name = class_names.get(label, f"Class{label}")
        text = f"{class_name} {score:.2f}"
        # text = f"{label}:{score:.2f}"
        plotter.add_point_labels([text_pos], [text], text_color='green', font_size=12, point_size=0)

    plotter.show()

def traverse_and_visualize(data_dir):
    bin_files = sorted(glob(os.path.join(data_dir, "*.bin")))
    for bin_path in bin_files:
        base_name = os.path.splitext(os.path.basename(bin_path))[0]
        txt_path = os.path.join(data_dir, base_name + ".txt")
        if not os.path.exists(txt_path):
            print(f"Missing txt for {bin_path}")
            continue
        print(f"Visualizing: {bin_path}")
        visualize_point_cloud_and_boxes(bin_path, txt_path)

if __name__ == "__main__":
    data_dir = "data/data_20250507/lidar_merged"  # 改为你的路径
    # data_dir = "data/20250604"  # 改为你的路径
    traverse_and_visualize(data_dir)
