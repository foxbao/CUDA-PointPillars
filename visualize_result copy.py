import numpy as np
import open3d as o3d
from matplotlib import cm

def read_bin_file(bin_path):
    """读取KITTI点云bin文件"""
    point_cloud = np.fromfile(bin_path, dtype=np.float32).reshape(-1, 4)
    return point_cloud[:, :3]  # 只取xyz坐标

def read_bboxes_from_txt(txt_path):
    """
    从txt文件读取3D检测框
    格式: x y z l w h rotation_y class score
    """
    bboxes = []
    with open(txt_path, 'r') as f:
        for line in f:
            parts = list(map(float, line.strip().split()))
            bbox = parts[:7]  # 取前7个参数 (x,y,z,l,w,h,rotation_y)
            bboxes.append(bbox)
    return np.array(bboxes)

def create_3d_bbox(bbox_3d, color=(1, 0, 0)):
    """
    根据3D检测框参数创建Open3D线框
    bbox_3d格式: [x, y, z, l, w, h, rotation_y] (KITTI格式)
    """
    center = bbox_3d[:3]
    length, width, height = bbox_3d[3:6]
    rotation = bbox_3d[6]
    
    # 计算8个角点的局部坐标
    corners = np.array([
        [length/2, width/2, height/2],
        [length/2, width/2, -height/2],
        [length/2, -width/2, height/2],
        [length/2, -width/2, -height/2],
        [-length/2, width/2, height/2],
        [-length/2, width/2, -height/2],
        [-length/2, -width/2, height/2],
        [-length/2, -width/2, -height/2]
    ])
    
    # 应用旋转（绕Y轴）
    rot_mat = np.array([
        [np.cos(rotation), 0, np.sin(rotation)],
        [0, 1, 0],
        [-np.sin(rotation), 0, np.cos(rotation)]
    ])
    corners = np.dot(corners, rot_mat.T)
    
    # 平移至中心点
    corners += center
    
    # 定义12条边（连接8个角点）
    lines = [
        [0, 1], [0, 2], [1, 3], [2, 3],
        [4, 5], [4, 6], [5, 7], [6, 7],
        [0, 4], [1, 5], [2, 6], [3, 7]]
    
    # 创建线框
    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(corners)
    line_set.lines = o3d.utility.Vector2iVector(lines)
    line_set.colors = o3d.utility.Vector3dVector([color for _ in range(len(lines))])
    
    return line_set

def visualize_point_cloud_with_boxes(bin_path, txt_path):
    """可视化点云和3D检测框"""
    # 读取点云和检测框
    points = read_bin_file(bin_path)
    bboxes = read_bboxes_from_txt(txt_path)
    
    # 创建Open3D点云对象
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    
    # 点云着色（根据高度）
    z_min, z_max = np.min(points[:, 2]), np.max(points[:, 2])
    colors = cm.viridis((points[:, 2] - z_min) / (z_max - z_min))[:, :3]
    pcd.colors = o3d.utility.Vector3dVector(colors)
    
    # 创建3D检测框（按置信度score排序，高置信度用暖色）
    bbox_geometries = []
    for i, bbox in enumerate(bboxes):
        # 使用plasma配色，越靠后的框置信度越高（颜色越红）
        color = cm.plasma(i / len(bboxes))[:3]
        bbox_geometries.append(create_3d_bbox(bbox, color=color))
        if i==10:
            break
    
    # 添加坐标系
    coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=2)
    
    # # 可视化
    o3d.visualization.draw_geometries([pcd, coordinate_frame] + bbox_geometries,
                                    window_name="KITTI Point Cloud with 3D BBoxes",
                                    width=1200, height=800,
                                    left=50, top=50)
    # o3d.visualization.draw_geometries([pcd, coordinate_frame],
    #                                 window_name="KITTI Point Cloud with 3D BBoxes",
    #                                 width=1200, height=800,
    #                                 left=50, top=50)

# 示例使用
if __name__ == "__main__":
    # 替换为你的文件路径
    bin_file = "data/1733211963.001387.bin"
    txt_file = "data/1733211963.001387.txt"
    # bin_file = "data_kitti/000000.bin"
    # txt_file = "data_kitti/000000.txt"
    
    visualize_point_cloud_with_boxes(bin_file, txt_file)
