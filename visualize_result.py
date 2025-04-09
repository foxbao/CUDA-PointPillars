import numpy as np
import open3d as o3d
from matplotlib import cm

def read_bin_file(bin_path):
    """读取KITTI点云bin文件"""
    point_cloud = np.fromfile(bin_path, dtype=np.float32).reshape(-1, 4)
    return point_cloud[:, :3]  # 只取xyz坐标

def read_bboxes_from_txt(txt_path):
    """从txt文件读取3D检测框"""
    bboxes = []
    with open(txt_path, 'r') as f:
        for line in f:
            parts = list(map(float, line.strip().split()))
            bbox = parts[:7]  # 取前7个参数 (x,y,z,l,w,h,rotation_y)
            bboxes.append(bbox)
    return np.array(bboxes)

def create_3d_bbox(bbox_3d, color=(1, 0, 0)):
    """创建3D检测框线框"""
    center = bbox_3d[:3]
    length, width, height = bbox_3d[3:6]
    rotation = bbox_3d[6]
    
    # 计算8个角点
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
    
    # 应用旋转和平移
    # rotation=0
    rot_mat = np.array([
        [np.cos(rotation), -np.sin(rotation), 0],
        [np.sin(rotation), np.cos(rotation), 0],
        [0, 0, np.cos(rotation)]
    ])

    # rot_mat=np.ones(3,3)
    corners = np.dot(corners, rot_mat.T) + center
    
    # 定义12条边
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
    """可视化点云和3D检测框（黑色背景+白色点）"""
    # 读取数据
    points = read_bin_file(bin_path)
    bboxes = read_bboxes_from_txt(txt_path)
    
    # 创建点云并设置为白色
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(np.ones((points.shape[0], 3)))  # 所有点设为白色
    
    # 创建检测框（按置信度着色）

    bbox_geometries = []
    print('num boxes:', len(bboxes))
    for i, bbox in enumerate(bboxes):
        color = cm.plasma(i / len(bboxes))[:3]  # 框的颜色渐变
        bbox_geometries.append(create_3d_bbox(bbox, color=color))
    
    # 添加坐标系
    coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=2)
    
    # 可视化设置
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="KITTI Point Cloud", width=1200, height=800)
    vis.add_geometry(pcd)
    vis.add_geometry(coordinate_frame)
    for bbox in bbox_geometries:
        vis.add_geometry(bbox)
    
    # 设置渲染选项
    render_opt = vis.get_render_option()
    render_opt.background_color = np.array([0, 0, 0])  # 黑色背景
    render_opt.point_size = 2.0  # 增大点大小（默认1.0）
    render_opt.light_on = True   # 启用光照（增强白色点可见性）
    
    vis.run()
    vis.destroy_window()

if __name__ == "__main__":
    bin_file = "data/1733211963.001387.bin"
    txt_file = "data/1733211963.001387.txt"
    # bin_file = "data/000000.bin"
    # txt_file = "data/000000.txt"
    visualize_point_cloud_with_boxes(bin_file, txt_file)