import numpy as np
import open3d as o3d
from matplotlib import cm
import os

def read_kitti_bin(bin_path):
    """读取KITTI格式的bin文件（xyzi）"""
    data = np.fromfile(bin_path, dtype=np.float32).reshape(-1, 4)
    return data

def visualize_point_cloud(bin_path):
    """可视化点云文件（带强度颜色映射）"""
    # 读取数据
    xyzi = read_kitti_bin(bin_path)
    points = xyzi[:, :3]
    intensity = xyzi[:, 3]
    
    # 创建Open3D点云对象
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    
    # 强度值归一化并映射为颜色
    intensity_normalized = (intensity - np.min(intensity)) / (np.max(intensity) - np.min(intensity) + 1e-6)
    colors = cm.plasma(intensity_normalized)[:, :3]  # 使用plasma配色
    pcd.colors = o3d.utility.Vector3dVector(colors)
    
    # 添加坐标系参考
    coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=2.0)
    
    # 可视化设置
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name=f"Point Cloud: {os.path.basename(bin_path)}", 
                     width=1200, height=800)
    vis.add_geometry(pcd)
    vis.add_geometry(coordinate_frame)
    
    # 设置渲染选项
    opt = vis.get_render_option()
    opt.point_size = 1.5  # 点大小
    opt.background_color = np.asarray([0.1, 0.1, 0.1])  # 深色背景
    
    # 运行可视化
    vis.run()
    vis.destroy_window()

def visualize_folder(bin_folder):
    """可视化文件夹中的所有bin文件"""
    bin_files = [f for f in os.listdir(bin_folder) if f.endswith('.bin')]
    print(f"找到 {len(bin_files)} 个BIN文件")
    
    for bin_file in bin_files:
        bin_path = os.path.join(bin_folder, bin_file)
        print(f"\n正在可视化: {bin_file}")
        visualize_point_cloud(bin_path)
        input("按Enter继续查看下一个文件...")  # 暂停控制

if __name__ == "__main__":
    # 配置路径（指向之前转换的bin文件夹）
    bin_folder = "converted_bin_files"  # 替换为你的bin文件夹路径
    
    # 选择模式：单个文件或整个文件夹
    mode = input("请选择模式：\n1. 可视化单个文件\n2. 可视化整个文件夹\n输入选项(1/2): ")
    
    if mode == "1":
        # 可视化单个文件
        bin_file = input("请输入文件名（例如：000001.bin）: ")
        bin_path = os.path.join(bin_folder, bin_file)
        if os.path.exists(bin_path):
            visualize_point_cloud(bin_path)
        else:
            print(f"文件 {bin_path} 不存在！")
    elif mode == "2":
        # 批量可视化
        visualize_folder(bin_folder)
    else:
        print("无效选项！")