import os
import numpy as np
import open3d as o3d
from tqdm import tqdm  # 进度条工具

def process_pcd_folder(input_folder, output_folder):
    """批量处理PCD文件夹"""
    os.makedirs(output_folder, exist_ok=True)
    
    pcd_files = [f for f in os.listdir(input_folder) if f.endswith('.pcd')]
    pcd_files.sort()
    print(f"找到 {len(pcd_files)} 个PCD文件需要转换")
    
    for pcd_file in tqdm(pcd_files, desc="转换进度"):
        # 读取数据
        pcd_path = os.path.join(input_folder, pcd_file)

        points = np.loadtxt(pcd_path, skiprows=11)
        # xyzi_data = read_pcd_with_intensity(pcd_path)
        
        # 保存为bin文件（同名）
        bin_file = os.path.splitext(pcd_file)[0] + '.bin'
        bin_path = os.path.join(output_folder, bin_file)
        points.astype(np.float32).tofile(bin_path)

if __name__ == "__main__":
    # 配置路径（请修改为实际路径）
    input_folder = "pcd_test_pointpillar_helios"  # 输入PCD文件夹
    output_folder = "converted_bin_files"         # 输出BIN文件夹
    
    # 执行转换
    process_pcd_folder(input_folder, output_folder)
    
    # 打印结果摘要
    print(f"\n转换完成！结果已保存到 {output_folder} 文件夹")
    print(f"生成 {len(os.listdir(output_folder))} 个BIN文件")