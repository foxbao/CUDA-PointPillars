import os
import struct
import numpy as np
from tqdm import tqdm

def read_pcd(pcd_path):
    """读取PCD文件，支持ASCII和binary格式，返回Nx4的numpy数组（x, y, z, intensity）"""
    with open(pcd_path, 'rb') as f:
        lines = []
        while True:
            line = f.readline()
            if not line:
                raise ValueError(f"{pcd_path} 文件头读取失败")
            line_str = line.decode('utf-8').strip()
            lines.append(line_str)
            if line_str.startswith("DATA"):
                break

        header = "\n".join(lines)
        fields = []
        size = []
        count = []
        dtype = None

        for line in lines:
            if line.startswith("FIELDS"):
                fields = line.split()[1:]
            elif line.startswith("SIZE"):
                size = list(map(int, line.split()[1:]))
            elif line.startswith("COUNT"):
                count = list(map(int, line.split()[1:]))
            elif line.startswith("TYPE"):
                dtype = line.split()[1:]
            elif line.startswith("POINTS"):
                num_points = int(line.split()[1])
            elif line.startswith("DATA"):
                data_type = line.split()[1]

        if not all(f in fields for f in ['x', 'y', 'z', 'intensity']):
            raise ValueError(f"{pcd_path} 缺少 x/y/z/intensity 字段")

        idx_x = fields.index('x')
        idx_y = fields.index('y')
        idx_z = fields.index('z')
        idx_i = fields.index('intensity')

        if data_type == "ascii":
            data = np.loadtxt(f, dtype=np.float32)
        elif data_type == "binary":
            # 构造 struct 格式
            fmt = ''.join(
                f'{c}{"f" if t=="F" else "i"}' for c, t in zip(count, dtype)
            )
            point_size = sum(size)
            raw_data = f.read(num_points * point_size)
            unpacked = struct.iter_unpack(fmt, raw_data)
            data = np.array([list(p) for p in unpacked], dtype=np.float32)
        else:
            raise ValueError(f"不支持的数据格式: {data_type}")

        # 只保留 x y z intensity
        return data[:, [idx_x, idx_y, idx_z, idx_i]]

def process_pcd_folder(input_folder, output_folder):
    """批量转换PCD为BIN，保留x,y,z,intensity"""
    os.makedirs(output_folder, exist_ok=True)
    
    pcd_files = [f for f in os.listdir(input_folder) if f.endswith('.pcd')]
    pcd_files.sort()
    print(f"找到 {len(pcd_files)} 个PCD文件需要转换")

    for pcd_file in tqdm(pcd_files, desc="转换进度"):
        try:
            pcd_path = os.path.join(input_folder, pcd_file)
            points = read_pcd(pcd_path)  # Nx4

            bin_file = os.path.splitext(pcd_file)[0] + '.bin'
            bin_path = os.path.join(output_folder, bin_file)
            points.astype(np.float32).tofile(bin_path)
        except Exception as e:
            print(f"处理 {pcd_file} 失败: {e}")

if __name__ == "__main__":
    
    # 配置路径（请修改为实际路径）
    # input_folder = "pcd_test_pointpillar_helios"  # 输入PCD文件夹
    # output_folder = "converted_bin_files"         # 输出BIN文件夹
    
    input_folder = "data/test_data/002/20250604"
    output_folder = "data/test_data/002/20250604_bin"

    process_pcd_folder(input_folder, output_folder)

    print(f"\n转换完成！结果已保存到 {output_folder} 文件夹")
    print(f"生成 {len([f for f in os.listdir(output_folder) if f.endswith('.bin')])} 个BIN文件")
