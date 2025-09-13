import os
import glob

def process_files(small_box_file, unadjusted_file, output_file):
    with open(small_box_file, 'r', encoding='utf-8') as f1, open(unadjusted_file, 'r', encoding='utf-8') as f2, open(output_file, 'w', encoding='utf-8') as out_f:
        for line1, line2 in zip(f1, f2):
            # 读取第一个文件的数据（新的框坐标）
            line1 = line1.strip()
            if not line1:
                continue
            new_data = list(map(float, line1.split()))
            if len(new_data) < 4:
                print(f"第一个文件的数据不足：{line1}")
                continue
            nx_cx, nx_cy, nx_w, nx_h = new_data[:4]
            
            # 读取第二个文件的数据（旧的框坐标和点坐标）
            line2 = line2.strip()
            if not line2:
                continue
            old_data = list(map(float, line2.split()))
            if len(old_data) < 5 + 34:  # 类别 + 旧的框坐标 + 17个点的x和y
                print(f"第二个文件的数据不足：{line2}")
                continue
            class_id = int(old_data[0])
            ox_cx, ox_cy, ox_w, ox_h = old_data[1:5]
            points = old_data[5:]
            
            # 调整点坐标
            adjusted_points = []
            for i in range(0, len(points), 2):
                xi = points[i]
                yi = points[i+1]
                # 计算相对位置
                x_rel = (xi - (ox_cx - ox_w / 2)) / ox_w
                y_rel = (yi - (ox_cy - ox_h / 2)) / ox_h
                # 计算新的位置
                xi_new = (nx_cx - nx_w / 2) + x_rel * nx_w
                yi_new = (nx_cy - nx_h / 2) + y_rel * nx_h
                adjusted_points.extend([xi_new, yi_new])
            # 写入更新后的数据
            output_line = [str(class_id), str(nx_cx), str(nx_cy), str(nx_w), str(nx_h)] + [str(p) for p in adjusted_points]
            out_f.write(' '.join(output_line) + '\n')

def batch_process(small_box_dir, unadjusted_dir, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    # 获取所有的縮小遮罩的box文件
    small_box_files = glob.glob(os.path.join(small_box_dir, '*.txt'))
    for small_box_file in small_box_files:
        base_name = os.path.basename(small_box_file)
        # 假设未调整的txt文件与缩小遮罩的box文件具有相同的文件名
        unadjusted_file = os.path.join(unadjusted_dir, base_name)
        if not os.path.exists(unadjusted_file):
            print(f"未找到对应的未调整文件：{unadjusted_file}")
            continue
        output_file = os.path.join(output_dir, base_name)
        process_files(small_box_file, unadjusted_file, output_file)
        print(f"已处理文件：{base_name}")

# 示例用法
small_box_dir = r"E:\FOLDER\yolo\資料集\論文\COMPOSITE\boxes"  # 縮小遮罩的box文件夹
unadjusted_dir = r"E:\FOLDER\yolo\資料集\論文\RGB\txt未調整"    # 未調整的txt文件夹
output_dir = r"E:\FOLDER\yolo\資料集\論文\txt"              # 输出文件夹

batch_process(small_box_dir, unadjusted_dir, output_dir)
