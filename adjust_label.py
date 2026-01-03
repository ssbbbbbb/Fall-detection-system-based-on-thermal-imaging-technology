import os
import glob

def process_files(small_box_file, unadjusted_file, output_file):
    with open(small_box_file, 'r', encoding='utf-8') as f1, \
         open(unadjusted_file, 'r', encoding='utf-8') as f2, \
         open(output_file, 'w', encoding='utf-8') as out_f:
        
        for line1, line2 in zip(f1, f2):
            # Read data from the first file (new bounding box coordinates)
            line1 = line1.strip()
            if not line1:
                continue
            new_data = list(map(float, line1.split()))
            if len(new_data) < 4:
                print(f"Insufficient data in the first file: {line1}")
                continue
            nx_cx, nx_cy, nx_w, nx_h = new_data[:4]
            
            # Read data from the second file (old bounding box and keypoint coordinates)
            line2 = line2.strip()
            if not line2:
                continue
            old_data = list(map(float, line2.split()))
            if len(old_data) < 5 + 34:  # Class ID + old box coordinates + 17 keypoints (x, y)
                print(f"Insufficient data in the second file: {line2}")
                continue
            class_id = int(old_data[0])
            ox_cx, ox_cy, ox_w, ox_h = old_data[1:5]
            points = old_data[5:]
            
            # Adjust keypoint coordinates
            adjusted_points = []
            for i in range(0, len(points), 2):
                xi = points[i]
                yi = points[i+1]
                
                # Calculate relative position within the old box
                x_rel = (xi - (ox_cx - ox_w / 2)) / ox_w
                y_rel = (yi - (ox_cy - ox_h / 2)) / ox_h
                
                # Calculate new position based on the new box
                xi_new = (nx_cx - nx_w / 2) + x_rel * nx_w
                yi_new = (nx_cy - nx_h / 2) + y_rel * nx_h
                adjusted_points.extend([xi_new, yi_new])
                
            # Write updated data to the output file
            output_line = [str(class_id), str(nx_cx), str(nx_cy), str(nx_w), str(nx_h)] + [str(p) for p in adjusted_points]
            out_f.write(' '.join(output_line) + '\n')

def batch_process(small_box_dir, unadjusted_dir, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    # Get all box files from the shrunken mask directory
    small_box_files = glob.glob(os.path.join(small_box_dir, '*.txt'))
    
    for small_box_file in small_box_files:
        base_name = os.path.basename(small_box_file)
        # Assuming the unadjusted txt file shares the same name as the shrunken box file
        unadjusted_file = os.path.join(unadjusted_dir, base_name)
        
        if not os.path.exists(unadjusted_file):
            print(f"Corresponding unadjusted file not found: {unadjusted_file}")
            continue
            
        output_file = os.path.join(output_dir, base_name)
        process_files(small_box_file, unadjusted_file, output_file)
        print(f"Processed file: {base_name}")

# Example usage
# Updated paths to use English naming conventions
small_box_dir = r"E:\FOLDER\yolo\dataset\thesis\COMPOSITE\boxes"  # Directory for shrunken mask boxes
unadjusted_dir = r"E:\FOLDER\yolo\dataset\thesis\RGB\unadjusted_txt"  # Directory for unadjusted txt files
output_dir = r"E:\FOLDER\yolo\dataset\thesis\txt_output"               # Output directory

batch_process(small_box_dir, unadjusted_dir, output_dir)