import os
import io
import mxnet as mx
from PIL import Image 
from tqdm import tqdm  

rec_file = "casia-webface/train.rec"
idx_file = "casia-webface/train.idx"
output_dir = "extracted_images"

os.makedirs(output_dir, exist_ok=True)

record = mx.recordio.MXIndexedRecordIO(idx_file, rec_file, 'r')

class_counter = {}
total_images = 0



idx = 1
error_count = 0

while True:
    try:
        item = record.read_idx(idx)
        if item is None:
            break
            
        header, raw_img = mx.recordio.unpack(item)
        
        if hasattr(header.label, '__len__'):
            label = int(header.label[0])
        else:
            label = int(header.label)
        
        class_folder = os.path.join(output_dir, f"class_{label:06d}")  
        os.makedirs(class_folder, exist_ok=True)
        
        if label not in class_counter:
            class_counter[label] = 0
        class_counter[label] += 1
        
        img = Image.open(io.BytesIO(raw_img))
        
        img_path = os.path.join(class_folder, f"image_{class_counter[label]:06d}.jpg")
        img.save(img_path)
        
        total_images += 1
        
        if total_images % 1000 == 0:
            print(f"total : {total_images}")
        
        idx += 1
        
    except Exception as e:
        print(f"error at index : {idx} -> {e}")
        error_count += 1
        idx += 1
        
        if error_count > 100:
            break

