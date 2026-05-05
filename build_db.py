import os
import torch
import faiss
import numpy as np
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
import random

# 1. Cấu hình đường dẫn
IMAGE_DIR = "images_hd"       
INDEX_FILE = "fashion_clip.index" 
PATHS_FILE = "image_paths.npy"
MAX_IMAGES = 4000            

print("Đang khởi động lõi AI (CLIP Model)...")
device = "cuda" if torch.cuda.is_available() else "cpu"
model_id = "openai/clip-vit-base-patch32"
model = CLIPModel.from_pretrained(model_id).to(device)
processor = CLIPProcessor.from_pretrained(model_id)

# 2. Quét thư mục ảnh
supported_formats = (".jpg", ".jpeg", ".png")
all_images = [
    os.path.join(IMAGE_DIR, f) for f in os.listdir(IMAGE_DIR) 
    if f.lower().endswith(supported_formats)
]

# Xáo trộn ngẫu nhiên 
random.shuffle(all_images)
image_paths = all_images[:MAX_IMAGES]

print(f"Đã chọn ngẫu nhiên {len(image_paths)} hình ảnh để xử lý.")
print("Bắt đầu trích xuất đặc trưng (Feature Extraction)...")

embeddings = []
valid_paths = []

# 3. Trích xuất Vector
for i, path in enumerate(image_paths):
    try:
        img = Image.open(path).convert("RGB")
        inputs = processor(images=img, return_tensors="pt").to(device)
        
        with torch.no_grad():
            outputs = model.vision_model(**inputs)
            pooled = outputs.pooler_output
            proj = model.visual_projection
            
            if isinstance(proj, torch.nn.Module):
                embeds = proj(pooled)
            else:
                embeds = torch.matmul(pooled, proj)
                
            # Chuẩn hóa Vector
            embeds = embeds / embeds.norm(p=2, dim=-1, keepdim=True)
            embeddings.append(embeds.cpu().numpy().flatten())
            valid_paths.append(path)
            
        if (i + 1) % 50 == 0:
            print(f"   Đã hoàn thành {i + 1}/{len(image_paths)} ảnh...")
            
    except Exception as e:
        print(f"Bỏ qua file lỗi {path}: {e}")

# 4. Lưu vào FAISS
if embeddings:
    print("Đang ghi cơ sở dữ liệu Vector (FAISS)...")
    dimension = 512
    index = faiss.IndexFlatIP(dimension)
    index.add(np.array(embeddings).astype('float32'))
    
    faiss.write_index(index, INDEX_FILE)
    np.save(PATHS_FILE, np.array(valid_paths))
    print("Cập nhật Dataset thành công.")
else:
    print("Lỗi: Không có ảnh nào được xử lý.")