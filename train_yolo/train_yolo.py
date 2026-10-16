import cv2
import numpy as np
import os
import glob
import shutil
from ultralytics import YOLO

def augment_dataset_with_opencv(dataset_path):
    """
    自動巡迴 train 資料夾，利用 OpenCV 幫照片注入局部強光與對比度增強，
    並自動複製對應的 YOLO 標註文字檔 (.txt)。
    """
    img_dir = os.path.join(dataset_path, "train", "images")
    lbl_dir = os.path.join(dataset_path, "train", "labels")
    
    # 🧹【防禦：清除上一次殘留的加工舊檔】
    print("🧹 正在清除上一次訓練留下的舊加工檔案，確保資料集純淨...")
    old_aug_imgs = glob.glob(os.path.join(img_dir, "*_aug_*"))
    old_aug_lbls = glob.glob(os.path.join(lbl_dir, "*_aug_*"))
    for f in old_aug_imgs: os.remove(f)
    for f in old_aug_lbls: os.remove(f)
    
    print("⏳ [OpenCV 強化大招啟動] 正在為最新資料集進行極限抗強光特訓加工...")
    
    # 鎖定當前的原始圖片
    all_files = glob.glob(os.path.join(img_dir, "*.jpg")) + glob.glob(os.path.join(img_dir, "*.png"))
    img_paths = [f for f in all_files if "_aug_" not in f]
    
    augmented_count = 0
    print(f" 找到 {len(img_paths)} 張原始種子照片（僅包含 TEL 標記），開始進行 3 倍量擴增...")
    
    for img_path in img_paths:
        base_name = os.path.splitext(os.path.basename(img_path))[0]
        ext = os.path.splitext(img_path)[1]
        
        lbl_path = os.path.join(lbl_dir, f"{base_name}.txt")
        if not os.path.exists(lbl_path):
            continue  
            
        img = cv2.imread(img_path)
        if img is None:
            continue
            
        h, w, _ = img.shape
        
        # --- 強化手法 1：模擬局部反光 ---
        flash_mask = np.zeros((h, w, 3), dtype=np.uint8)
        cx = np.random.randint(int(w * 0.1), int(w * 0.9))
        cy = np.random.randint(int(h * 0.1), int(h * 0.5))
        radius = np.random.randint(60, 100)
        cv2.circle(flash_mask, (cx, cy), radius, (180, 180, 180), -1)
        flash_mask = cv2.GaussianBlur(flash_mask, (99, 99), 0)
        img_flash = cv2.add(img, flash_mask)
        
        flash_img_name = f"{base_name}_aug_flash{ext}"
        cv2.imwrite(os.path.join(img_dir, flash_img_name), img_flash)
        shutil.copy(lbl_path, os.path.join(lbl_dir, f"{base_name}_aug_flash.txt"))
        
        # --- 強化手法 2：CLAHE 對比度極限增強 ---
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8,8))
        cl = clahe.apply(l)
        limg = cv2.merge((cl,a,b))
        img_clahe = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
        
        clahe_img_name = f"{base_name}_aug_clahe{ext}"
        cv2.imwrite(os.path.join(img_dir, clahe_img_name), img_clahe)
        shutil.copy(lbl_path, os.path.join(lbl_dir, f"{base_name}_aug_clahe.txt"))
        
        augmented_count += 2

    print(f"✨ OpenCV 資料強化完成！成功自動生成了 {augmented_count} 張抗反光照片。")


if __name__ == '__main__':
    # 🎯 你的 Roboflow 資料集根目錄路徑
    dataset_root = r'C:\Users\candy\OneDrive\Desktop\TEL_Robot_Vision\TEL.yolo26'
    
    # 1. 執行具有防禦力的自動改造
    augment_dataset_with_opencv(dataset_root)
    
    # 2. 確定有正統的 data.yaml 路徑
    yaml_path = os.path.join(dataset_root, 'data.yaml')
    
    # 3. 載入官方預訓練權重（自動下載標準輕量模型）
    # 建議：yolov8n.pt 或是最新的 yolo11n.pt 
    model = YOLO('yolo26n.pt') 

    # 4. 啟動最終特訓
    model.train(
        data=yaml_path,      
        epochs=100,            
        imgsz=640,            
        batch=8,              
        device=0              # 全力壓榨 NVIDIA 顯示卡！
    )
    
    print("✨ 北伐成功！專屬你的 TEL 辨識模型已就位 Ciallo～(∠·ω<)⌒☆")