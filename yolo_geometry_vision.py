import cv2
import torch
import numpy as np
from ultralytics import YOLO

# 💡 統一在這裡控制相機編號：0=內建, 1=手機Iriun, 2=外接
cap = cv2.VideoCapture(1, cv2.CAP_DSHOW) 

def main():
    # 載入最佳 YOLO 模型權重
    model = YOLO('runs/detect/train/weights/best.pt') 

    print("【YOLO26 東京威力即時辨識】偵測視窗啟動中...")

    if not cap.isOpened():
        print("❌ 錯誤：找不到指定編號的相機。")
        return

    false_ret_count = 0

    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            false_ret_count += 1
            if false_ret_count > 30:
                print("❌ 錯誤：相機訊號中斷。")
                break
            cv2.waitKey(10)
            continue
        
        false_ret_count = 0

        try:
            # 🔥【CLAHE 抗光增強】
            lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
            cl = clahe.apply(l)
            limg = cv2.merge((cl,a,b))
            enhanced_frame = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)

            # 將 conf 稍微調低到 0.12，逼 YOLO 把所有看過的東西都認出來
            results = model.predict(enhanced_frame, conf=0.12, device=0, verbose=False)

            all_detected_points = []

            # 🛠️ 遍歷 YOLO 抓到的所有東西，強制畫出原生的框和名稱
            for result in results:
                boxes = result.boxes
                for box in boxes:
                    cls_id = int(box.cls[0].item())  # 類別 ID
                    conf_score = box.conf[0].item()  # 信心度
                    
                    # 取得框框邊界
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                    # 🎯 【精準計算中心點】這會保證點死死釘在被框住物件的「幾何幾何中心」上
                    cx = int((x1 + x2) / 2)
                    cy = int((y1 + y2) / 2)
        
                    # 濾除高度重複的雜訊點
                    if all(np.linalg.norm(np.array([cx, cy]) - np.array(existing)) > 20 for existing in all_detected_points):
                        all_detected_points.append((cx, cy))
                        
                        # 從模型的 names 字典動態取得你當初在 Roboflow 標記的類別名稱
                        class_name = model.names[cls_id]
                        
                        # 🎨 1. 畫出藍色原生偵測框 (不管它是什麼類別，通通現形！)
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 128, 0), 2)
                        
                        # 🔴 2. 在物理中心點畫上紅點
                        cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)
                        
                        # 🏷️ 3. 顯示類別標籤與信心度（例如 big hole 85%）
                        label_text = f"{class_name} {conf_score:.2f}"
                        cv2.putText(frame, label_text, (x1, y1 - 8),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)

            # 🎯【V2.1 幾何多點網格吸附反推 12 個洞口】
            # 當大腦訓練好後，抓到的總點數會大於 3
            if len(all_detected_points) >= 3:
                frame_width = frame.shape[1]
                x_values = [pt[0] for pt in all_detected_points]
                y_values = [pt[1] for pt in all_detected_points]
                
                # 推算平均欄寬 (過濾掉靠太近的點)
                sorted_x = sorted(x_values)
                distances_x = [sorted_x[i+1] - sorted_x[i] for i in range(len(sorted_x)-1)]
                valid_distances = [d for d in distances_x if d > (frame_width * 0.12)]
                
                avg_col_width = np.mean(valid_distances) if len(valid_distances) > 0 else frame_width * 0.26
                
                # 計算中心基準點
                mass_center_x = np.mean(x_values)
                mass_center_y = np.mean(sorted(y_values)[:max(1, len(y_values)//2)])

                # 相對比例展開 (微調了偏移矩陣，使其更加貼合)
                col_indices = [-1.35, -0.3, 0.75, 1.8]
                row_indices = [2.0, 4.4, 6.8]

                for r_idx, r_ind in enumerate(row_indices):
                    for c_idx, c_ind in enumerate(col_indices):
                        hole_x = int(mass_center_x + avg_col_width * c_ind)
                        hole_y = int(mass_center_y + avg_col_width * r_ind) 

                        # 🟢 畫出綠色模擬目標投球點（12洞）
                        cv2.circle(frame, (hole_x, hole_y), 4, (0, 255, 0), -1)
                        cv2.putText(frame, f"R{r_idx+1}-C{c_idx+1}", (hole_x - 15, hole_y - 8),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 0), 1)

            # 頂部狀態列
            cv2.putText(frame, f"YOLO Total Detected Points: {len(all_detected_points)}", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

            cv2.imshow("TEL Robot - YOLO26 AI Vision", frame)

        except Exception as e:
            print(f"⚠️ 畫面處理出錯: {e}")

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()