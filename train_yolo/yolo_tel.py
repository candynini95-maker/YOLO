import cv2
import numpy as np
from ultralytics import YOLO

cap = cv2.VideoCapture(1, cv2.CAP_DSHOW) 

def main():
    # 🎯 載入你最準、只認得 "TEL" 的第七代單類別模型
    model = YOLO('runs/detect/train-7/weights/best.pt') 
    print("【YOLO26 純TEL投影 + 信心值顯示】啟動中...")

    # 📐 每一排洞口距離 TEL 中心，相當於「幾個 TEL 框的高度(h)」
    # 記得用上一動我們討論過的修正值（往上提拉一整格）
    row_ratios = [2.0, 6.0, 10.0, 14.0] 

    if not cap.isOpened(): 
        print("❌ 找不到相機")
        return

    while True:
        ret, frame = cap.read()
        if not ret or frame is None: continue
        
        try:
            results = model.predict(frame, conf=0.6, device=0, verbose=False)

            tels = []
            
            # 1. 收集所有 TEL 的中心點、框高度以及【信心值】
            for box in results[0].boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                cx = int((x1 + x2) / 2)
                cy = int((y1 + y2) / 2)
                h = y2 - y1  # 取得這個 TEL 框的高度
                
                # 🔴 【新增】取出當前框的信心值
                conf = box.conf[0].item()
                
                tels.append({'cx': cx, 'cy': cy, 'h': h, 'box': (x1, y1, x2, y2), 'conf': conf})

            # 2. 🎯 當剛好偵測到 3 個 TEL 時，啟動無敵幾何投影
            if len(tels) == 3:
                # 依 X 座標由左至右排序 (左、中、右)
                tels.sort(key=lambda x: x['cx'])
                
                # 計算板子的水平傾斜向量
                dx = tels[2]['cx'] - tels[0]['cx']
                dy = tels[2]['cy'] - tels[0]['cy']
                board_width_pixels = np.sqrt(dx**2 + dy**2)
                
                # 將水平向量標準化
                ux = dx / board_width_pixels
                uy = dy / board_width_pixels
                
                # 旋轉 90 度得到向下向量
                down_x = -uy
                down_y = ux

                # 3. 畫出原生的 TEL 偵測框與【顯示信心值】
                for t in tels:
                    x1, y1, x2, y2 = t['box']
                    conf_val = t['conf']
                    
                    # 🔵 畫藍色框框與中心紅點
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 128, 0), 2)
                    cv2.circle(frame, (t['cx'], t['cy']), 4, (0, 0, 255), -1)
                    
                    # 🏷️ 【新增】在框框上方顯示 "TEL 0.95" 字樣
                    cv2.putText(frame, f"TEL {conf_val:.2f}", (x1, y1 - 7), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 128, 0), 2)

                # 4. 開始投影 12 個洞
                for c_idx, t in enumerate(tels):
                    cx, cy, h = t['cx'], t['cy'], t['h']
                    
                    for r_idx, ratio in enumerate(row_ratios):
                        hole_x = int(cx + (ratio * h * down_x))
                        hole_y = int(cy + (ratio * h * down_y))
                        
                        # 🟢 畫出綠色瞄準點
                        cv2.drawMarker(frame, (hole_x, hole_y), (0, 255, 0), 
                                       markerType=cv2.MARKER_CROSS, markerSize=14, thickness=2)
                        cv2.putText(frame, f"R{r_idx+1}-C{c_idx+1}", (hole_x - 15, hole_y - 10), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 255), 1)

            cv2.imshow("TEL Robot - Pure TEL Projection", frame)

        except Exception as e:
            print(f"⚠️ 錯誤: {e}")

        if cv2.waitKey(1) & 0xFF == ord('q'): break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()