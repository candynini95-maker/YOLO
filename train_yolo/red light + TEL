import cv2
import numpy as np
from ultralytics import YOLO

cap = cv2.VideoCapture(1, cv2.CAP_DSHOW) 

# 🎛️ 全域紅光門檻優化設定（在此統一修改，確保偵測與視覺化同步）
# 將 S(飽和度)下限從 100 降到 30，用以捕捉因太亮而過曝泛白的 LED 中心
# 將 V(明度)下限從 200 降到 150，用以捕捉外圍擴散的紅光暈
LOWER_RED1 = np.array([0, 30, 150])
UPPER_RED1 = np.array([10, 255, 255])
LOWER_RED2 = np.array([170, 30, 150])
UPPER_RED2 = np.array([180, 255, 255])

def check_red_light(roi):
    """
    🧠 紅光偵測核心：計算這個區域內有多少紅光像素
    """
    if roi is None or roi.size == 0:
        return 0
    
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    
    # 使用優化後的低門檻進行二值化
    mask1 = cv2.inRange(hsv, LOWER_RED1, UPPER_RED1)
    mask2 = cv2.inRange(hsv, LOWER_RED2, UPPER_RED2)
    
    full_mask = mask1 + mask2
    return cv2.countNonZero(full_mask)


def main():
    model = YOLO('runs/detect/train-7/weights/best.pt') 
    print("【YOLO26 純TEL投影 + 紅光得分洞辨識系統】啟動...")

    row_ratios = [2.0, 6.0, 10.0, 14.0] 

    if not cap.isOpened(): return

    while True:
        ret, frame = cap.read()
        if not ret or frame is None: continue
        
        display_frame = frame.copy()
        h_img, w_img, _ = frame.shape
        
        # -------------------------------------------------------------------------
        # 🔴 步驟一：先做全圖的低門檻紅光過濾 (暫存原始紅光遮罩)
        # -------------------------------------------------------------------------
        hsv_full = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask1_full = cv2.inRange(hsv_full, LOWER_RED1, UPPER_RED1)
        mask2_full = cv2.inRange(hsv_full, LOWER_RED2, UPPER_RED2)
        full_mask_raw = mask1_full + mask2_full
        
        # 預設視覺化除錯視窗為全黑 (若未鎖定目標，則整張圖遮蔽)
        full_mask_visual = np.zeros((h_img, w_img), dtype=np.uint8)
        # -------------------------------------------------------------------------

        active_column = -1
        target_detected = False

        try:
            results = model.predict(frame, conf=0.6, device=0, verbose=False)
            tels = []
            
            for box in results[0].boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
                tels.append({'cx': cx, 'cy': cy, 'h': y2 - y1, 'box': (x1, y1, x2, y2), 'conf': box.conf[0].item()})

            if len(tels) == 3:
                target_detected = True
                tels.sort(key=lambda x: x['cx'])
                
                dx, dy = tels[2]['cx'] - tels[0]['cx'], tels[2]['cy'] - tels[0]['cy']
                board_w = np.sqrt(dx**2 + dy**2)
                ux, uy = dx / board_w, dy / board_w
                down_x, down_y = -uy, ux

                for t in tels:
                    x1, y1, x2, y2 = t['box']
                    cv2.rectangle(display_frame, (x1, y1), (x2, y2), (255, 128, 0), 2)
                    cv2.putText(display_frame, f"TEL {t['conf']:.2f}", (x1, y1 - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 128, 0), 2)

                r1_red_scores = {} 
                r1_coordinates = {}

                for c_idx, t in enumerate(tels):
                    ratio = row_ratios[0] 
                    hole_x = int(t['cx'] + (ratio * t['h'] * down_x))
                    hole_y = int(t['cy'] + (ratio * t['h'] * down_y))
                    r1_coordinates[c_idx] = (hole_x, hole_y)
                    
                    r_size = 12
                    y_start = max(0, hole_y - r_size)
                    y_end = min(h_img, hole_y + r_size)
                    x_start = max(0, hole_x - r_size)
                    x_end = min(w_img, hole_x + r_size)
                    
                    roi = frame[y_start:y_end, x_start:x_end]
                    r1_red_scores[c_idx] = check_red_light(roi)

                if max(r1_red_scores.values()) > 50:
                    active_column = max(r1_red_scores, key=r1_red_scores.get)

                # -------------------------------------------------------------------------
                # 🎯 🧠 【核心新增：動態幾何孔洞遮罩】
                # 當成功鎖定目標時，建立一個只允許 12 個孔洞區域顯現的白色遮罩
                # -------------------------------------------------------------------------
                dynamic_hole_mask = np.zeros((h_img, w_img), dtype=np.uint8)
                
                for c_idx, t in enumerate(tels):
                    for r_idx, ratio in enumerate(row_ratios):
                        hole_x = int(t['cx'] + (ratio * t['h'] * down_x))
                        hole_y = int(t['cy'] + (ratio * t['h'] * down_y))
                        
                        # 在遮罩上畫出得分格區域（半徑設為 20 像素，稍微留寬一點看光暈）
                        cv2.circle(dynamic_hole_mask, (hole_x, hole_y), 20, 255, -1)
                        
                        # 順便繪製主畫面彩色 UI
                        if r_idx == 0 and c_idx == active_column:
                            cv2.rectangle(display_frame, (hole_x - 18, hole_y - 18), (hole_x + 18, hole_y + 18), (0, 0, 255), 3)
                            cv2.drawMarker(display_frame, (hole_x, hole_y), (0, 0, 255), markerType=cv2.MARKER_TILTED_CROSS, markerSize=20, thickness=3)
                            cv2.putText(display_frame, f"R1-C{c_idx+1} [ SCORE! ]", (hole_x - 45, hole_y - 25), 
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
                        else:
                            cv2.drawMarker(display_frame, (hole_x, hole_y), (0, 255, 0), markerType=cv2.MARKER_CROSS, markerSize=14, thickness=2)
                            cv2.putText(display_frame, f"R{r_idx+1}-C{c_idx+1}", (hole_x - 15, hole_y - 10), 
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 255), 1)

                # 使用位元運算 (AND)，讓全圖紅光遮罩只保留那 12 個白圈內的區域，其餘通通抹黑
                full_mask_visual = cv2.bitwise_and(full_mask_raw, dynamic_hole_mask)
                # -------------------------------------------------------------------------

        except Exception as e:
            print(f"⚠️ 錯誤: {e}")

        # 🏆 彈出乾淨無雜訊的 HSV 除錯視窗 (此時只會看到隨機得分格區域內的紅光像素)
        cv2.imshow("NCU BOLTNUT - HSV Red Mask View", full_mask_visual)

        # HUD 戰術面板渲染 (彩色主畫面)
        center_x, center_y = w_img // 2, h_img // 2
        cv2.drawMarker(display_frame, (center_x, center_y), (0, 255, 0), cv2.MARKER_CROSS, 40, 2)
        cv2.circle(display_frame, (center_x, center_y), 20, (0, 255, 0), 1)

        cv2.rectangle(display_frame, (15, 15), (280, 95), (0, 0, 0), -1)
        cv2.rectangle(display_frame, (15, 15), (280, 95), (0, 255, 0), 1)
        cv2.putText(display_frame, "SYS: NCU BOLTNUT v2.6", (25, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        cv2.putText(display_frame, "MODE       : PURE VISION ONLY", (25, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        target_str = f"ROW 1 - COL {active_column+1}" if active_column != -1 else "NONE"
        cv2.putText(display_frame, f"TARGET COL : {target_str}", (25, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        if not target_detected:
            cv2.putText(display_frame, "SCANNING TARGET...", (20, h_img - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        else:
            cv2.putText(display_frame, "TARGET ACQUIRED", (20, h_img - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        fire_control_state = 'READY' if (target_detected and active_column != -1) else 'HOLD'
        if fire_control_state == 'READY':
            cv2.rectangle(display_frame, (w_img - 220, h_img - 60), (w_img - 20, h_img - 20), (0, 255, 0), -1)
            cv2.putText(display_frame, "READY TO FIRE", (w_img - 200, h_img - 33), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
        else:
            cv2.rectangle(display_frame, (w_img - 240, h_img - 60), (w_img - 20, h_img - 20), (0, 0, 255), -1)
            cv2.putText(display_frame, "FIRE CONTROL HOLD", (w_img - 230, h_img - 33), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)

        # 🏆 主視窗：顯示整合完戰術 HUD 的最終機器人視角畫面
        cv2.imshow("NCU BOLTNUT - Robot Vision HUD", display_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'): break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()