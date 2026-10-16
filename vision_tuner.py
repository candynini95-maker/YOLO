import cv2
import numpy as np
from ultralytics import YOLO

# 沒功能的回呼函數（OpenCV 滑桿必備）
def nothing(x):
    pass

def main():
    # 載入你之前訓練好的 best.pt 模型
    # (如果還沒訓練新模型，可以先用 'yolo26n.pt' 確保程式能跑)
    model = YOLO('runs/detect/train/weights/best.pt') 

    # 開啟相機
    cap = cv2.VideoCapture(0)
    
    # 建立一個專門用來顯示與調整的視窗
    window_name = "TEL Robot - Geometry Tuner"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    # 🛠️ 建立 4 個動態調教滑桿 (調整範圍 0 ~ 400，預設值對應我們之前的比例)
    # 滑桿數值會除以 100 轉為實際的幾何比例參數
    cv2.createTrackbar("Col_Step1", window_name, 8, 150, nothing)   # 第一欄與第四欄橫向邊界
    cv2.createTrackbar("Col_Step2", window_name, 36, 150, nothing)  # 第二欄與第三欄內縮比例
    cv2.createTrackbar("Row_Start", window_name, 120, 400, nothing) # 第一排洞口下移距離
    cv2.createTrackbar("Row_Pitch", window_name, 100, 300, nothing) # 每一排洞口之間的垂直間距

    print("================================================================")
    print("【幾何對位調教工具已啟動】")
    print("1. 請將相機對準你們的『紙做小模型』。")
    print("2. 透過滑桿微調綠色目標點，直到它們『完美吸附』在紙模型的 12 個洞心。")
    print("3. 調整滿意後，按下鍵盤 's' 鍵，終端機將自動印出最精準的比例參數！")
    print("4. 按下 'q' 鍵可直接關閉程式。")
    print("================================================================")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 讀取當前滑桿的數值並轉換為浮點數比例
        c_step1 = cv2.getTrackbarPos("Col_Step1", window_name) / 100.0
        c_step2 = cv2.getTrackbarPos("Col_Step2", window_name) / 100.0
        r_start = cv2.getTrackbarPos("Row_Start", window_name) / 100.0
        r_pitch = cv2.getTrackbarPos("Row_Pitch", window_name) / 100.0

        # 計算四個縱列與三個橫排的相對偏移矩陣
        col_offsets = [c_step1, c_step2, 1.0 - c_step2, 1.0 - c_step1]
        row_offsets = [r_start, r_start + r_pitch, r_start + r_pitch * 2]

        # 讓 YOLO26 尋找畫面上小模型紙上的 TEL
        results = model.predict(frame, conf=0.25, device=0, verbose=False)
        detected_tels = []

        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy)
                cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
                if all(np.linalg.norm(np.array([cx, cy]) - np.array(e)) > 30 for e in detected_tels):
                    detected_tels.append((cx, cy))

        detected_tels = sorted(detected_tels, key=lambda x: x)

        # 畫出偵測到的基準點 TEL
        for idx, (cx, cy) in enumerate(detected_tels):
            cv2.circle(frame, (cx, cy), 6, (0, 0, 255), -1)
            cv2.putText(frame, f"TEL_{idx+1}", (cx - 20, cy - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

        # 🎯 核心幾何盲推（只要車體相機抓到左右兩個 TEL）
        if len(detected_tels) >= 2:
            p1, p3 = detected_tels, detected_tels[-1]
            base_width = p3 - p1 # 基準寬度（車子靠近時會變大、遠離時會變小，達成自動縮放）

            if base_width > 30:
                for r_idx, r_off in enumerate(row_offsets):
                    for c_idx, c_off in enumerate(col_offsets):
                        # 根據滑桿即時計算出的比例，推算虛擬洞口坐標
                        hole_x = int(p1 + base_width * c_off)
                        hole_y = int(p1 + base_width * r_off)

                        # 在畫面上即時渲染綠色預測點
                        cv2.circle(frame, (hole_x, hole_y), 5, (0, 255, 0), -1)
                        cv2.putText(frame, f"R{r_idx+1}-C{c_idx+1}", (hole_x - 15, hole_y - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 0), 1)

        # 顯示即時畫面
        cv2.imshow(window_name, frame)
        key = cv2.waitKey(1) & 0xFF

        # 按 's' 鍵，儲存並印出當前最完美的參數
        if key == ord('s'):
            print("\n🎉【成功取得小模型最精準參數！】請將以下程式碼片段替換至正式程式中：")
            print(f"col_offsets = [{col_offsets[0]:.2f}, {col_offsets[1]:.2f}, {col_offsets[2]:.2f}, {col_offsets[3]:.2f}]")
            print(f"row_offsets = [{row_offsets[0]:.2f}, {row_offsets[1]:.2f}, {row_offsets[2]:.2f}]")
            print("-" * 60)

        # 按 'q' 鍵退出
        if key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
