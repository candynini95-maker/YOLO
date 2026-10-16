import cv2
import numpy as np

def main():
    # 1. 直接讀取你手動剪裁好的完美 TEL 範本
    template = cv2.imread('tel_template.png')
    if template is None:
        print("【錯誤】找不到 tel_template.png！請確認你已經把剪好的 TEL 圖片放進資料夾。")
        return

    # 2. 啟動筆電相機
    cap = cv2.VideoCapture(0)
    print("【東京威力影像辨識】優化版啟動！請拿手機畫面對著鏡頭測試。")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 3. 轉為灰階降低光線干擾
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray_template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
        w_t, h_t = gray_template.shape[::-1]

        # 4. 執行模板匹配
        res = cv2.matchTemplate(gray_frame, gray_template, cv2.TM_CCOEFF_NORMED)
        
        # 🔥【修正點】將門檻調低到 0.45，讓手機翻拍的畫面更容易被認出來
        threshold = 0.45
        loc = np.where(res >= threshold)

        detected_tels = []

        # 5. 篩選重複點
        for pt in zip(*loc[::-1]):
            cx = pt[0] + w_t // 2
            cy = pt[1] + h_t // 2
            
            # 濾除太近的重複框
            if all(np.linalg.norm(np.array([cx, cy]) - np.array(existing)) > 40 for existing in detected_tels):
                detected_tels.append((cx, cy))

        # 6. 依照 X 軸從左到右排序
        detected_tels = sorted(detected_tels, key=lambda x: x[0])

        # 在畫面上畫出偵測到的 TEL 圓心與編號
        for idx, (cx, cy) in enumerate(detected_tels):
            cv2.circle(frame, (cx, cy), 6, (255, 0, 0), -1) # 藍色圓心
            cv2.putText(frame, f"TEL_{idx+1}", (cx - 20, cy - 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)

        # 7. 🔥 幾何網格反推（只要認出至少 2 個 TEL）
        if len(detected_tels) >= 2:
            p1 = detected_tels[0]     # 最左邊的 TEL
            p3 = detected_tels[-1]    # 最右邊的 TEL
            
            base_width = p3[0] - p1[0] # 計算這兩個 TEL 的橫向像素距離
            
            if base_width > 40:
                # 比例參數（根據標準場地幾何量身打造）
                col_offsets = [0.08, 0.36, 0.64, 0.92] # 四個縱列的橫向偏移比例
                row_offsets = [1.2, 2.2, 3.2]          # 三個橫排的縱向下移比例

                for r_idx, r_off in enumerate(row_offsets):
                    for c_idx, c_off in enumerate(col_offsets):
                        # 運用相對幾何公式，推算出 12 個洞口的模擬坐標
                        hole_x = int(p1[0] + base_width * c_off)
                        # 以左邊 TEL 的 Y 軸為基準線向下偏移
                        hole_y = int(p1[1] + base_width * r_off) 

                        # 在相機畫面上繪製綠色目標圓心
                        cv2.circle(frame, (hole_x, hole_y), 5, (0, 255, 0), -1)
                        cv2.putText(frame, f"R{r_idx+1}-C{c_idx+1}", (hole_x - 15, hole_y - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 0), 1)

        # 印出目前畫面上抓到了幾個 TEL，方便除錯
        cv2.putText(frame, f"Detected TEL count: {len(detected_tels)}", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        cv2.imshow("TEL Robot - Optimized Geometry Vision", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
