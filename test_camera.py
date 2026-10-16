import cv2
import torch
from ultralytics import YOLO

def main():
    # 1. 再次確認 VS Code 內也有抓到 GPU
    print(f"VS Code 抓取 CUDA 狀態: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"當前使用的顯卡: {torch.cuda.get_device_name(0)}")

    # 2. 載入 YOLO26 輕量版預訓練模型 (第一次執行會自動下載 yolo26n.pt 檔案)
    model = YOLO('yolo26n.pt') 

    # 3. 開啟筆電相機 (0 代表預設鏡頭)
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("錯誤：找不到相機，請確認鏡頭未被其他程式（如 Line、Meet）佔用。")
        return

    print("測試成功！即時畫面已開啟，按下鍵盤上的 'q' 鍵可以關閉視窗。")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 4. 讓 YOLO26 使用 GPU (device=0) 進行即時預測
        # 它現在會先辨識你房間裡的人、手機或杯子
        results = model.predict(frame, conf=0.5, device=0, verbose=False)

        # 5. 將辨識結果畫在畫面上並顯示
        annotated_frame = results[0].plot()
        cv2.imshow("TEL Robot - YOLO26 Camera Test", annotated_frame)

        # 按 'q' 鍵退出
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
