import cv2
from ultralytics import YOLO

def main():
    # 💡 關鍵改動：將 'runs/.../best.pt' 改為 'yolo26n.pt'
    # 第一次執行時，系統會自動聯網下載官方在 COCO 資料庫訓練好的權重（約幾 MB，很快）
    model = YOLO('yolo26n.pt') 
    
    # 打開相機 (1=手機Iriun, 0=內建, 2=外接)
    cap = cv2.VideoCapture(2, cv2.CAP_DSHOW)
    
    print("🚀 官方 COCO 資料庫模型已載入！")
    print("📱 現在可以對著鏡頭測試看人(person)、手機(cell phone)等物品囉！按 'q' 鍵結束。")
    
    false_ret_count = 0
    
    while True:
        ret, frame = cap.read()
        
        if not ret or frame is None:
            false_ret_count += 1
            if false_ret_count > 30:
                print("❌ 相機連線中斷。")
                break
            cv2.waitKey(10)
            continue
        
        false_ret_count = 0
        
        try:
            # 讓 YOLO 進行預測，並用內建的畫布畫框
            results = model.predict(frame, conf=0.25, device=0, stream=True)
            
            for r in results:
                # 拿內建畫好人、手機標籤的畫面
                annotated_frame = r.plot()  
                
                # 顯示畫面
                cv2.imshow("YOLO26 Official Pretrained Demo", annotated_frame)
                
        except Exception as e:
            print(f"⚠️ 畫面處理出錯: {e}")
            
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("👋 👋 程式已安全關閉。")

if __name__ == '__main__':
    main()