import serial
import time

# 🔌 初始化序列埠 (COM3 請根據你 Windows 裝置管理員看到的 Port 修改)
try:
    ser = serial.Serial('COM3', 9600, timeout=1)
    time.sleep(2) # 等待 Arduino 重啟
    print("序列埠連接成功！")
    
    # 範例：假設我們要投 R2-C3 洞，計算出來的座標是 (320, 240)
    target_x, target_y = 320, 240
    data_string = f"{target_x},{target_y}\n"
    
    # 傳送給 Arduino
    ser.write(data_string.encode('utf-8'))
    print(f"已成功傳送坐標：{data_string}")
    
    ser.close()
except Exception as e:
    print(f"連接失敗，請確認 Arduino 是否插上電腦：{e}")
