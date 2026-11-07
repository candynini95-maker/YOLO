# TEL Robot Vision - 重構變更、待確認事項與跨專案建議報告

本文件記錄本次針對 **TEL Robot Vision** 專案進行的架構重構、程式碼修正、待確認細節，以及對未來主專案 **[TELBot-Host](https://github.com/Alllen95Wei/TELBot-Host)** 整合的具體架構建議。

---

## 1. 已完成的變更與重構 (What changes have been made)

### 1.1 解決專案結構混亂與命名的技術債
* **檔案命名規範化**：
  * 原先由 ChatGPT 生成的無副檔名且含空白字元檔名已透過 Git 正式重命名：
    * `train_yolo/red light + TEL` ➔ [`train_yolo/red_light_tel.py`](file:///c:/Users/candy/OneDrive/Desktop/TEL_Robot_Vision/train_yolo/red_light_tel.py)
    * `train_yolo/yolo tel` ➔ [`train_yolo/yolo_tel.py`](file:///c:/Users/candy/OneDrive/Desktop/TEL_Robot_Vision/train_yolo/yolo_tel.py)
    * `yolo26` ➔ [`yolo26_demo.py`](file:///c:/Users/candy/OneDrive/Desktop/TEL_Robot_Vision/yolo26_demo.py)
  * 修復 `.gitignore`，補上 `__pycache__/`、`*.py[cod]` 及快取目錄過濾。

### 1.2 修復潛在語法與邏輯 Bug
* **修復 [`vision_tuner.py`](file:///c:/Users/candy/OneDrive/Desktop/TEL_Robot_Vision/vision_tuner.py) 的幾何計算錯誤**：
  * 修正第 71 行將整份列表賦值給單一變數造成的 `p3 - p1` tuple/list 減法型態錯誤 (`TypeError`)。
  * 修正 `sorted(..., key=lambda x: x[0])` 與像素座標差值計算，恢復滑桿調校工具的正常運作。

### 1.3 核心架構模組化 (`tel_vision` 套件)
建立了高內聚、低耦合的 Python 套件 [`tel_vision/`](file:///c:/Users/candy/OneDrive/Desktop/TEL_Robot_Vision/tel_vision)，將所有核心業務分離為獨立職責模組：
1. **[`tel_vision/config.py`](file:///c:/Users/candy/OneDrive/Desktop/TEL_Robot_Vision/tel_vision/config.py)** & **[`config.yaml`](file:///c:/Users/candy/OneDrive/Desktop/TEL_Robot_Vision/config.yaml)**：
   * 將所有硬編碼參數（相機串流 URL、模型路徑、推論裝置、剛體幾何倍率、HSV 門檻、HTTP POST 目標等）完全外部化。
2. **[`tel_vision/video_stream.py`](file:///c:/Users/candy/OneDrive/Desktop/TEL_Robot_Vision/tel_vision/video_stream.py)**：
   * 實現了統一視訊來源抽象層，原生支援 **HTTP MJPEG 網路串流**（如 `http://<ip>:<port>/stream.mjpg`）、RTSP 串流與本機 USB 攝影機，並具備自動斷線重連與本機攝影機 fallback 機制。
3. **[`tel_vision/detector.py`](file:///c:/Users/candy/OneDrive/Desktop/TEL_Robot_Vision/tel_vision/detector.py)**：
   * 支援 PyTorch `.pt` 權重與 ONNX 輕量化模型 (`best.onnx`)。
   * 自動探測硬體加速：在 Windows 檢測 NVIDIA GPU 時優先啟用 CUDA 加速，無 GPU 時安全降級至 CPU。
4. **[`tel_vision/geometry.py`](file:///c:/Users/candy/OneDrive/Desktop/TEL_Robot_Vision/tel_vision/geometry.py)**：
   * 封裝 3 個 TEL 剛體標籤的向量推算演算法（計算水平基準向量 $\vec{u}$ 與向下法向量 $\vec{d}$），推導 3 列 × 4 排 = 12 個孔洞坐標並動態繪製遮罩。
5. **[`tel_vision/color_filter.py`](file:///c:/Users/candy/OneDrive/Desktop/TEL_Robot_Vision/tel_vision/color_filter.py)**：
   * 實作針對 LED 核心過曝特性的低飽和度（$S \ge 30, V \ge 150$）雙區間 HSV 濾波。
   * 嚴格對準使用者需求：僅巡檢第一排（Row 1）的三個孔洞（左 `left`、中 `center`、右 `right`），並鎖定單一有效最高分孔洞。
6. **[`tel_vision/http_client.py`](file:///c:/Users/candy/OneDrive/Desktop/TEL_Robot_Vision/tel_vision/http_client.py)**：
   * 實作獨立的背景非同步執行緒與單一深度佇列，主動向 Host 發送 HTTP POST。
   * 傳輸格式完全符合要求：
     ```json
     {
       "timestamp": 1731001234.567,
       "active_target": "center",
       "target_detected": true,
       "fire_control_state": "READY"
     }
     ```
   * **非同步隔離設計**：當 TELBot-Host 離線或網路延遲時，HTTP 發送完全不阻塞相機取像與視覺推論迴圈。
7. **[`tel_vision/renderer.py`](file:///c:/Users/candy/OneDrive/Desktop/TEL_Robot_Vision/tel_vision/renderer.py)**：
   * 將鋼鐵人風格 HUD 戰術面板與動態黑白遮罩的繪製邏輯完全從推論流程中抽離。
8. **[`tel_vision/pipeline.py`](file:///c:/Users/candy/OneDrive/Desktop/TEL_Robot_Vision/tel_vision/pipeline.py)**：
   * 整合各模組的高階協調器，支援直接呼叫 `process_frame(frame)`（利於未來模組化嵌入 TELBot-Host）。

### 1.4 Headless 無介面部署支援與統一進入點
* **[`main.py`](file:///c:/Users/candy/OneDrive/Desktop/TEL_Robot_Vision/main.py)**：
  * 提供豐富的命令列選項：
    * 樹莓派 5 Headless 運行：`python main.py --headless`
    * Windows 開發預覽：`python main.py --gui`
    * 指定網路串流：`python main.py --source "http://192.168.1.100:8080/stream.mjpg"`
    * 指定輕量 ONNX 模型：`python main.py --model best.onnx --device cpu --headless`
* **自動化單元測試**：
  * 新增 [`tests/test_pipeline.py`](file:///c:/Users/candy/OneDrive/Desktop/TEL_Robot_Vision/tests/test_pipeline.py)，測試設定載入、幾何向量推算、過曝色彩過濾與 HTTP 酬載封裝，全數測試通過。

---

## 2. 需後續進一步確認的事項 (What should be done that requires further confirmation)

1. **TELBot-Host 的實際接收端點 URL**：
   * 目前設定檔預設為 `http://127.0.0.1:8000/api/vision/target`。
   * 當 TELBot-Host 在機器人主控板上啟動時，需在 `config.yaml` 或透過 `--endpoint` 參數指定其實際 IP 與路由。
2. **場地物理尺寸與幾何倍率微調**：
   * 目前採用的 `row_ratios: [2.0, 6.0, 10.0, 14.0]` 與 `roi_sample_radius: 14` 是根據前期模型標定。若比賽現場更換為實體大靶板，建議使用修復後的 [`vision_tuner.py`](file:///c:/Users/candy/OneDrive/Desktop/TEL_Robot_Vision/vision_tuner.py) 進行最後的物理尺寸吸附確認。
3. **Windows 本機 CUDA 環境補全**：
   * 本機硬體具備 RTX 4060 GPU，但系統 Python 3.13 安裝的是 CPU 版 PyTorch。若需要在本機 Windows 達成極限推論幀率，可安裝支援 CUDA 的 PyTorch 版本，或直接以 `python main.py --model best.onnx` 調用 ONNX Runtime。

---

## 3. 超出本專案範圍的外部建議 (Any suggestions that's outside of this project's scope: TELBot-Host)

針對未來將本視覺子系統整合至主專案 **[TELBot-Host](https://github.com/Alllen95Wei/TELBot-Host)**，提出以下架構設計建議：

### 3.1 TELBot-Host 端 HTTP 接收器實作建議
建議在 TELBot-Host 內部使用輕量級非同步框架（如 `FastAPI` 或 `aiohttp`）開設端點：
```python
# 建議之 TELBot-Host 接收端範例
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

class VisionPayload(BaseModel):
    timestamp: float
    active_target: str       # "left" | "center" | "right" | "none"
    target_detected: bool
    fire_control_state: str  # "READY" | "HOLD"

@app.post("/api/vision/target")
async def receive_vision_target(data: VisionPayload):
    # 1. 更新主控狀態機
    # 2. 觸發底盤瞄準或雲台對位
    # 3. 若 fire_control_state == 'READY' 且對準，發送擊發訊號給發射機構
    return {"status": "ok"}
```

### 3.2 增加通訊守門員 (Watchdog / Heartbeat)
* **安全防護**：機器人競賽具備高度動態性。若網路封包遺失、MJPEG 串流延遲過高或視覺程式異常崩潰，Host 端應建立一組逾時看門狗機制（例如超過 300ms 未收到新封包，立即將機器人降級為安全 `HOLD` 狀態，禁止盲目擊發）。

### 3.3 樹莓派 5 上的影像串流傳輸建議
* 在 RPi 5 上將相機影像轉換為 HTTP MJPEG 串流時，建議使用高效能的 **`ustreamer`** 或硬體加速的 **`libcamera-vid`**：
  ```bash
  # 範例：使用 ustreamer 在樹莓派上提供超低延遲 MJPEG 串流
  ustreamer --device=/dev/video0 --host=0.0.0.0 --port=8080 --format=MJPEG --resolution=640x480 --desired-fps=30
  ```
  相較於直接在 Python 內逐張編碼，專用串流伺服器可大幅降低樹莓派 CPU 負載。

### 3.4 兩種整合至 TELBot-Host 的途徑比較
1. **獨立進程微服務模式（推薦目前使用）**：
   * 視覺系統以獨立 process 運行，透過 HTTP POST 與 TELBot-Host 通訊。
   * **優點**：解耦徹底、視覺崩潰不影響底盤運動控制，且可在 Windows/筆電遠端運算視覺，只將瞄準結果傳回機器人。
2. **直接代碼庫內嵌模式**：
   * 未來若將本專案目錄納入 `TELBot-Host/vision/`，可直接 `from tel_vision import VisionPipeline`，在同一個 Python 程序內呼叫 `pipeline.process_frame(frame)`，省去 HTTP 網路序列化開銷。

