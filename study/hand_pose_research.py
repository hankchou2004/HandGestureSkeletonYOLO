"""研究用最小版 YOLO 手部關鍵點訓練腳本。

這個檔案只保留你做「比較不同 YOLO 模型在手部資料集上的訓練效果」所需的核心流程：
1) 載入模型
2) 指向 hand-keypoints 資料集
3) 啟動訓練
4) 輸出結果路徑

其他 docs、examples、部署程式碼都不屬於這個研究主流程。
"""

# 1. 匯入 Python 與 YOLO API，這是整個研究流程的基本入口。
from pathlib import Path

from ultralytics import YOLO


# 2. 設定研究用的資料路徑與訓練參數，避免把不必要的設定混入程式。
REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_YAML = REPO_ROOT / "datasets" / "hand-keypoints" / "data.yaml"
RUNS_DIR = REPO_ROOT / "runs" / "hand-pose-study"
MODEL_NAMES = ["yolo11n-pose.pt", "yolo26n-pose.pt"]
EPOCHS = 100  # 研究初步比較可先設為 1 epoch，之後再調整為正式實驗值。
IMGSZ = 640


# 3. 定義單次訓練函式，讓你能快速重複比較不同模型。
def train_one_model(model_name: str) -> None:
    """訓練單一個 YOLO pose 模型並輸出實驗結果。"""

    # 3.1 載入指定的 pose 模型；這一步是研究比較的核心起點。
    model = YOLO(model_name)

    # 3.2 啟動訓練；data 直接指向 hand-keypoints 資料集 YAML。
    #      project 會把每次實驗結果存到獨立資料夾，方便後續比較。
    model.train(
        data=str(DATA_YAML),
        epochs=EPOCHS,
        imgsz=IMGSZ,
        project=str(RUNS_DIR),
        name=model_name.replace(".pt", ""),
        exist_ok=True,
    )

    # 3.3 訓練完成後印出結果路徑，方便你查看每一組實驗輸出。
    print(f"\n[OK] {model_name} 已完成訓練，結果位於: {RUNS_DIR / model_name.replace('.pt', '')}")


# 4. 主流程：依序比較多個模型，這是你研究的最小執行入口。
if __name__ == "__main__":
    # 4.1 在開始前先確認資料檔存在，避免後續訓練因路徑錯誤失敗。
    if not DATA_YAML.exists():
        raise FileNotFoundError(f"找不到資料集 YAML：{DATA_YAML}\n請先下載並解壓資料集。")

    # 4.2 逐一訓練每個模型，方便你收集不同 YOLO 版本的表現。
    for model_name in MODEL_NAMES:
        print(f"\n=== 開始訓練: {model_name} ===")
        train_one_model(model_name)

    # 4.3 完成後提醒使用者可在 runs/hand-pose-study 目錄查看所有實驗結果。
    print("\n所有實驗已完成。請查看 runs/hand-pose-study 目錄中的結果檔。")
