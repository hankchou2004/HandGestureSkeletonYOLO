# 研究用最小訓練流程

這個資料夾只保留你做「比較不同 YOLO 在手部關鍵點資料集上的訓練效果」所需的最小程式碼。

## 你現在應該只用這一個檔案
- study/hand_pose_research.py

## 這個檔案在做什麼
1. 載入 YOLO pose 模型
2. 指向 datasets/hand-keypoints/data.yaml
3. 執行訓練
4. 儲存結果到 runs/hand-pose-study

## 不需要的內容
- docs/：文件與說明
- examples/：範例與部署程式碼
- tests/：測試程式碼
- 其他任務模型（detect / segment / classify 等）
