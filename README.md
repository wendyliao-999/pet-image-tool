# 寵物商品圖片抓取與去背工具

Streamlit 工具，用商品 ID 抓取 Petpetgo 或寵物公園商品圖，去背後輸出 PNG。

## 主要功能

- 支援 Petpetgo 商品 ID。
- 支援寵物公園 Petpark 商品 ID。
- Petpetgo 若自動挑圖有疑慮，會進入人工選圖。
- 可選擇輸出模式：
  - 貼齊商品尺寸，適合 Illustrator 排版。
  - 800x800 正方形透明畫布，適合固定電商規格。
- 批次處理失敗時會在 `output/error_log.txt` 記錄錯誤。

## 本機執行

```bash
pip install -r requirements.txt
streamlit run app.py
```

## 上線

目前部署在 Streamlit Community Cloud：

```text
https://pet-image-tool-2026.streamlit.app/
```

Streamlit 會依 GitHub `main` 分支重新部署。
