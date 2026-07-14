# 維護紀錄

## GitHub Repo

```text
wendyliao-999/pet-image-tool
```

## 部署位置

```text
https://pet-image-tool-2026.streamlit.app/
```

## 建議更新流程

1. 從 GitHub repo 建立新分支。
2. 將本機確認過的 `app.py`、`fetch_images.py`、`requirements.txt` 同步進 repo。
3. 執行語法檢查。
4. 用 `git diff` 確認只包含預期修改。
5. commit 並 push 分支。
6. 確認後再合併到 `main`，讓 Streamlit 重新部署。

## 2026-07-14 更新重點

- 將 Streamlit 舊參數 `use_container_width` 改為 `width="stretch"`。
- 新增每次執行前清理 `output` 舊結果，降低雲端檔案累積風險。
- 新增錯誤紀錄 `output/error_log.txt`。
- rembg session 改為重複使用，避免每張圖重新載入模型。
- 去背前限制大圖尺寸，降低 Streamlit Cloud 記憶體壓力。
- 新增輸出模式：
  - 貼齊商品尺寸，適合 Illustrator。
  - 800x800 正方形透明畫布。
