import streamlit as st
import requests
from bs4 import BeautifulSoup
from pathlib import Path
import re
import tempfile
import zipfile
import io

# 載入原本寫好的工具包
import fetch_images as fi

# --- 1. 網頁標題與外觀設定 ---
st.set_page_config(page_title="雙平台圖片抓取神器", page_icon="📦", layout="wide")
st.title("📦 寵物商品圖片抓取與去背工具")

# --- 🧠 狀態暫存區 (解決下載中斷的關鍵魔法) ---
if "task_completed" not in st.session_state:
    st.session_state.task_completed = False
    st.session_state.successful_files = []
    st.session_state.result_rows = []

# --- 2. 左側邊欄設定區 ---
st.sidebar.header("⚙️ 進階設定")
size = st.sidebar.number_input("輸出圖片大小 (px)", min_value=100, max_value=2000, value=800)
padding = st.sidebar.number_input("邊界留白 (px)", min_value=0, max_value=200, value=24)

class UIArgs:
    def __init__(self):
        self.output_dir = Path("output")
        self.report = Path("output/report.csv")
        self.size = size
        self.padding = padding
        self.keep_original = False

# --- ✨ 寵物公園專屬抓圖引擎 ---
def process_petpark(product_id, args):
    session = requests.Session()
    session.headers.update({"User-Agent": fi.USER_AGENT})
    
    url = f"https://shop.petpark.com.tw/{product_id}"
    resp = session.get(url, timeout=30)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")
    
    og_img = soup.find("meta", property="og:image")
    if not og_img or not og_img.get("content"):
        raise ValueError("這頁找不到商品主圖喔！")
        
    raw_img_url = og_img["content"].split("?")[0]
    
    img_resp = session.get(raw_img_url, timeout=30)
    img_resp.raise_for_status()
    
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir) / f"{product_id}.jpg"
        temp_path.write_bytes(img_resp.content)
        
        removed = fi.remove_background(temp_path)
        final = fi.fit_to_canvas(removed, size=args.size, padding=args.padding)
    
    args.output_dir.joinpath("final").mkdir(parents=True, exist_ok=True)
    final_file = args.output_dir / "final" / f"{product_id}.png"
    final.save(final_file)
    
    return {
        "product_id": f"{product_id}",
        "status": "success",
        "final_path": str(final_file),
        "selected_image_url": raw_img_url
    }

# --- 3. 主畫面輸入區 ---
platform = st.radio("🏷️ 請選擇這次要抓取的平台：", ["Petpetgo", "寵物公園 (Petpark)"], horizontal=True)

input_text = st.text_area(
    "🔗 請貼上商品 ID (可換行或逗號分隔)：",
    height=150,
    placeholder="範例寫法:\n12345\nWP006404"
)

# --- 4. 執行邏輯 ---
if st.button(f"🚀 開始執行抓取任務", use_container_width=True):
    # 每次按鈕按下，先清除舊的記憶
    st.session_state.task_completed = False
    st.session_state.successful_files = []
    
    if not input_text.strip():
        st.warning("請先輸入商品 ID 喔！")
    else:
        args = UIArgs()
        product_ids = list(dict.fromkeys(re.findall(r"[A-Za-z0-9]+", input_text)))
        
        if not product_ids:
            st.error("找不到有效的商品 ID...")
        elif len(product_ids) > 20:
            st.error(f"⚠️ 為了避免雲端主機當機或被網站封鎖，每次最多只能處理 20 筆 ID 喔！（您目前輸入了 {len(product_ids)} 筆，請分批執行）")
        else:
            st.success(f"✅ 成功載入 {len(product_ids)} 筆【{platform}】的任務！準備開工...")
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            image_cols = st.columns(3)
            rows = []
            
            session = requests.Session()
            session.headers.update({"User-Agent": fi.USER_AGENT, "Accept-Language": "zh-TW"})
            
            for index, pid in enumerate(product_ids, 1):
                status_text.text(f"⏳ 正在處理中: ID {pid} ({index}/{len(product_ids)})")
                
                try:
                    if platform == "寵物公園 (Petpark)":
                        row = process_petpark(pid, args)
                    else:
                        row = fi.process_product(session, pid, args)
                        
                    if row.get("final_path") and Path(row["final_path"]).exists():
                        col_idx = (index - 1) % 3
                        image_cols[col_idx].image(
                            row["final_path"], 
                            caption=f"✅ {row['product_id']} 完成"
                        )
                except Exception as exc:
                    row = {"product_id": pid, "status": "failed", "error": str(exc)}
                    st.error(f"❌ ID {pid} 處理失敗: {exc}")
                    
                rows.append(row)
                progress_bar.progress(index / len(product_ids))
                
            status_text.text("🎉 所有任務處理完成！")
            st.balloons()
            
            # --- 關鍵：把成功抓到的圖片清單「記在腦海裡 (Session State)」---
            st.session_state.successful_files = [r["final_path"] for r in rows if r.get("status") == "success" and r.get("final_path")]
            st.session_state.result_rows = rows
            st.session_state.task_completed = True

# --- 5. 獨立的下載區塊 (完全不受重整影響) ---
if st.session_state.task_completed and st.session_state.successful_files:
    st.markdown("---")
    st.markdown("### 🎁 打包下載區")
    
    # 顯示報表
    st.dataframe(st.session_state.result_rows, use_container_width=True)
    
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zip_file:
        for file_path in st.session_state.successful_files:
            zip_file.write(file_path, Path(file_path).name)
    
    # 我也順便幫你優化了按鈕文字，讓你可以看到總共包了幾張圖進去！
    st.download_button(
        label=f"📦 一鍵下載 {len(st.session_state.successful_files)} 張去背圖片 (ZIP 壓縮檔)",
        data=zip_buffer.getvalue(),
        file_name="pet_images_output.zip",
        mime="application/zip",
        use_container_width=True
    )