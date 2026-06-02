import streamlit as st
import requests
from bs4 import BeautifulSoup
from pathlib import Path
import re
import tempfile  # 👈 新增：用來建立閱後即焚的暫存資料夾

# 載入原本寫好的工具包
import fetch_images as fi

# --- 1. 網頁標題與外觀設定 ---
st.set_page_config(page_title="雙平台圖片抓取神器", page_icon="📦", layout="wide")
st.title("📦 寵物商品圖片抓取與去背工具")

# --- 2. 左側邊欄設定區 ---
st.sidebar.header("⚙️ 進階設定")
size = st.sidebar.number_input("輸出圖片大小 (px)", min_value=100, max_value=2000, value=800)
padding = st.sidebar.number_input("邊界留白 (px)", min_value=0, max_value=200, value=24)

# 建立參數物件
class UIArgs:
    def __init__(self):
        self.output_dir = Path("output")
        self.report = Path("output/report.csv")
        self.size = size
        self.padding = padding
        self.keep_original = False  # 👈 這裡設為 False，Petpetgo 的原始圖就不會存下來了

# --- ✨ 修改：寵物公園專屬抓圖引擎 (加上自動清理機制) ---
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
    
    # 👇 使用 with 建立暫存資料夾，一旦這個區塊執行完畢，原始圖就會自動被刪除！
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir) / f"{product_id}.jpg"
        temp_path.write_bytes(img_resp.content)
        
        # 進行去背與縮放
        removed = fi.remove_background(temp_path)
        final = fi.fit_to_canvas(removed, size=args.size, padding=args.padding)
    
    # 只把最終去背好的圖存進 output/final 裡
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
    placeholder="範例寫法:\n12345\n67890"
)

# --- 4. 執行邏輯 ---
if st.button(f"🚀 開始執行抓取任務", use_container_width=True):
    if not input_text.strip():
        st.warning("請先輸入商品 ID 喔！")
    else:
        args = UIArgs()
        product_ids = list(dict.fromkeys(re.findall(r"[A-Za-z0-9]+", input_text)))
        
        if not product_ids:
            st.error("找不到有效的商品 ID...")
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
                fi.write_report(rows, args.report)
                progress_bar.progress(index / len(product_ids))
                
            status_text.text("🎉 所有任務處理完成！")
            st.balloons()
            st.dataframe(rows, use_container_width=True)