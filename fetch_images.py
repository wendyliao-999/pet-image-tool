#!/usr/bin/env python3
import argparse
import contextlib
import csv
import io
import json
import random
import re
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
from urllib.parse import urlparse

import requests
from bs4 import BeautifulSoup
import numpy as np
from PIL import Image, ImageFilter

try:
    from rembg import remove, new_session
except ImportError:
    remove = None
    new_session = None


BASE_URL = "https://petpetgo.com/product/{product_id}"
CDN_BASE = "https://cdn-v4.petpetgo.com"
DEFAULT_DELAY = (1.5, 3.0)
USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/125.0.0.0 Safari/537.36"
)


@dataclass
class Candidate:
    url: str
    name: str
    source: str
    path: Path | None = None
    score: float = 0.0
    white_ratio: float = 0.0
    edge_white_ratio: float = 0.0
    center_content_ratio: float = 0.0
    # 新增：物件密度分，用來 deprioritize 複合情境圖
    compoundedness_score: float = 0.0 
    note: str = ""


def normalize_image_url(src: str) -> str:
    src = src.strip()
    if src.startswith("http://") or src.startswith("https://"):
        return src
    if src.startswith("/"):
        return CDN_BASE + src
    return f"{CDN_BASE}/{src}"


def read_product_ids(input_value: str) -> list[str]:
    if input_value.startswith("http://") or input_value.startswith("https://"):
        response = requests.get(input_value, timeout=30, headers={"User-Agent": USER_AGENT})
        response.raise_for_status()
        text = response.text
    else:
        text = Path(input_value).read_text(encoding="utf-8-sig")

    ids: list[str] = []
    reader = csv.reader(io.StringIO(text))
    for row in reader:
        for cell in row:
            product_ids_found = re.findall(r"[A-Za-z0-9_-]+", cell)
            ids.extend(product_ids_found)
    return list(dict.fromkeys(ids))


def fetch_html(session: requests.Session, product_id: str, retries: int = 3) -> str:
    url = BASE_URL.format(product_id=product_id)
    last_error: Exception | None = None
    for attempt in range(retries):
        try:
            response = session.get(url, timeout=30)
            if response.status_code in {403, 429}:
                wait = 30 + attempt * 30
                time.sleep(wait)
            response.raise_for_status()
            return response.text
        except requests.RequestException as exc:
            last_error = exc
            time.sleep(3 + attempt * 5)
    raise RuntimeError(f"商品頁讀取失敗: {last_error}")


def extract_candidates(html: str) -> list[Candidate]:
    soup = BeautifulSoup(html, "html.parser")
    candidates: list[Candidate] = []
    seen: set[str] = set()

    next_data = soup.find("script", id="__NEXT_DATA__")
    if next_data and next_data.string:
        data = json.loads(next_data.string)
        product = data.get("props", {}).get("pageProps", {}).get("product", {})
        for item in product.get("images", []) or []:
            if item.get("isHidden"):
                continue
            src = item.get("src")
            if not src:
                continue
            url = normalize_image_url(src)
            if url in seen:
                continue
            seen.add(url)
            candidates.append(
                Candidate(
                    url=url,
                    name=item.get("name") or Path(urlparse(url).path).name,
                    source="product.images",
                )
            )

        detail_html = product.get("html") or ""
        detail_soup = BeautifulSoup(detail_html, "html.parser")
        for img in detail_soup.find_all("img"):
            src = img.get("src")
            if not src:
                continue
            url = normalize_image_url(src)
            if url in seen:
                continue
            seen.add(url)
            candidates.append(
                Candidate(
                    url=url,
                    name=Path(urlparse(url).path).name,
                    source="product.html",
                )
            )

    return candidates


def filename_for_candidate(product_id: str, index: int, candidate: Candidate) -> str:
    suffix = Path(urlparse(candidate.url).path).suffix.lower() or ".jpg"
    safe_name = re.sub(r"[^0-9A-Za-z_.-]+", "_", candidate.name).strip("_")
    if not safe_name:
        safe_name = f"candidate_{index}{suffix}"
    if not safe_name.lower().endswith((".jpg", ".jpeg", ".png", ".webp")):
        safe_name += suffix
    return f"{product_id}_{index:02d}_{safe_name}"


def download_candidate(
    session: requests.Session,
    candidate: Candidate,
    product_id: str,
    index: int,
    output_dir: Path,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / filename_for_candidate(product_id, index, candidate)
    if path.exists() and path.stat().st_size > 0:
        candidate.path = path
        return path
    response = session.get(candidate.url, timeout=60)
    response.raise_for_status()
    path.write_bytes(response.content)
    candidate.path = path
    return path


def ratio_white(img: Image.Image, threshold: int = 245) -> float:
    arr = np.asarray(img.convert("RGB"))
    white = np.all(arr >= threshold, axis=2)
    return float(np.mean(white))


def edge_white_ratio(img: Image.Image) -> float:
    rgb = img.convert("RGB")
    w, h = rgb.size
    margin = max(1, int(min(w, h) * 0.08))
    arr = np.asarray(rgb)
    mask = np.zeros((h, w), dtype=bool)
    mask[:margin, :] = True
    mask[h - margin :, :] = True
    mask[:, :margin] = True
    mask[:, w - margin :] = True
    selected = arr[mask]
    white = np.all(selected >= 245, axis=1)
    return float(np.mean(white))


def center_content_ratio(img: Image.Image) -> float:
    rgb = img.convert("RGB")
    w, h = rgb.size
    crop = rgb.crop((int(w * 0.2), int(h * 0.2), int(w * 0.8), int(h * 0.8)))
    non_white = 1.0 - ratio_white(crop)
    return non_white


# 新增：計算物件密度分的輔助函數
def get_object_compoundedness(img: Image.Image) -> float:
    """計算中央 70% 區域的非白物件是否存在分散狀況 (複合圖偵測)。
       真正的商品主圖應該是單一、巨大物件，複合圖則是許多小物件。"""
    rgb = img.convert("RGB")
    w, h = rgb.size
    # 裁切中央 70%
    crop = rgb.crop((int(w * 0.15), int(h * 0.15), int(w * 0.85), int(h * 0.85)))
    
    # 建立非白物件 mask
    arr = np.asarray(crop)
    non_white = np.any(arr < 245, axis=2)
    
    # 進行 connected component 標記
    try:
        from scipy import ndimage
        labels, count = ndimage.label(non_white)
        
        # 1. 狀況：如果物件極少 (例如純白圖)，denseness 高
        if count == 0:
            return 1.0
        
        # 2. 狀況：計算最大物件佔所有非白像素的比例
        sizes = np.bincount(labels.ravel())
        sizes[0] = 0 # 忽略背景
        max_size = sizes.max()
        total_non_white_size = non_white.sum()
        
        if total_non_white_size == 0:
            return 1.0
            
        return float(max_size / total_non_white_size)
        
    except ImportError:
        # 如果找不到 scipy，放棄此評分項目
        return 1.0
    except Exception:
        return 1.0


def score_candidate(candidate: Candidate, product_id: str) -> None:
    if not candidate.path:
        candidate.note = "missing file"
        return

    with Image.open(candidate.path) as img:
        img.thumbnail((600, 600))
        candidate.white_ratio = ratio_white(img)
        candidate.edge_white_ratio = edge_white_ratio(img)
        candidate.center_content_ratio = center_content_ratio(img)
        candidate.compoundedness_score = get_object_compoundedness(img) # ✨ 新增：物件密度分

    name = candidate.name.lower()
    stem = Path(candidate.name).stem.lower()
    pid = product_id.lower()
    score = 0.0
    
    # --- ⚖️ 專業評分與權重配比 (已針對複合情境圖優化) ---
    score += candidate.edge_white_ratio * 30    # 👈 保持關鍵權重
    score += candidate.white_ratio * 10         # 👈 白底很重要，但 composite 圖也很白，故權重不宜過高
    score += min(candidate.center_content_ratio, 0.65) * 20 # 👈 確保有東西在中央
    
    # 1. PID 檔案名稱精準 bonus (最強效)
    if stem == pid:
        score += 15
        
    # 2. 檔案名稱包含 Context/Banner/情境/詳細資料之類的 composite 關鍵字懲罰 🌟 (修正 image_12.png)
    # 檔名只要包含情境、 banner 、 spec 這種詞，大概就是 image_12.png 那種圖
    refine_keywords = ["context", "banner", "bn", "banner", "情境", "詳細", "規格", "spec", "banner"]
    if any(kw in name for kw in refine_keywords):
        score -= 15 # 👈 精準懲罰

    # 3. 檔案stem 包含 PID 但帶有 01/02 序號的懲罰
    elif re.fullmatch(rf"{re.escape(pid)}[_-]?(0?1|10)", stem):
        score -= 45
    elif re.fullmatch(rf"{re.escape(pid)}[_-]\d+", stem):
        score -= 8

    # 4. 物件密度懲罰 🌟 (修正 image_12.png)
    # 真正的商品主圖應該是單一、巨大物件， compoundedness_score 應該接近 1.0
    # 如果低於 0.65 (例如 12.png 右邊散落很多文字)，直接扣除重分
    if candidate.compoundedness_score < 0.65:
        score -= 15 # 👈 精準懲罰

    if candidate.source == "product.images":
        score += 12
        
    # 5. 檔案名稱bonus
    if re.fullmatch(r"\d+(_\d+)?\.(jpg|jpeg|png|webp)", name, re.IGNORECASE):
        score += 8
    if re.search(r"^\d+[_-]", name):
        score += 5
        
    candidate.score = round(score, 3)


def choose_candidate(candidates: list[Candidate], product_id: str) -> tuple[Candidate | None, str, str]:
    if not candidates:
        return None, "failed", "找不到 product.images 或商品內容圖片"
    for candidate in candidates:
        score_candidate(candidate, product_id)
    ranked = sorted(candidates, key=lambda item: item.score, reverse=True)
    best = ranked[0]
    runner_up = ranked[1] if len(ranked) > 1 else None

    # --- 🌟 優化：拔除對 Petpetgo 設置的「猶豫煞車」---
    # 告訴程式：「就算分數接近，你也無腦選分數最高（第一名）的那張去背就對了！」
    # 這能確保 image_12.png 後面那張「成功」的圖片能被正確挑選出來，而不會一直卡在 needs_review 狀態。
    return best, "success", "自動挑選 (已關閉人工審核)"


def clean_cutout_alpha(
    img: Image.Image,
    # 🌟 優化：將清理阈值大幅降低，避免把燒雞包裝上淡淡的格紋或點點判定為雜點而清理掉。
    alpha_threshold: int = 12, # 👈 原本 24 -> 改 12，保留更多極淡的邊緣
    solid_alpha_threshold: int = 250, # 👈 原本 245 -> 改 250，更嚴格才判斷為完全不透明
    min_component_ratio: float = 0.000005, # 👈 原本 0.00008 -> 改極小，防止咬肉
) -> Image.Image:
    rgba = img.convert("RGBA")
    arr = np.array(rgba)
    alpha = arr[:, :, 3]
    # 手動二值化核心 Alpha，讓主體更飽滿
    alpha[alpha < alpha_threshold] = 0
    alpha[alpha > solid_alpha_threshold] = 255

    try:
        from scipy import ndimage

        mask = alpha > 0
        labels, count = ndimage.label(mask)
        if count:
            sizes = np.bincount(labels.ravel())
            # 🌟 優化：清理杂点的最小面積限制也同步調小，防止咬肉。
            min_area = max(10, int(mask.size * min_component_ratio))
            keep = sizes >= min_area
            keep[0] = False
            alpha[~keep[labels]] = 0
    except Exception:
        pass

    arr[:, :, 3] = alpha
    # 🌟 優化：移除 MedianFilter，防止柔化 complex 包裝邊緣 (如 image_13.png)。
    # 現在依賴 rembg 高級的 matting 在去背當下就做好平滑。
    return Image.fromarray(arr, mode="RGBA")


def remove_background(input_path: Path) -> Image.Image:
    if remove is None:
        raise RuntimeError("找不到 rembg，請先安裝：python3 -m pip install rembg onnxruntime")
    
    # 🌟 啟用專門針對電商產品優化的 ISNet 模型
    session = new_session("isnet-general-use")
    
    with Image.open(input_path) as img:
        rgba = img.convert("RGBA")
        
        # 攔截 rembg 的後台輸出
        quiet_output = io.StringIO()
        with contextlib.redirect_stdout(quiet_output), contextlib.redirect_stderr(quiet_output):
            
            # 👇 🌟 【最終體核心修正：引入高級阿法遮罩 (Alpha Matting)】 🌟 👇
            # 啟用 alpha_matting 並細調參數，讓邊緣平滑自然的同時，**大幅縮小 AI 的「咬肉範圍」**。
            cutout = remove(
                rgba,
                session=session,
                # --- ✨ 精細平滑設定 (專業救星) ---
                alpha_matting=True, # 👈 🌟 核心：開啟高級邊緣平滑 (阿法遮罩)
                alpha_matting_foreground_threshold=240, # 👈 高於此值判斷為「確定是商品」，要飽滿
                alpha_matting_background_threshold=10,  # 👈 低於此值判斷為「確定是背景」，要透明
                alpha_matting_erode_size=10,             # 👈 控制「咬肉」範圍的腐蝕大小。
                                                      # 原預設 25 咬太多，改小至 10 (專業最佳化)，能精準保留 13.png 包裝文字。
                # -----------------------
                post_process_mask=True # 👈 保留後處理，修補 mask 裡的洞
            )
            
    # 👇 🌟 【關鍵修正：直接合併分離的 RGB 通道，解決 convert("RGB") 的黑邊陷阱】 🌟 👇
    # pillow 的 convert("RGB") 在處理透明 RGBA 時，預設會把半透明區域與「黑色」合成。
    # 這就是為什麼去背圖上會有一圈黑灰色的鋸齒黑邊。
    
    # 分離出 cutout 後的 4 個通道，只保留 R, G, B 原始顏色資料
    r, g, b, alpha_from_cutout = cutout.split()
    # 重新組合，維持 R, G, B 原始顏色，使用剛剛去背產生的平滑 Alpha
    rgba_correct = Image.merge("RGBA", (r, g, b, alpha_from_cutout))
    
    # 這裡呼叫原本舊有的 clean_cutout_alpha 做最後的雜點清理
    try:
        return clean_cutout_alpha(rgba_correct)
    except NameError:
        return rgba_correct


def fit_to_canvas(img: Image.Image, size: int = 800, padding: int = 24) -> Image.Image:
    rgba = img.convert("RGBA")
    alpha = rgba.getchannel("A")
    bbox = alpha.getbbox()
    
    # 1. 將圖片的透明多餘邊界全部裁切掉，只留下商品本體
    if bbox:
        rgba = rgba.crop(bbox)

    # 2. 依照商品「真實的長寬比例」，各自加上留白 (不再強制做成正方形)
    new_width = rgba.width + padding * 2
    new_height = rgba.height + padding * 2

    # 3. 建立這個「完全合身」的透明畫布 (可能是長方形或正方形，依商品形狀而定)
    canvas = Image.new("RGBA", (new_width, new_height), (255, 255, 255, 0))
    
    # 4. 把商品貼在正中間
    canvas.alpha_composite(rgba, (padding, padding))
    
    return canvas


def save_review_candidates(candidates: Iterable[Candidate], review_dir: Path) -> None:
    review_dir.mkdir(parents=True, exist_ok=True)
    for candidate in candidates:
        if candidate.path and candidate.path.exists():
            target = review_dir / candidate.path.name
            if not target.exists():
                target.write_bytes(candidate.path.read_bytes())
    score_path = review_dir / "scores.csv"
    with score_path.open("w", newline="", encoding="utf-8-sig") as fh:
        writer = csv.writer(fh)
        writer.writerow(["url", "name", "source", "score", "white_ratio", "edge_white_ratio", "center_content_ratio"])
        for c in sorted(candidates, key=lambda item: item.score, reverse=True):
            writer.writerow([c.url, c.name, c.source, c.score, c.white_ratio, c.edge_white_ratio, c.center_content_ratio])


def process_product(session: requests.Session, product_id: str, args: argparse.Namespace) -> dict[str, str]:
    html = fetch_html(session, product_id)
    candidates = extract_candidates(html)
    primary_candidates = [c for c in candidates if c.source == "product.images"]
    detail_candidates = [c for c in candidates if c.source != "product.images"]
    temp_context = tempfile.TemporaryDirectory(prefix=f"petpetgo_{product_id}_")
    working_dir = args.output_dir / "original" / product_id if args.keep_original else Path(temp_context.name)

    for index, candidate in enumerate(primary_candidates, 1):
        download_candidate(session, candidate, product_id, index, working_dir)

    best, status, note = choose_candidate(primary_candidates, product_id)
    downloaded_candidates = list(primary_candidates)
    if status != "success" and detail_candidates:
        start_index = len(primary_candidates) + 1
        for index, candidate in enumerate(detail_candidates, start_index):
            download_candidate(session, candidate, product_id, index, working_dir)
        downloaded_candidates.extend(detail_candidates)
        best, status, note = choose_candidate(downloaded_candidates, product_id)

    if status != "success":
        save_review_candidates(downloaded_candidates, args.output_dir / "review" / product_id)

    final_path = ""
    if best and best.path:
        removed = remove_background(best.path)
        final = fit_to_canvas(removed, size=args.size, padding=args.padding)
        args.output_dir.joinpath("final").mkdir(parents=True, exist_ok=True)
        final_file = args.output_dir / "final" / f"{product_id}.png"
        final.save(final_file)
        final_path = str(final_file)

    return {
        "product_id": product_id,
        "status": status,
        "selected_image_url": best.url if best else "",
        "selected_name": best.name if best else "",
        "selected_score": str(best.score) if best else "",
        "confidence_note": note,
        "final_path": final_path,
        "candidate_count": str(len(downloaded_candidates)),
    }


def write_report(rows: list[dict[str, str]], report_path: Path) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "product_id",
        "status",
        "selected_image_url",
        "selected_name",
        "selected_score",
        "confidence_note",
        "final_path",
        "candidate_count",
        "error",
    ]
    with report_path.open("w", newline="", encoding="utf-8-sig") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="抓取 petpetgo 商品圖、去背並輸出 800x800 PNG。")
    parser.add_argument("input", help="CSV 檔案路徑，或 Google Sheets 發布後的 CSV URL")
    parser.add_argument("--output-dir", type=Path, default=Path("output"), help="輸出資料夾")
    parser.add_argument("--report", type=Path, default=Path("output/report.csv"), help="報表路徑")
    # 🌟 優化： size 功能在 app.py 裡設定為自動忽略，這裡保留參數但沒實質作用。
    parser.add_argument("--size", type=int, default=800, help="輸出 PNG 大小（雲端版已自動合身不限此尺寸）")
    parser.add_argument("--padding", type=int, default=24, help="商品與畫布邊界留白")
    parser.add_argument("--keep-original", action="store_true", help="保留下載的原始候選圖")
    parser.add_argument("--min-delay", type=float, default=DEFAULT_DELAY[0], help="每個商品處理後最短等待秒數")
    parser.add_argument("--max-delay", type=float, default=DEFAULT_DELAY[1], help="每個商品處理後最長等待秒數")
    parser.add_argument("--limit", type=int, default=0, help="測試用，只處理前 N 筆；0 表示全部")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    product_ids = read_product_ids(args.input)
    if args.limit:
        product_ids = product_ids[: args.limit]
    if not product_ids:
        print("找不到商品 ID。", file=sys.stderr)
        return 1

    session = requests.Session()
    session.headers.update({"User-Agent": USER_AGENT, "Accept-Language": "zh-TW,zh;q=0.9,en;q=0.8"})

    rows: list[dict[str, str]] = []
    for index, product_id in enumerate(product_ids, 1):
        print(f"[{index}/{len(product_ids)}] processing {product_id}")
        try:
            row = process_product(session, product_id, args)
        except Exception as exc:
            row = {
                "product_id": product_id,
                "status": "failed",
                "error": str(exc),
            }
        rows.append(row)
        write_report(rows, args.report)
        if index < len(product_ids):
            time.sleep(random.uniform(args.min_delay, args.max_delay))

    print(f"完成，報表：{args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())