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
    compoundedness_score: float = 1.0 
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


def get_object_compoundedness(img: Image.Image) -> float:
    """計算中央 70% 區域的非白物件是否存在分散狀況 (複合圖偵測)。"""
    rgb = img.convert("RGB")
    w, h = rgb.size
    crop = rgb.crop((int(w * 0.15), int(h * 0.15), int(w * 0.85), int(h * 0.85)))
    arr = np.asarray(crop)
    non_white = np.any(arr < 245, axis=2)
    
    try:
        from scipy import ndimage
        labels, count = ndimage.label(non_white)
        if count == 0:
            return 1.0
        
        sizes = np.bincount(labels.ravel())
        sizes[0] = 0 
        max_size = sizes.max()
        total_non_white_size = non_white.sum()
        
        if total_non_white_size == 0:
            return 1.0
            
        return float(max_size / total_non_white_size)
        
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
        candidate.compoundedness_score = get_object_compoundedness(img)

    name = candidate.name.lower()
    stem = Path(candidate.name).stem.lower()
    pid = product_id.lower()
    score = 0.0
    
    score += candidate.edge_white_ratio * 30    
    score += candidate.white_ratio * 10         
    score += min(candidate.center_content_ratio, 0.65) * 20 
    
    if stem == pid:
        score += 15
        
    harsh_context_keywords = ["context", "banner", "bn", "情境", "詳細", "詳細資料", "規格", "spec", "specs", "detail", "details", "size"]
    if any(kw in name for kw in harsh_context_keywords):
        score -= 80 

    elif re.fullmatch(rf"{re.escape(pid)}[_-]?(0?1|10)", stem):
        score -= 45
    elif re.fullmatch(rf"{re.escape(pid)}[_-]\d+", stem):
        score -= 8

    if candidate.compoundedness_score < 0.65:
        score -= 80 

    if candidate.source == "product.images":
        score += 80
        
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
    return best, "success", "自動挑選 (已關閉人工審核)"


def clean_cutout_alpha(
    img: Image.Image,
    alpha_threshold: int = 12,
    solid_alpha_threshold: int = 250,
) -> Image.Image:
    rgba = img.convert("RGBA")
    arr = np.array(rgba)
    alpha = arr[:, :, 3]
    
    alpha[alpha < alpha_threshold] = 0
    alpha[alpha > solid_alpha_threshold] = 255

    # 移除 scipy 過度清理雜點的機制，以防咬肉。
    arr[:, :, 3] = alpha
    return Image.fromarray(arr, mode="RGBA")


def remove_background(input_path: Path) -> Image.Image:
    if remove is None:
        raise RuntimeError("找不到 rembg，請先安裝：python3 -m pip install rembg onnxruntime")
    
    session = new_session("isnet-general-use")
    
    with Image.open(input_path) as img:
        rgba = img.convert("RGBA")
        
        quiet_output = io.StringIO()
        with contextlib.redirect_stdout(quiet_output), contextlib.redirect_stderr(quiet_output):
            
            # 使用高級 Alpha Matting，並且將erode設為 0 以防止咬肉
            cutout = remove(
                rgba,
                session=session,
                alpha_matting=True, 
                alpha_matting_foreground_threshold=240, 
                alpha_matting_background_threshold=10,  
                alpha_matting_erode_size=0, # 完全禁用腐蝕防咬肉
                post_process_mask=True
            )
            
    # 防止黑邊：提取原色的RGB，只換掉 Alpha 通道
    r, g, b, alpha_from_cutout = cutout.split()
    rgba_correct = Image.merge("RGBA", (r, g, b, alpha_from_cutout))
    
    return clean_cutout_alpha(rgba_correct)


def fit_to_canvas(img: Image.Image, size: int = 800, padding: int = 24) -> Image.Image:
    rgba = img.convert("RGBA")
    alpha = rgba.getchannel("A")
    bbox = alpha.getbbox()
    
    if bbox:
        rgba = rgba.crop(bbox)

    new_width = rgba.width + padding * 2
    new_height = rgba.height + padding * 2

    canvas = Image.new("RGBA", (new_width, new_height), (255, 255, 255, 0))
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
    parser.add_argument("--size", type=int, default=800, help="輸出 PNG 大小")
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