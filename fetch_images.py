#!/usr/bin/env python3
import argparse
import contextlib
import csv
import io
import json
import os
import random
import re
import sys
import tempfile
import time
import traceback
import gc
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
from urllib.parse import urlparse

import requests
from bs4 import BeautifulSoup
import numpy as np
from PIL import Image, ImageFilter

os.environ.setdefault("NUMBA_CACHE_DIR", str(Path(__file__).resolve().parent / ".numba_cache"))

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
MAX_REMBG_INPUT_SIDE = 1800
_rembg_session = None


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
    bottom_band_ratio: float = 0.0
    bottom_band_span: float = 0.0
    side_white_min: float = 0.0
    note: str = ""
    reject_reason: str = ""


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
            match = re.search(r"\d+", cell)
            if match:
                ids.append(match.group(0))
                break
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


def get_rembg_session():
    global _rembg_session
    if new_session is None:
        raise RuntimeError("找不到 rembg，請先安裝：python3 -m pip install rembg onnxruntime")
    if _rembg_session is None:
        _rembg_session = new_session("u2net")
    return _rembg_session


def load_background_input(input_path: Path) -> Image.Image:
    with Image.open(input_path) as img:
        rgba = img.convert("RGBA")
    if max(rgba.size) > MAX_REMBG_INPUT_SIDE:
        rgba.thumbnail((MAX_REMBG_INPUT_SIDE, MAX_REMBG_INPUT_SIDE), Image.Resampling.LANCZOS)
    return rgba


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


def side_white_ratios(img: Image.Image) -> dict[str, float]:
    rgb = img.convert("RGB")
    w, h = rgb.size
    margin = max(1, int(min(w, h) * 0.08))
    regions = {
        "top": rgb.crop((0, 0, w, margin)),
        "bottom": rgb.crop((0, h - margin, w, h)),
        "left": rgb.crop((0, 0, margin, h)),
        "right": rgb.crop((w - margin, 0, w, h)),
    }
    return {side: ratio_white(region) for side, region in regions.items()}


def bottom_band_metrics(img: Image.Image) -> tuple[float, float]:
    rgb = img.convert("RGB")
    w, h = rgb.size
    band_h = max(1, int(h * 0.18))
    band = rgb.crop((0, h - band_h, w, h))
    arr = np.asarray(band)
    non_white = np.any(arr < 245, axis=2)
    non_white_ratio = float(np.mean(non_white))
    column_span = float(np.mean(np.any(non_white, axis=0)))
    return non_white_ratio, column_span


def center_content_ratio(img: Image.Image) -> float:
    rgb = img.convert("RGB")
    w, h = rgb.size
    crop = rgb.crop((int(w * 0.2), int(h * 0.2), int(w * 0.8), int(h * 0.8)))
    non_white = 1.0 - ratio_white(crop)
    return non_white


def get_object_compoundedness(img: Image.Image) -> float:
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
        candidate.bottom_band_ratio, candidate.bottom_band_span = bottom_band_metrics(img)
        side_ratios = side_white_ratios(img)
        candidate.side_white_min = min(side_ratios.values())

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

    if candidate.bottom_band_ratio > 0.28 and candidate.bottom_band_span > 0.72:
        score -= 120
    if candidate.side_white_min < 0.45:
        score -= 70

    if candidate.source == "product.images":
        score += 80
        
    if re.fullmatch(r"\d+(_\d+)?\.(jpg|jpeg|png|webp)", name, re.IGNORECASE):
        score += 8
    if re.search(r"^\d+[_-]", name):
        score += 5
        
    candidate.score = round(score, 3)


def review_reason(candidate: Candidate) -> str:
    reasons: list[str] = []
    stem = Path(candidate.name).stem.lower()
    if candidate.bottom_band_ratio > 0.28 and candidate.bottom_band_span > 0.72:
        reasons.append("底部疑似有橫向色塊或文案條")
    if candidate.side_white_min < 0.45:
        reasons.append("圖片邊界不是穩定白底")
    if candidate.edge_white_ratio < 0.68:
        reasons.append("四周白底比例偏低")
    if candidate.compoundedness_score < 0.62:
        reasons.append("疑似有商品以外的文字或多個區塊")
    if re.search(r"[_-](0?1|10)$", stem):
        reasons.append("檔名像主視覺/文案圖")
    return "；".join(reasons)


def choose_candidate(candidates: list[Candidate], product_id: str) -> tuple[Candidate | None, str, str]:
    if not candidates:
        return None, "failed", "找不到 product.images 或商品內容圖片"
    for candidate in candidates:
        score_candidate(candidate, product_id)
    ranked = sorted(candidates, key=lambda item: item.score, reverse=True)
    best = ranked[0]
    reason = review_reason(best)
    runner_up = ranked[1] if len(ranked) > 1 else None
    if runner_up and best.score - runner_up.score < 10:
        reason = "前兩名候選圖分數接近" if not reason else f"{reason}；前兩名候選圖分數接近"
    if reason:
        best.reject_reason = reason
        return best, "needs_review", reason
    return best, "success", "自動挑選"


def remove_background(input_path: Path) -> Image.Image:
    original_rgba = load_background_input(input_path)
    if should_use_edge_white_removal(original_rgba):
        return remove_white_background_from_edges(input_path)

    if remove is None:
        raise RuntimeError("找不到 rembg，請先安裝：python3 -m pip install rembg onnxruntime")

    session = get_rembg_session()
    source_rgba = original_rgba.copy()
        
    quiet_output = io.StringIO()
    with contextlib.redirect_stdout(quiet_output), contextlib.redirect_stderr(quiet_output):
        # 取得純黑白遮罩
        mask = remove(
            original_rgba,
            session=session,
            only_mask=True,
            post_process_mask=True
        )
            
    mask = mask.convert("L")
    
    # 🌟【超級關鍵：消除原圖白底帶來的白邊】🌟
    # 先使用 MinFilter(3) 讓白色遮罩向內收縮（侵蝕）約 1 像素，把原圖邊緣的白底像素徹底切除
    eroded_mask = mask.filter(ImageFilter.MinFilter(3))
    
    # 收縮完後，再做非常輕微的模糊，保證邊緣滑順、不生硬、不鋸齒
    smoothed_mask = eroded_mask.filter(ImageFilter.GaussianBlur(radius=0.6))
    
    # 將這張完美乾淨的遮罩放回原圖上
    original_rgba.putalpha(smoothed_mask)

    if should_use_edge_white_fallback(source_rgba, original_rgba):
        return remove_white_background_from_edges(input_path)
    
    return original_rgba


def should_use_edge_white_removal(source_rgba: Image.Image) -> bool:
    edge_ratio = edge_white_ratio(source_rgba)
    side_min = min(side_white_ratios(source_rgba).values())
    white_ratio = ratio_white(source_rgba)
    return edge_ratio >= 0.72 and side_min >= 0.55 and white_ratio >= 0.18


def colored_content_loss_ratio(source_rgba: Image.Image, removed_rgba: Image.Image) -> float:
    source = np.array(source_rgba.convert("RGBA"))
    removed_alpha = np.array(removed_rgba.convert("RGBA"))[:, :, 3]
    source_alpha = source[:, :, 3]
    rgb = source[:, :, :3].astype(np.int16)

    channel_range = np.max(rgb, axis=2) - np.min(rgb, axis=2)
    brightness = np.mean(rgb, axis=2)
    saturated_content = (channel_range > 18) & np.any(rgb < 248, axis=2)
    neutral_content = brightness < 225
    source_content = (source_alpha > 200) & (saturated_content | neutral_content)

    content_pixels = int(np.sum(source_content))
    if content_pixels < 500:
        return 0.0

    lost_pixels = int(np.sum(source_content & (removed_alpha < 48)))
    return lost_pixels / content_pixels


def should_use_edge_white_fallback(source_rgba: Image.Image, removed_rgba: Image.Image) -> bool:
    if edge_white_ratio(source_rgba) < 0.55:
        return False
    return colored_content_loss_ratio(source_rgba, removed_rgba) > 0.08


def remove_white_background_from_edges(
    input_path: Path,
    white_threshold: int = 246,
    tolerance: int = 14,
    soften_radius: float = 0.6,
) -> Image.Image:
    rgba = load_background_input(input_path)

    arr = np.array(rgba)
    rgb = arr[:, :, :3].astype(np.int16)
    h, w = rgb.shape[:2]

    near_white = np.all(rgb >= white_threshold, axis=2)
    soft_white = (
        np.min(rgb, axis=2) >= white_threshold - tolerance
    ) & (
        np.max(rgb, axis=2) - np.min(rgb, axis=2) <= tolerance * 2
    )
    background_like = near_white | soft_white

    visited = np.zeros((h, w), dtype=bool)
    queue: list[tuple[int, int]] = []

    for x in range(w):
        if background_like[0, x]:
            visited[0, x] = True
            queue.append((0, x))
        if background_like[h - 1, x]:
            visited[h - 1, x] = True
            queue.append((h - 1, x))
    for y in range(h):
        if background_like[y, 0]:
            visited[y, 0] = True
            queue.append((y, 0))
        if background_like[y, w - 1]:
            visited[y, w - 1] = True
            queue.append((y, w - 1))

    head = 0
    while head < len(queue):
        y, x = queue[head]
        head += 1
        for ny, nx in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
            if 0 <= ny < h and 0 <= nx < w and not visited[ny, nx] and background_like[ny, nx]:
                visited[ny, nx] = True
                queue.append((ny, nx))

    alpha = np.full((h, w), 255, dtype=np.uint8)
    alpha[visited] = 0
    alpha_img = Image.fromarray(alpha, mode="L")
    if soften_radius > 0:
        alpha_img = alpha_img.filter(ImageFilter.GaussianBlur(radius=soften_radius))

    result = rgba.copy()
    result.putalpha(alpha_img)
    return result


def fit_to_canvas(img: Image.Image, size: int = 800, padding: int = 24) -> Image.Image:
    img = crop_to_content(img, alpha_threshold=80)

    max_side = size - padding * 2
    img.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)

    canvas = Image.new("RGBA", (size, size), (255, 255, 255, 0))
    left = (size - img.width) // 2
    top = (size - img.height) // 2
    canvas.alpha_composite(img, (left, top))
    
    return canvas


def crop_to_content(img: Image.Image, padding: int = 0, alpha_threshold: int = 80) -> Image.Image:
    img = img.convert("RGBA")
    arr = np.array(img)
    alpha = arr[:, :, 3]

    rows = np.any(alpha >= alpha_threshold, axis=1)
    cols = np.any(alpha >= alpha_threshold, axis=0)

    if not rows.any() or not cols.any():
        return img

    ymin, ymax = np.where(rows)[0][[0, -1]]
    xmin, xmax = np.where(cols)[0][[0, -1]]
    xmin = max(0, xmin - padding)
    ymin = max(0, ymin - padding)
    xmax = min(img.width - 1, xmax + padding)
    ymax = min(img.height - 1, ymax + padding)
    return img.crop((xmin, ymin, xmax + 1, ymax + 1))


def add_transparent_padding(img: Image.Image, padding: int = 0) -> Image.Image:
    img = img.convert("RGBA")
    if padding <= 0:
        return img
    canvas = Image.new("RGBA", (img.width + padding * 2, img.height + padding * 2), (255, 255, 255, 0))
    canvas.alpha_composite(img, (padding, padding))
    return canvas


def fit_to_product_bounds(img: Image.Image, size: int = 800, padding: int = 12) -> Image.Image:
    img = crop_to_content(img, alpha_threshold=96)
    arr = np.array(img)
    arr[:, :, 3] = np.where(arr[:, :, 3] < 16, 0, arr[:, :, 3])
    img = Image.fromarray(arr, mode="RGBA")
    if max(img.size) > size:
        img.thumbnail((size, size), Image.Resampling.LANCZOS)
    img = crop_to_content(img, alpha_threshold=32)
    return add_transparent_padding(img, padding=padding)


def make_output_image(img: Image.Image, args: argparse.Namespace) -> Image.Image:
    output_mode = getattr(args, "output_mode", "square")
    if output_mode == "product":
        return fit_to_product_bounds(img, size=args.size, padding=args.padding)
    return fit_to_canvas(img, size=args.size, padding=args.padding)


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
        writer.writerow([
            "url",
            "name",
            "source",
            "score",
            "white_ratio",
            "edge_white_ratio",
            "center_content_ratio",
            "compoundedness_score",
            "bottom_band_ratio",
            "bottom_band_span",
            "side_white_min",
            "review_reason",
        ])
        for c in sorted(candidates, key=lambda item: item.score, reverse=True):
            writer.writerow([
                c.url,
                c.name,
                c.source,
                c.score,
                c.white_ratio,
                c.edge_white_ratio,
                c.center_content_ratio,
                c.compoundedness_score,
                c.bottom_band_ratio,
                c.bottom_band_span,
                c.side_white_min,
                review_reason(c),
            ])


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
    if status == "success" and best and best.path:
        removed = remove_background(best.path)
        final = make_output_image(removed, args)
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
        "reject_reason": best.reject_reason if best else note,
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
        "reject_reason",
        "final_path",
        "candidate_count",
        "error",
    ]
    with report_path.open("w", newline="", encoding="utf-8-sig") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def write_error_log(product_id: str, exc: Exception, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "error_log.txt"
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    with log_path.open("a", encoding="utf-8") as fh:
        fh.write(f"[{timestamp}] product_id={product_id}\n")
        fh.write(f"{type(exc).__name__}: {exc}\n")
        fh.write(traceback.format_exc())
        fh.write("\n---\n")


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
            write_error_log(product_id, exc, args.output_dir)
            row = {
                "product_id": product_id,
                "status": "failed",
                "error": str(exc),
            }
        finally:
            gc.collect()
        rows.append(row)
        write_report(rows, args.report)
        if index < len(product_ids):
            time.sleep(random.uniform(args.min_delay, args.max_delay))

    print(f"完成，報表：{args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
