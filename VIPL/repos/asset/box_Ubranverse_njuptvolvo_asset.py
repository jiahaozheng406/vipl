"""
box_Ubranverse_njuptvolvo_asset.py
用 OBB 比例从 UrbanVerse-100K 匹配最佳 3D 车辆资产，下载 GLB 并输出预览图。

流程: 读取 OBB → 提取 extent 比例 → UrbanVerse 按类别+语义查询获取候选
    → 下载 GLB → 从 GLB 读取真实 bbox 尺寸 → 按比例相似度排序 → 输出最佳资产预览
"""

import json
import logging
from pathlib import Path

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger(__name__)

# ─── 路径配置 ───
OBB_JSON = r"E:\vipl\VIPL\repos\box\outputs\njuptVolvo\boxes\car_obb.json"
OUTPUT_DIR = Path(r"E:\vipl\VIPL\repos\asset\outputs\njuptVolvo")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
CACHE_DIR = Path(r"E:\vipl\VIPL\repos\asset\outputs\urbanverse_cache")

# ─── 查询参数 ───
QUERY_TEXT = "gray sedan car Volvo"
CANDIDATE_K = 20   # 语义查询候选数
BEST_K = 5          # 最终输出 top-5


def extent_to_ratio(ext):
    """将 extent [a, b, c] 归一化为比例向量 (按最大值归一)"""
    ext = np.array(sorted(ext, reverse=True), dtype=np.float64)
    return ext / ext[0]


def ratio_distance(r1, r2):
    """两个比例向量的欧氏距离 (越小越相似)"""
    return np.linalg.norm(r1 - r2)


def glb_bbox_extent(glb_path):
    """从 GLB 文件读取 axis-aligned bounding box 的 extent [长, 宽, 高]"""
    import trimesh
    scene = trimesh.load(glb_path, force='scene')
    bbox = scene.bounding_box.extents
    return sorted(bbox, reverse=True)


def main():
    # ── Stage 1: 读取 OBB，提取比例 ──
    log.info("Stage 1: 读取 OBB 并计算 extent 比例")
    with open(OBB_JSON, "r") as f:
        obb = json.load(f)
    extent = obb["extent"]
    obb_ratio = extent_to_ratio(extent)
    log.info(f"  OBB extent: {extent}")
    log.info(f"  OBB 比例 (归一化): {obb_ratio.tolist()}")

    # ── Stage 2: UrbanVerse 语义查询获取候选 (不限尺寸) ──
    log.info("Stage 2: UrbanVerse 语义查询")
    import urbanverse_asset as uva

    uva.set(str(CACHE_DIR))

    candidate_uids = uva.object.get_uids_conditioned(
        categories=["vehicle"],
        query=QUERY_TEXT,
        top_k=CANDIDATE_K,
    )
    log.info(f"  语义候选: {len(candidate_uids)} 个")
    if not candidate_uids:
        log.error("未找到候选资产")
        return

    # ── Stage 3: 下载 GLB + annotation + thumbnail ──
    log.info("Stage 3: 下载资产")
    result = uva.object.load(candidate_uids, what=("std_glb", "std_annotation", "thumbnail"))

    # ── Stage 4: 从 GLB 读取真实 bbox，按比例相似度排序 ──
    log.info("Stage 4: 比例匹配排序")
    scored = []
    for uid in candidate_uids:
        info = result.get(uid, {})
        glb_path = info.get("std_glb")
        if not glb_path or not Path(glb_path).exists():
            continue
        try:
            glb_ext = glb_bbox_extent(glb_path)
            glb_ratio = extent_to_ratio(glb_ext)
            dist = ratio_distance(obb_ratio, glb_ratio)
            scored.append((uid, dist, glb_ext))
            log.info(f"  {uid}: bbox={[f'{x:.2f}' for x in glb_ext]}, "
                      f"ratio={glb_ratio.round(3).tolist()}, dist={dist:.4f}")
        except Exception as e:
            log.warning(f"  {uid}: 读取 GLB 失败 - {e}")

    scored.sort(key=lambda x: x[1])
    best_uids = [s[0] for s in scored[:BEST_K]]
    log.info(f"  Top-{BEST_K} 比例最匹配:")
    for rank, (uid, dist, ext) in enumerate(scored[:BEST_K], 1):
        log.info(f"    [{rank}] {uid} dist={dist:.4f} bbox={[f'{x:.2f}' for x in ext]}")

    # ── Stage 5: 输出最佳资产预览 ──
    best_uid = best_uids[0]
    log.info(f"Stage 5: 最佳资产 UID = {best_uid}")
    best_info = result.get(best_uid, {})
    from PIL import Image

    # 保存最佳 thumbnail
    thumb_path = best_info.get("thumbnail")
    if thumb_path and Path(thumb_path).exists():
        img = Image.open(thumb_path)
        out_thumb = OUTPUT_DIR / f"best_{best_uid}_thumbnail.png"
        img.save(str(out_thumb))
        log.info(f"  缩略图: {out_thumb}")

    # 保存 top-5 拼图
    thumbs = []
    for uid in best_uids:
        tp = result.get(uid, {}).get("thumbnail")
        if tp and Path(tp).exists():
            thumbs.append((uid, Image.open(tp)))
    if thumbs:
        w = max(t.width for _, t in thumbs)
        h_total = sum(t.height for _, t in thumbs)
        grid = Image.new("RGB", (w, h_total), (255, 255, 255))
        y = 0
        for uid, t in thumbs:
            grid.paste(t, (0, y))
            y += t.height
        out_grid = OUTPUT_DIR / "top5_candidates.png"
        grid.save(str(out_grid))
        log.info(f"  Top-5 拼图: {out_grid}")

    # 保存结果 JSON
    best_ann_path = best_info.get("std_annotation")
    best_ann = {}
    if best_ann_path and Path(best_ann_path).exists():
        with open(best_ann_path, "r", encoding="utf-8") as f:
            best_ann = json.load(f)

    summary = {
        "best_uid": best_uid,
        "glb_path": str(best_info.get("std_glb", "")),
        "annotation": best_ann,
        "obb_extent_vggt": extent,
        "obb_ratio": obb_ratio.tolist(),
        "query": QUERY_TEXT,
        "ranking": [
            {"uid": uid, "ratio_dist": round(dist, 4),
             "bbox_meters": [round(x, 3) for x in ext]}
            for uid, dist, ext in scored[:BEST_K]
        ],
    }
    out_json = OUTPUT_DIR / "best_asset.json"
    with open(str(out_json), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    log.info(f"  结果: {out_json}")
    log.info("完成!")


if __name__ == "__main__":
    main()
