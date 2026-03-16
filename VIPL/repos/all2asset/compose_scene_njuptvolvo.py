"""
compose_scene_njuptvolvo.py
读取 all2asset 输出的全部 OBB + best GLB，按 OBB 位姿将 3D asset 组装到同一场景。
核心: 用 car_0 的 OBB extent vs GLB 真实尺寸计算全局缩放因子，
      将整个 VGGT 坐标系标定到米制，所有物体自动对齐。
road 用 OBB 平面生成灰色路面 mesh。
输出:
  - scene_composed.glb   (可用任意 3D 查看器打开)
  - scene_composed.png   (多视角渲染预览)
"""
import json, logging, sys
from pathlib import Path
import numpy as np
import trimesh

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger(__name__)

BASE = Path(r"E:\vipl\VIPL\repos\all2asset\outputs\njuptVolvo")
OUT = BASE / "scene"
OUT.mkdir(parents=True, exist_ok=True)

# ── 工具函数 ──────────────────────────────────────────────

def load_json(p):
    with open(p) as f:
        return json.load(f)

def compute_global_scale():
    """用 car_0 的 OBB extent 和 GLB 真实 bbox 计算 VGGT→米 的缩放因子"""
    obb = load_json(BASE / "car/boxes/car_0_obb.json")
    asset = load_json(BASE / "car/asset/car_0_asset.json")
    obb_ext = np.array(sorted(obb["extent"], reverse=True))
    # GLB 真实尺寸从 ranking 或 annotation 获取
    ranking = asset.get("ranking", [])
    if ranking:
        glb_real = np.array(sorted(ranking[0]["bbox_meters"], reverse=True))
    else:
        ann = asset["annotation"]
        glb_real = np.array(sorted([ann["length"], ann["width"], ann["height"]], reverse=True))
    # 全局缩放 = 真实尺寸 / OBB尺寸 (取各轴平均)
    ratios = glb_real / (obb_ext + 1e-9)
    scale = float(np.mean(ratios))
    log.info(f"全局缩放因子: {scale:.4f} (OBB ext={obb_ext}, GLB real={glb_real})")
    return scale

def load_glb_mesh(glb_path):
    """加载 GLB 并合并为单个 Trimesh"""
    scene = trimesh.load(str(glb_path), force="scene")
    if isinstance(scene, trimesh.Scene):
        parts = [g for g in scene.geometry.values() if isinstance(g, trimesh.Trimesh)]
        if not parts:
            return None
        mesh = trimesh.util.concatenate(parts)
    else:
        mesh = scene
    return mesh

def place_glb_at_obb(glb_path, obb, global_scale):
    """
    将 GLB asset 放置到 OBB 位置。
    1. 居中 GLB 到原点
    2. 将 GLB 各轴与 OBB 各轴按尺寸大小对应
    3. 缩放 GLB 使其 bbox 匹配 OBB 的米制尺寸
    4. 旋转+平移到 OBB 世界位姿
    """
    mesh = load_glb_mesh(glb_path)
    if mesh is None:
        return None

    # 居中
    mesh.apply_translation(-mesh.bounding_box.centroid)

    glb_ext = np.array(mesh.bounding_box.extents)
    obb_ext = np.array(obb["extent"]) * global_scale  # 转为米制

    # 按尺寸降序配对轴
    glb_order = np.argsort(glb_ext)[::-1]  # GLB 最大→最小轴
    obb_order = np.argsort(obb_ext)[::-1]  # OBB 最大→最小轴

    # 构建轴重映射: GLB 的第 glb_order[i] 轴 → OBB 的第 obb_order[i] 轴
    # 先做一个排列矩阵让 GLB 轴顺序匹配 OBB 轴顺序
    perm = np.zeros((3, 3))
    for i in range(3):
        perm[obb_order[i], glb_order[i]] = 1.0
    mesh.apply_transform(np.vstack([
        np.hstack([perm, [[0],[0],[0]]]),
        [0, 0, 0, 1]
    ]))

    # 重新计算 extent (轴已重排)
    glb_ext_new = np.array(mesh.bounding_box.extents)
    scale_vec = obb_ext / (glb_ext_new + 1e-9)
    S = np.diag([*scale_vec, 1.0])
    mesh.apply_transform(S)

    # 应用 OBB 旋转 + 平移 (center 也要乘 global_scale)
    R = np.array(obb["R"])
    c = np.array(obb["center"]) * global_scale
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = c
    mesh.apply_transform(T)

    return mesh

def make_obb_box(obb, global_scale, color):
    """用 OBB 生成带颜色的 box mesh (用于 road/grass/building)"""
    c = np.array(obb["center"]) * global_scale
    R = np.array(obb["R"])
    ext = np.array(obb["extent"]) * global_scale
    half = ext / 2.0
    signs = np.array([
        [-1,-1,-1],[1,-1,-1],[1,1,-1],[-1,1,-1],
        [-1,-1, 1],[1,-1, 1],[1,1, 1],[-1,1, 1]
    ], dtype=float)
    local_pts = signs * half
    world_pts = (R @ local_pts.T).T + c
    box = trimesh.convex.convex_hull(world_pts)
    box.visual.face_colors = color
    return box

def collect_assets(global_scale):
    """遍历全部类别，收集 mesh 列表"""
    meshes = []

    # ── 有 GLB 的类别 ──
    for cat in ["car", "tree", "pole"]:
        asset_dir = BASE / cat / "asset"
        box_dir = BASE / cat / "boxes"
        if not asset_dir.exists():
            continue
        asset_jsons = sorted(asset_dir.glob(f"{cat}_*_asset.json"))
        if not asset_jsons:
            asset_jsons = sorted(asset_dir.glob(f"{cat}_asset.json"))
        for aj in asset_jsons:
            stem = aj.stem.replace("_asset", "")
            obb_path = box_dir / f"{stem}_obb.json"
            if not obb_path.exists():
                continue
            asset_info = load_json(aj)
            obb = load_json(obb_path)
            glb_path = asset_info.get("glb_path")
            if not glb_path or not Path(glb_path).exists():
                continue
            mesh = place_glb_at_obb(glb_path, obb, global_scale)
            if mesh is not None:
                log.info(f"  + {cat}/{stem}: {len(mesh.vertices)} verts")
                meshes.append(mesh)

    # ── 无 GLB 的类别: OBB → 几何体 ──
    color_map = {
        "road": [128, 128, 128, 255],
        "grass": [80, 160, 60, 255],
        "building": [200, 200, 210, 255],
    }
    for cat, color in color_map.items():
        box_dir = BASE / cat / "boxes"
        if not box_dir.exists():
            continue
        for oj in sorted(box_dir.glob(f"{cat}*_obb.json")):
            obb = load_json(oj)
            mesh = make_obb_box(obb, global_scale, color)
            if mesh is not None:
                log.info(f"  + {cat}/{oj.stem.replace('_obb','')}: box mesh")
                meshes.append(mesh)

    return meshes


def render_preview(scene_mesh, out_dir):
    """用 matplotlib 渲染多视角预览"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n_sample = min(80000, max(20000, len(scene_mesh.vertices) // 100))
    pts = scene_mesh.sample(n_sample)

    # 尝试获取采样点颜色
    try:
        pts, face_idx = scene_mesh.sample(n_sample, return_index=True)
        colors = scene_mesh.visual.face_colors[face_idx][:, :3] / 255.0
    except Exception:
        colors = pts[:, 2]  # fallback: 用 Z 高度着色

    fig = plt.figure(figsize=(20, 15))
    for idx, (elev, azim, title) in enumerate([
        (20, 45, "Front-Left"), (20, 135, "Back-Left"),
        (20, 225, "Back-Right"), (60, 90, "Top-Down")
    ]):
        ax = fig.add_subplot(2, 2, idx + 1, projection="3d")
        if isinstance(colors, np.ndarray) and colors.ndim == 2:
            ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], s=0.3, c=colors, alpha=0.7)
        else:
            ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], s=0.3, c=colors,
                       cmap="viridis", alpha=0.7)
        ax.view_init(elev=elev, azim=azim)
        ax.set_title(title, fontsize=14)
        ax.set_xlabel("X(m)"); ax.set_ylabel("Y(m)"); ax.set_zlabel("Z(m)")

    plt.suptitle("Composed Scene - njuptVolvo", fontsize=16)
    plt.tight_layout()
    plt.savefig(out_dir / "scene_composed.png", dpi=150)
    plt.close()
    log.info(f"预览图已保存: {out_dir / 'scene_composed.png'}")


def main():
    log.info("=" * 60)
    log.info("场景组装 v2: 全局尺度标定 + OBB 位姿放置")
    log.info("=" * 60)

    global_scale = compute_global_scale()

    meshes = collect_assets(global_scale)
    if not meshes:
        log.error("没有收集到任何 mesh，退出")
        sys.exit(1)

    log.info(f"共收集 {len(meshes)} 个 mesh，合并中...")
    scene_mesh = trimesh.util.concatenate(meshes)
    log.info(f"合并完成: {len(scene_mesh.vertices)} verts, {len(scene_mesh.faces)} faces")

    glb_path = OUT / "scene_composed.glb"
    scene_mesh.export(str(glb_path))
    log.info(f"场景 GLB: {glb_path}")

    render_preview(scene_mesh, OUT)
    log.info("完成!")


if __name__ == "__main__":
    main()
