import os
import h5py
import numpy as np
import xml.etree.ElementTree as ET

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt

try:
    from scipy.ndimage import binary_dilation, gaussian_filter
except ImportError:
    raise ImportError(
        "This visualization style needs scipy for mask dilation/smoothing.\n"
        "Please install it first: pip install scipy"
    )

try:
    import openslide
except ImportError:
    raise ImportError(
        "Please install openslide-python first:\n"
        "pip install openslide-python\n"
        "If it still fails, install system openslide library."
    )


# ============================================================
# Basic utilities
# ============================================================

def load_coords(h5_path):
    with h5py.File(h5_path, "r") as f:
        print("H5 keys:", list(f.keys()))
        if "coords" not in f:
            raise KeyError("Cannot find 'coords' in h5 file.")
        coords = f["coords"][:]
        print("Coords shape:", coords.shape)
    return coords


def load_xml_polygons(xml_path):
    if xml_path is None or not os.path.exists(xml_path):
        return []

    tree = ET.parse(xml_path)
    root = tree.getroot()

    polygons = []

    # Common CAMELYON-style XML format.
    for coordinates_node in root.iter("Coordinates"):
        points = []
        for coord in coordinates_node.iter("Coordinate"):
            x = coord.attrib.get("X")
            y = coord.attrib.get("Y")
            if x is not None and y is not None:
                points.append((float(x), float(y)))
        if len(points) > 2:
            polygons.append(points)

    # Fallback: all Coordinate nodes as one polygon.
    if len(polygons) == 0:
        points = []
        for coord in root.iter("Coordinate"):
            x = coord.attrib.get("X")
            y = coord.attrib.get("Y")
            if x is not None and y is not None:
                points.append((float(x), float(y)))
        if len(points) > 2:
            polygons.append(points)

    return polygons


def draw_roi(ax, polygons, scale_x, scale_y, color="deepskyblue", linewidth=0.7):
    """Draw thin ROI contours."""
    for poly in polygons:
        xs = [p[0] * scale_x for p in poly]
        ys = [p[1] * scale_y for p in poly]

        if len(xs) > 0:
            xs.append(xs[0])
            ys.append(ys[0])

        ax.plot(
            xs,
            ys,
            color=color,
            linewidth=linewidth,
            solid_capstyle="round",
            solid_joinstyle="round",
        )


def select_top_percent_indices(indices, scores, top_percent=0.5):
    """
    Select top-percent patches by score, by count rather than cumulative score.

    top_percent=0.5 means keep the top 50% highest-scoring patches.
    The selected patches will all be shown with the same fluorescent color.
    """
    indices = np.asarray(indices).astype(int)
    scores = np.asarray(scores).astype(np.float32)

    if len(indices) != len(scores):
        raise ValueError(
            f"Length mismatch: indices={len(indices)}, scores={len(scores)}"
        )

    if len(indices) == 0:
        return indices, scores

    top_percent = float(top_percent)
    top_percent = max(0.0, min(1.0, top_percent))
    keep_num = max(1, int(round(len(scores) * top_percent)))

    order = np.argsort(scores)[::-1]
    keep = order[:keep_num]

    print(f"Selected top {top_percent * 100:.1f}% patches: {keep_num}/{len(scores)}")
    print("Selected score range:", float(scores[keep].min()), "to", float(scores[keep].max()))

    return indices[keep], scores[keep]


def normalize_scores_for_brightness(scores, gamma=0.7, use_log=True):
    """
    Normalize attention scores to brightness values in [0, 1].

    The hue remains the same everywhere. Only brightness/opacity changes:
    high attention -> brighter; low attention -> darker.

    use_log=True is useful because softmax attention is often extremely skewed.
    gamma < 1 makes middle/low values easier to see.
    """
    scores = np.asarray(scores).astype(np.float32)
    if scores.size == 0:
        return scores

    x = scores.copy()
    if use_log:
        x = np.log1p(x)

    min_v = float(np.min(x))
    max_v = float(np.max(x))
    if max_v - min_v < 1e-12:
        return np.ones_like(x, dtype=np.float32)

    x = (x - min_v) / (max_v - min_v)
    x = np.clip(x, 0.0, 1.0)

    gamma = float(gamma)
    if gamma > 0:
        x = np.power(x, gamma)

    return x.astype(np.float32)


def build_weighted_patch_masks(
    coords,
    selected_indices,
    selected_scores,
    scale_x,
    scale_y,
    thumb_w,
    thumb_h,
    patch_size=256,
    min_patch_px=8,
    brightness_gamma=0.7,
    use_log=True,
):
    """
    Build two masks on the thumbnail canvas:
    1) binary_mask: selected patch support, used for zoom bbox.
    2) score_mask: attention-weighted mask, used for brightness.
    """
    binary_mask = np.zeros((thumb_h, thumb_w), dtype=np.float32)
    score_mask = np.zeros((thumb_h, thumb_w), dtype=np.float32)

    patch_w = max(int(round(patch_size * scale_x)), int(min_patch_px))
    patch_h = max(int(round(patch_size * scale_y)), int(min_patch_px))

    coords = np.asarray(coords)
    selected_indices = np.asarray(selected_indices).astype(int)
    selected_scores = np.asarray(selected_scores).astype(np.float32)
    brightness = normalize_scores_for_brightness(
        selected_scores, gamma=brightness_gamma, use_log=use_log
    )

    for idx, bright in zip(selected_indices, brightness):
        if idx < 0 or idx >= len(coords):
            continue

        x, y = coords[idx]
        x1 = int(round(x * scale_x))
        y1 = int(round(y * scale_y))
        x2 = min(x1 + patch_w, thumb_w)
        y2 = min(y1 + patch_h, thumb_h)

        if x1 >= thumb_w or y1 >= thumb_h or x2 <= 0 or y2 <= 0:
            continue

        x1 = max(x1, 0)
        y1 = max(y1, 0)
        binary_mask[y1:y2, x1:x2] = 1.0
        score_mask[y1:y2, x1:x2] = np.maximum(score_mask[y1:y2, x1:x2], bright)

    return binary_mask, score_mask


def smooth_weighted_patch_mask(binary_mask, score_mask, dilate_iter=2, blur_sigma=6):
    """
    Smooth the weighted score mask to create a fluorescent region.
    Higher attention remains brighter after smoothing.
    """
    support = binary_mask > 0

    if dilate_iter > 0:
        support = binary_dilation(support, iterations=int(dilate_iter))

    support_float = support.astype(np.float32)
    score_float = score_mask.astype(np.float32)

    if blur_sigma is not None and blur_sigma > 0:
        support_blur = gaussian_filter(support_float, sigma=float(blur_sigma))
        score_blur = gaussian_filter(score_float, sigma=float(blur_sigma))
    else:
        support_blur = support_float
        score_blur = score_float

    score_blur = score_blur * (support_blur > 0).astype(np.float32)

    if score_blur.max() > 0:
        score_blur = score_blur / score_blur.max()

    return score_blur.astype(np.float32)


def smooth_support_mask(binary_mask, dilate_iter=2, blur_sigma=6):
    """
    Build a smooth support mask from the selected top-percent patches.
    This mask decides where selected patches are visible.

    Difference from weighted attention mask:
    - support mask: selected patch area, all selected patches should be visible.
    - weighted mask: attention brightness, high attention brighter and low attention darker.
    """
    support = np.asarray(binary_mask) > 0

    if dilate_iter > 0:
        support = binary_dilation(support, iterations=int(dilate_iter))

    support_float = support.astype(np.float32)

    if blur_sigma is not None and blur_sigma > 0:
        support_blur = gaussian_filter(support_float, sigma=float(blur_sigma))
    else:
        support_blur = support_float

    if support_blur.max() > 0:
        support_blur = support_blur / support_blur.max()

    return support_blur.astype(np.float32)


def overlay_fluorescent_mask(
    ax,
    weighted_mask,
    support_mask=None,
    color=(0.0, 1.0, 1.0),
    max_alpha=0.85,
    min_alpha=0.18,
    threshold=0.0,
    draw_contour=False,
    contour_width=0.8,
):
    """
    Overlay a single-color fluorescent mask.

    support_mask controls WHERE to display.
    weighted_mask controls BRIGHTNESS within the displayed selected regions.

    This guarantees that top_percent-selected patches are all displayed:
    - selected low-attention patches are darker, but still visible;
    - selected high-attention patches are brighter;
    - the hue stays the same everywhere.
    """
    weighted_mask = np.asarray(weighted_mask).astype(np.float32)

    if support_mask is None:
        support = (weighted_mask > threshold).astype(np.float32)
    else:
        support = np.asarray(support_mask).astype(np.float32)
        support = np.clip(support, 0.0, 1.0)

    visible = support > threshold

    h, w = weighted_mask.shape
    overlay = np.zeros((h, w, 4), dtype=np.float32)
    overlay[..., 0] = color[0]
    overlay[..., 1] = color[1]
    overlay[..., 2] = color[2]

    alpha_map = np.zeros_like(weighted_mask, dtype=np.float32)

    # Base visibility comes from support so every selected top-percent patch appears.
    # Brightness variation comes from weighted_mask.
    alpha_map[visible] = (
        min_alpha * support[visible]
        + weighted_mask[visible] * (float(max_alpha) - min_alpha)
    )
    alpha_map = np.clip(alpha_map, 0.0, 1.0)

    overlay[..., 3] = alpha_map
    ax.imshow(overlay)

    if draw_contour:
        ax.contour(
            visible.astype(np.float32),
            levels=[0.5],
            colors=[color],
            linewidths=contour_width,
        )

def darken_thumbnail(thumb, dark_factor=0.45):
    """Darken RGB thumbnail to make the fluorescent annotation more visible."""
    thumb_np = np.asarray(thumb).astype(np.float32) / 255.0
    thumb_np = np.clip(thumb_np * float(dark_factor), 0, 1)
    return thumb_np


def get_patch_bbox(coords, scale_x, scale_y, thumb_w, thumb_h, margin=150):
    coords = np.asarray(coords)
    xs = coords[:, 0] * scale_x
    ys = coords[:, 1] * scale_y

    x_min = max(int(xs.min()) - margin, 0)
    y_min = max(int(ys.min()) - margin, 0)
    x_max = min(int(xs.max()) + margin, thumb_w)
    y_max = min(int(ys.max()) + margin, thumb_h)

    return x_min, y_min, x_max, y_max


def get_mask_bbox(mask, thumb_w, thumb_h, margin=150):
    """Get bbox around fluorescent mask; fallback is full thumbnail."""
    ys, xs = np.where(mask > 0)
    if len(xs) == 0 or len(ys) == 0:
        return 0, 0, thumb_w, thumb_h

    x_min = max(int(xs.min()) - margin, 0)
    y_min = max(int(ys.min()) - margin, 0)
    x_max = min(int(xs.max()) + margin, thumb_w)
    y_max = min(int(ys.max()) + margin, thumb_h)
    return x_min, y_min, x_max, y_max


def set_zoom(ax, bbox):
    x_min, y_min, x_max, y_max = bbox
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_max, y_min)


# ============================================================
# Main visualization function
# ============================================================

def visualize_final_hard_attention_one_wsi(
    wsi_path,
    h5_path,
    vis_npz_path,
    xml_path=None,
    save_dir="final_hard_attention_visualizations",
    patch_size=256,
    thumbnail_size=6000,
    top_percent=0.5,
    make_zoom=True,
    fluorescent_color=(0.0, 1.0, 1.0),
    background_dark_factor=0.45,
    mask_alpha=0.85,
    min_alpha=0.18,
    mask_threshold=0.0,
    dilate_iter=2,
    blur_sigma=6,
    roi_linewidth=0.7,
    min_patch_px=8,
    brightness_gamma=0.7,
    use_log_brightness=True,
):
    os.makedirs(save_dir, exist_ok=True)

    case_name = os.path.basename(vis_npz_path).replace("_vis.npz", "")

    slide = openslide.OpenSlide(wsi_path)
    w0, h0 = slide.dimensions

    thumb = slide.get_thumbnail((thumbnail_size, thumbnail_size)).convert("RGB")
    wt, ht = thumb.size

    scale_x = wt / w0
    scale_y = ht / h0

    print(f"WSI size: {w0} x {h0}")
    print(f"Thumbnail size: {wt} x {ht}")
    print(f"Scale: {scale_x:.6f}, {scale_y:.6f}")

    coords = load_coords(h5_path)

    vis = np.load(vis_npz_path, allow_pickle=True)
    print("NPZ keys:", list(vis.keys()))

    if "final_hard_indices" not in vis:
        raise KeyError("Cannot find 'final_hard_indices' in npz.")
    if "final_hard_attn_scores" not in vis:
        raise KeyError("Cannot find 'final_hard_attn_scores' in npz.")

    final_hard_indices = vis["final_hard_indices"]
    final_hard_attn_scores = vis["final_hard_attn_scores"]

    print("final_hard_indices shape:", final_hard_indices.shape)
    print("final_hard_attn_scores shape:", final_hard_attn_scores.shape)

    selected_indices, selected_scores = select_top_percent_indices(
        final_hard_indices,
        final_hard_attn_scores,
        top_percent=top_percent,
    )

    raw_mask, score_mask = build_weighted_patch_masks(
        coords=coords,
        selected_indices=selected_indices,
        selected_scores=selected_scores,
        scale_x=scale_x,
        scale_y=scale_y,
        thumb_w=wt,
        thumb_h=ht,
        patch_size=patch_size,
        min_patch_px=min_patch_px,
        brightness_gamma=brightness_gamma,
        use_log=use_log_brightness,
    )

    fluor_mask = smooth_weighted_patch_mask(
        raw_mask,
        score_mask,
        dilate_iter=dilate_iter,
        blur_sigma=blur_sigma,
    )

    # Support mask guarantees all selected top-percent patches are displayed.
    # Fluor mask only controls brightness.
    support_mask = smooth_support_mask(
        raw_mask,
        dilate_iter=dilate_iter,
        blur_sigma=blur_sigma,
    )

    label = vis["label"] if "label" in vis else None
    pred_label = vis["pred_label"] if "pred_label" in vis else None
    pred_prob = vis["pred_prob_class1"] if "pred_prob_class1" in vis else None

    title_extra = ""
    try:
        if label is not None and pred_label is not None and pred_prob is not None:
            title_extra = f" | label={int(label)}, pred={int(pred_label)}, prob={float(pred_prob):.3f}"
    except Exception:
        title_extra = ""

    polygons = load_xml_polygons(xml_path) if xml_path else []
    print("Number of ROI polygons:", len(polygons))

    thumb_dark = darken_thumbnail(thumb, dark_factor=background_dark_factor)

    aspect = ht / max(wt, 1)
    fig_w = 10
    fig_h = max(10, min(28, fig_w * aspect))

    # =========================
    # Figure 1: fluorescent final hard attention, no boxes, same color
    # =========================
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.imshow(thumb_dark)

    overlay_fluorescent_mask(
        ax=ax,
        weighted_mask=fluor_mask,
        support_mask=support_mask,
        color=fluorescent_color,
        max_alpha=mask_alpha,
        min_alpha=min_alpha,
        threshold=mask_threshold,
        draw_contour=False,
    )

    draw_roi(
        ax,
        polygons,
        scale_x,
        scale_y,
        color="deepskyblue",
        linewidth=roi_linewidth,
    )

    ax.set_title(
        f"{case_name} Final Hard Attention Fluorescent Overlay{title_extra}\n"
        f"top {top_percent * 100:.0f}% patches, higher attention = brighter"
    )
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(
        os.path.join(save_dir, f"{case_name}_final_hard_attention_fluorescent.png"),
        dpi=300,
    )
    plt.close(fig)

    # =========================
    # Figure 2: original background + fluorescent highlight
    # =========================
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.imshow(thumb)

    overlay_fluorescent_mask(
        ax=ax,
        weighted_mask=fluor_mask,
        support_mask=support_mask,
        color=fluorescent_color,
        max_alpha=mask_alpha,
        min_alpha=min_alpha,
        threshold=mask_threshold,
        draw_contour=False,
    )

    draw_roi(
        ax,
        polygons,
        scale_x,
        scale_y,
        color="deepskyblue",
        linewidth=roi_linewidth,
    )

    ax.set_title(
        f"{case_name} Fluorescent Highlight on Original WSI{title_extra}"
    )
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(
        os.path.join(save_dir, f"{case_name}_final_hard_attention_fluorescent_original_bg.png"),
        dpi=300,
    )
    plt.close(fig)

    # =========================
    # Figure 3: zoomed fluorescent view
    # =========================
    if make_zoom:
        # Use the selected mask bbox for more focused zoom.
        bbox = get_mask_bbox(raw_mask, thumb_w=wt, thumb_h=ht, margin=180)

        fig, ax = plt.subplots(figsize=(10, 10))
        ax.imshow(thumb_dark)

        overlay_fluorescent_mask(
            ax=ax,
            weighted_mask=fluor_mask,
            support_mask=support_mask,
            color=fluorescent_color,
            max_alpha=mask_alpha,
            min_alpha=min_alpha,
            threshold=mask_threshold,
            draw_contour=False,
        )

        draw_roi(
            ax,
            polygons,
            scale_x,
            scale_y,
            color="deepskyblue",
            linewidth=roi_linewidth,
        )

        set_zoom(ax, bbox)
        ax.set_title(
            f"{case_name} Zoomed Fluorescent Final Hard Attention{title_extra}"
        )
        ax.axis("off")
        fig.tight_layout()
        fig.savefig(
            os.path.join(save_dir, f"{case_name}_zoom_final_hard_attention_fluorescent.png"),
            dpi=300,
        )
        plt.close(fig)

    print(f"Saved fluorescent final hard attention visualizations to: {save_dir}")

    # Return detailed information so batch mode can save it into CSV.
    selected_num = len(selected_indices)
    total_final_hard_num = len(final_hard_indices)

    if len(selected_scores) > 0:
        selected_score_min = float(np.min(selected_scores))
        selected_score_max = float(np.max(selected_scores))
        selected_score_mean = float(np.mean(selected_scores))
    else:
        selected_score_min = ""
        selected_score_max = ""
        selected_score_mean = ""

    result_info = {
        "case_id": case_name,

        "wsi_width": int(w0),
        "wsi_height": int(h0),

        "thumbnail_width": int(wt),
        "thumbnail_height": int(ht),

        "scale_x": float(scale_x),
        "scale_y": float(scale_y),

        "coords_num": int(coords.shape[0]),
        "coords_dim": int(coords.shape[1]) if len(coords.shape) > 1 else "",

        "final_hard_indices_shape": str(tuple(final_hard_indices.shape)),
        "final_hard_attn_scores_shape": str(tuple(final_hard_attn_scores.shape)),
        "final_hard_num": int(total_final_hard_num),

        "selected_top_percent": float(top_percent),
        "selected_patch_num": int(selected_num),
        "selected_score_min": selected_score_min,
        "selected_score_max": selected_score_max,
        "selected_score_mean": selected_score_mean,

        "roi_polygon_num": int(len(polygons)),

        "label": int(label) if label is not None else "",
        "pred_label": int(pred_label) if pred_label is not None else "",
        "pred_prob_class1": float(pred_prob) if pred_prob is not None else "",

        "save_dir": save_dir,
    }

    return result_info


# ============================================================
# Batch visualization helpers
# ============================================================

def find_existing_file(directory, case_id, suffixes):
    """Return the first existing file path for case_id with any suffix."""
    if directory is None:
        return None
    for suffix in suffixes:
        path = os.path.join(directory, case_id + suffix)
        if os.path.exists(path):
            return path
    return None


def infer_case_id_from_vis_file(filename):
    """Infer case id from a visualization npz filename."""
    base = os.path.basename(filename)
    if base.endswith("_vis.npz"):
        return base[:-len("_vis.npz")]
    if base.endswith(".npz"):
        return base[:-len(".npz")]
    return os.path.splitext(base)[0]


def visualize_batch_from_dirs(
    wsi_dir,
    h5_dir,
    vis_npz_dir,
    xml_dir=None,
    save_dir="final_hard_attention_visualizations_batch",
    patch_size=256,
    thumbnail_size=6000,
    top_percent=0.5,
    make_zoom=True,
    fluorescent_color=(0.0, 1.0, 1.0),
    background_dark_factor=0.45,
    mask_alpha=0.85,
    min_alpha=0.18,
    mask_threshold=0.0,
    dilate_iter=2,
    blur_sigma=6,
    roi_linewidth=0.7,
    min_patch_px=8,
    brightness_gamma=0.7,
    use_log_brightness=True,
    max_cases=None,
):
    """
    Batch-generate fluorescent visualizations for all *_vis.npz files in vis_npz_dir.

    Expected filename matching:
        vis_npz_dir/case_id_vis.npz
        wsi_dir/case_id.tif or case_id.tiff
        h5_dir/case_id.h5
        xml_dir/case_id.xml  (optional; skipped if missing)

    The function creates one subfolder per case under save_dir.
    """
    os.makedirs(save_dir, exist_ok=True)

    vis_files = [
        f for f in os.listdir(vis_npz_dir)
        if f.endswith("_vis.npz") or f.endswith(".npz")
    ]
    vis_files = sorted(vis_files)

    if max_cases is not None:
        vis_files = vis_files[:int(max_cases)]

    print("=" * 80)
    print(f"Found {len(vis_files)} npz files in: {vis_npz_dir}")
    print("Batch save dir:", save_dir)
    print("=" * 80)

    summary_rows = []

    for i, vis_file in enumerate(vis_files, start=1):
        case_id = infer_case_id_from_vis_file(vis_file)
        vis_npz_path = os.path.join(vis_npz_dir, vis_file)

        wsi_path = find_existing_file(
            wsi_dir,
            case_id,
            [".tif", ".tiff", ".svs", ".ndpi"]
        )
        h5_path = find_existing_file(h5_dir, case_id, [".h5"])
        xml_path = find_existing_file(xml_dir, case_id, [".xml"]) if xml_dir else None

        print("\n" + "-" * 80)
        print(f"[{i}/{len(vis_files)}] Case: {case_id}")
        print("VIS:", vis_npz_path)
        print("WSI:", wsi_path)
        print("H5 :", h5_path)
        print("XML:", xml_path)

        status = "ok"
        message = ""
        result_info = {}

        if wsi_path is None:
            status = "missing_wsi"
            message = f"Cannot find WSI for {case_id}"
            print("SKIP:", message)
        elif h5_path is None:
            status = "missing_h5"
            message = f"Cannot find H5 for {case_id}"
            print("SKIP:", message)
        else:
            case_save_dir = os.path.join(save_dir, case_id)
            try:
                result_info = visualize_final_hard_attention_one_wsi(
                    wsi_path=wsi_path,
                    h5_path=h5_path,
                    vis_npz_path=vis_npz_path,
                    xml_path=xml_path,
                    save_dir=case_save_dir,
                    patch_size=patch_size,
                    thumbnail_size=thumbnail_size,
                    top_percent=top_percent,
                    make_zoom=make_zoom,
                    fluorescent_color=fluorescent_color,
                    background_dark_factor=background_dark_factor,
                    mask_alpha=mask_alpha,
                    min_alpha=min_alpha,
                    mask_threshold=mask_threshold,
                    dilate_iter=dilate_iter,
                    blur_sigma=blur_sigma,
                    roi_linewidth=roi_linewidth,
                    min_patch_px=min_patch_px,
                    brightness_gamma=brightness_gamma,
                    use_log_brightness=use_log_brightness,
                )
                message = case_save_dir
            except Exception as e:
                status = "error"
                message = str(e)
                print("ERROR:", message)

        row = {
            "case_id": case_id,
            "status": status,
            "message": message,
            "vis_npz_path": vis_npz_path,
            "wsi_path": wsi_path if wsi_path is not None else "",
            "h5_path": h5_path if h5_path is not None else "",
            "xml_path": xml_path if xml_path is not None else "",
        }

        # If visualization succeeded, add the detailed statistics returned by
        # visualize_final_hard_attention_one_wsi().
        if status == "ok":
            row.update(result_info)
        else:
            # Keep CSV columns complete even for skipped/error cases.
            row.update({
                "wsi_width": "",
                "wsi_height": "",
                "thumbnail_width": "",
                "thumbnail_height": "",
                "scale_x": "",
                "scale_y": "",
                "coords_num": "",
                "coords_dim": "",
                "final_hard_indices_shape": "",
                "final_hard_attn_scores_shape": "",
                "final_hard_num": "",
                "selected_top_percent": "",
                "selected_patch_num": "",
                "selected_score_min": "",
                "selected_score_max": "",
                "selected_score_mean": "",
                "roi_polygon_num": "",
                "label": "",
                "pred_label": "",
                "pred_prob_class1": "",
                "save_dir": "",
            })

        summary_rows.append(row)

    summary_path = os.path.join(save_dir, "batch_visualization_summary.csv")
    try:
        import pandas as pd
        pd.DataFrame(summary_rows).to_csv(summary_path, index=False, encoding="utf-8-sig")
    except Exception:
        # pandas may be unavailable; fallback to csv module.
        import csv
        with open(summary_path, "w", newline="", encoding="utf-8-sig") as f:
            # Collect all keys so newly added statistic columns are also saved.
            fieldnames = []
            for row in summary_rows:
                for key in row.keys():
                    if key not in fieldnames:
                        fieldnames.append(key)

            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(summary_rows)

    print("\n" + "=" * 80)
    print("Batch visualization finished.")
    print("Summary saved to:", summary_path)
    print("=" * 80)



if __name__ == "__main__":
    # ============================================================
    # Batch mode: set your directories here.
    # ============================================================
    # Directory containing all WSI files, such as tumor_001.tif / tumor_001.tiff.
    wsi_dir = r"H:\CAMELYON16\Raw_Data"

    # Directory containing all XML annotation files, such as tumor_001.xml.
    # If some cases do not have XML, they will still be processed without ROI contour.
    xml_dir = r"H:\CAMELYON16\training\lesion_annotations"

    # Directory containing all H5 coordinate files, such as tumor_001.h5.
    h5_dir = r"H:\CAMELYON16\Preprocessing\CLAM_20x_TransMIL\patches"

    # Directory containing all visualization npz files, such as tumor_001_vis.npz.
    vis_npz_dir = r"D:\WJ\论文\方法一\可视化\注意力热图\数据\009110_SAP"
    # Output directory. A separate subfolder will be created for each case.
    save_dir = r"D:\WJ\论文\方法一\可视化\注意力热图\PLIP_SAP_hard_009110_0.85"

    visualize_batch_from_dirs(
        wsi_dir=wsi_dir,
        h5_dir=h5_dir,
        vis_npz_dir=vis_npz_dir,
        xml_dir=xml_dir,
        save_dir=save_dir,
        patch_size=256,
        thumbnail_size=6000,

        # Keep top 60% highest final-hard-attention patches.
        # The score is used for selection and brightness; color hue stays the same.
        top_percent=0.85,

        # Single-color fluorescent style. Red = (1, 0, 0); cyan = (0, 1, 1).
        fluorescent_color=(1.0, 0.0, 0.0),
        background_dark_factor=0.45,
        mask_alpha=0.85,
        min_alpha=0.18,
        mask_threshold=0.0,
        dilate_iter=2,
        blur_sigma=6,
        roi_linewidth=0.7,
        min_patch_px=8,
        brightness_gamma=0.7,
        use_log_brightness=True,
        make_zoom=True,

        # Set this to a small number for testing, e.g. max_cases=3.
        # Use None to process all cases.
        max_cases=None,
    )