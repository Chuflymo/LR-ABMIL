import torch
import numpy as np
import h5py
from xml.dom.minidom import parse
import xml.dom.minidom
from openTSNE import TSNE
from shapely.geometry import Polygon
from copy import deepcopy
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import os
import umap

# def read_annotation(anno_file, return_type=False):
#     anno_tumor = []
#     anno_normal = []
#     anno_type = set()
#     DOMTree = xml.dom.minidom.parse(anno_file)
#     annotations = DOMTree.documentElement.getElementsByTagName('Annotations')[0].getElementsByTagName('Annotation')
#     for i in range(len(annotations)):
#         anno_type.add(annotations[i].getAttribute('PartOfGroup'))
#         if annotations[i].getAttribute('PartOfGroup') == 'Exclusion':
#             coordinates = annotations[i].getElementsByTagName('Coordinates')
#             _tmp = []
#             for node in coordinates[0].childNodes:
#                 if type(node) == xml.dom.minidom.Element:
#                     _tmp.append([int(float(node.getAttribute("X"))), int(float(node.getAttribute("Y")))])
#
#             anno_normal.append(_tmp)
#         elif annotations[i].getAttribute('PartOfGroup') != 'None':
#             coordinates = annotations[i].getElementsByTagName('Coordinates')
#             _tmp = []
#             for node in coordinates[0].childNodes:
#                 if type(node) == xml.dom.minidom.Element:
#                     _tmp.append([int(float(node.getAttribute("X"))), int(float(node.getAttribute("Y")))])
#
#             anno_tumor.append(_tmp)
#     if return_type:
#         return anno_tumor, anno_normal, anno_type
#     else:
#         return anno_tumor, anno_normal
#
# def get_label(coords, anno_file, _l=None):
#     if anno_file is None:
#         return None
#
#     annos_tumor, annos_normal = read_annotation(anno_file)
#     annos_tumor_polygon = [Polygon(_anno) for _anno in annos_tumor]
#     annos_normal_polygon = [Polygon(_anno) for _anno in annos_normal]
#
#     annos_tumor_in_normal_idx = [
#         idx for idx, _anno in enumerate(annos_tumor_polygon)
#         if any(_anno.covered_by(_anno_1) for _anno_1 in annos_normal_polygon)
#     ]
#
#     label = np.zeros(len(coords), dtype=int)
#
#     tumor_polygons = {poly: True for idx, poly in enumerate(annos_tumor_polygon) if idx not in annos_tumor_in_normal_idx}
#     exclusion_polygons = {poly: True for idx, poly in enumerate(annos_tumor_polygon) if idx in annos_tumor_in_normal_idx}
#
#     for i, coord in enumerate(coords):
#         patch = Polygon([
#             [coord[0], coord[1]],
#             [coord[0] + 512, coord[1]],
#             [coord[0] + 512, coord[1] + 512],
#             [coord[0], coord[1] + 512]
#         ])
#
#         intersects_tumor = any(patch.intersects(poly) for poly in tumor_polygons)
#         intersects_exclusion = any(patch.intersects(poly) for poly in exclusion_polygons)
#
#         if intersects_tumor and not intersects_exclusion:
#             if _l is not None:
#                 label[i] = 0
#             else:
#                 label[i] = 1
#         else:
#             label[i] = 0
#
#     if _l is not None:
#         label[_l > 0] = 1
#
#     return label
#
# def plot(x, y, ax=None, title=None, draw_legend=True, draw_centers=False, draw_cluster_labels=False, colors=None, legend_kwargs=None, label_order=None, **kwargs):
#     import matplotlib
#
#     if ax is None:
#         _, ax = plt.subplots(figsize=(8, 8))
#
#     if title is not None:
#         ax.set_title(title)
#
#     plot_params = {"alpha": kwargs.get("alpha", 0.8)}
#
#     # Create main plot
#     if label_order is not None:
#         assert all(np.isin(np.unique(y), label_order))
#         classes = [l for l in label_order if l in np.unique(y)]
#     else:
#         classes = np.unique(y)
#     if colors is None:
#         default_colors = matplotlib.rcParams["axes.prop_cycle"]
#         colors = {k: v["color"] for k, v in zip(classes, default_colors())}
#
#     point_colors = list(map(colors.get, y))
#
#     size = deepcopy(y)
#     point_size = deepcopy(y)
#     point_size[size != 1] = 1
#     point_size[size == 1] = 50
#     point_size[size == 2] = 1
#
#     fig = ax.scatter(x[:, 0], x[:, 1], c=point_colors, rasterized=True, s=point_size, **plot_params)
#
#     # Plot mediods
#     if draw_centers:
#         centers = []
#         for yi in classes:
#             mask = yi == y
#             centers.append(np.median(x[mask, :2], axis=0))
#         centers = np.array(centers)
#
#         center_colors = list(map(colors.get, classes))
#         ax.scatter(
#             centers[:, 0], centers[:, 1], c=center_colors, s=48, alpha=1, edgecolor="k"
#         )
#
#         # Draw mediod labels
#         if draw_cluster_labels:
#             for idx, label in enumerate(classes):
#                 ax.text(
#                     centers[idx, 0],
#                     centers[idx, 1] + 2.2,
#                     label + ': ' + str(len(x)),
#                     fontsize=kwargs.get("fontsize", 6),
#                     horizontalalignment="center",
#                 )
#
#     # Hide ticks and axis
#     ax.set_xticks([]), ax.set_yticks([]), ax.axis("off")
#
#     if draw_legend:
#         legend_handles = [
#             mlines.Line2D(
#                 [],
#                 [],
#                 marker="s",
#                 color="w",
#                 markerfacecolor=colors[yi],
#                 ms=10,
#                 alpha=1,
#                 linewidth=0,
#                 label=str(yi) + ': ' + str(len(y[y == yi])),
#                 markeredgecolor="k",
#             )
#             for yi in classes
#         ]
#         legend_kwargs_ = dict(loc="center left", bbox_to_anchor=(1, 0.5), frameon=False, )
#         if legend_kwargs is not None:
#             legend_kwargs_.update(legend_kwargs)
#         ax.legend(handles=legend_handles, **legend_kwargs_)
#
#     return fig
#
# def tsne(feat, coords=None, anno_file=None, _l=None, **kwargs):
#     try:
#         label = get_label(coords, anno_file, _l)
#     except Exception as e:
#         print(f"Error in get_label: {e}")
#         label = None
#
#     feat = feat.cpu().numpy()
#     try:
#         embedding = TSNE(n_jobs=1).fit(feat)
#     except Exception as e:
#         print(f"Error in t-SNE: {e}")
#         return None
#
#     y = label if label is not None else np.array([1 for _ in range(len(embedding))])
#     return plot(embedding, y, **kwargs)
#
# directory_1 = r"D:\DATA\ABMILX\draw_pics\weights\fhabmil_feat"
# for root, dirs, files in os.walk(directory_1):
#     for file in files:
#         try:
#             if file.endswith('.pt'):
#                 file_path = os.path.join(root, file)
#                 name = file.split('.')[-2].split('_')[0].split('0')[:][0]+'_'+file.split('.')[-2].split('_')[0].split('r')[:][1]
#                 str_id = file.split('.')[-2].split('_')[0].split('0')[:][0]+file.split('.')[-2].split('_')[0].split('r')[:][1]
#                 feat = torch.load(file_path)
#                 feat = feat.squeeze(0)
#                 data_1 = torch.load(fr"D:\DATA\ABMILX\draw_pics\weights\fhabmil_attention\{str_id}_fhabmil_attention.pt")
#                 data_1 = data_1.squeeze()
#                 data_1 = data_1.mean(axis=0)
#                 features = torch.einsum('ns,n->ns', feat, data_1)
#                 patch = h5py.File(rf"D:\DATA\PLIP_MIL\draw_pictures\h5\{name}.h5", "r")
#                 coords = patch['coords'][:]
#                 plip_tsne = tsne(features, coords, rf"D:\DATA\PLIP_MIL\draw_pictures\lesion_annotations\{name}.xml", draw_legend=False)
#                 if plip_tsne is not None:
#                     plip_tsne.get_figure().savefig(rf'D:\DATA\ABMILX\draw_pics\pics\tsne_lowdim\fhabmil_tsne\{name}.png', dpi=600, bbox_inches='tight')
#                     print(f"Saved {name}")
#         except:
#             continue

def read_annotation(anno_file, return_type=False):
    anno_tumor = []
    anno_normal = []
    anno_type = set()
    DOMTree = xml.dom.minidom.parse(anno_file)
    annotations = DOMTree.documentElement.getElementsByTagName('Annotations')[0].getElementsByTagName('Annotation')
    for i in range(len(annotations)):
        anno_type.add(annotations[i].getAttribute('PartOfGroup'))
        if annotations[i].getAttribute('PartOfGroup') == 'Exclusion':
            coordinates = annotations[i].getElementsByTagName('Coordinates')
            _tmp = []
            for node in coordinates[0].childNodes:
                if type(node) == xml.dom.minidom.Element:
                    _tmp.append([int(float(node.getAttribute("X"))), int(float(node.getAttribute("Y")))])

            anno_normal.append(_tmp)
        elif annotations[i].getAttribute('PartOfGroup') != 'None':
            coordinates = annotations[i].getElementsByTagName('Coordinates')
            _tmp = []
            for node in coordinates[0].childNodes:
                if type(node) == xml.dom.minidom.Element:
                    _tmp.append([int(float(node.getAttribute("X"))), int(float(node.getAttribute("Y")))])

            anno_tumor.append(_tmp)
    if return_type:
        return anno_tumor, anno_normal, anno_type
    else:
        return anno_tumor, anno_normal

def get_label(coords, anno_file, _l=None):
    if anno_file is None:
        return None

    annos_tumor, annos_normal = read_annotation(anno_file)
    annos_tumor_polygon = [Polygon(_anno) for _anno in annos_tumor]
    annos_normal_polygon = [Polygon(_anno) for _anno in annos_normal]

    annos_tumor_in_normal_idx = [
        idx for idx, _anno in enumerate(annos_tumor_polygon)
        if any(_anno.covered_by(_anno_1) for _anno_1 in annos_normal_polygon)
    ]

    label = np.zeros(len(coords), dtype=int)

    tumor_polygons = {poly: True for idx, poly in enumerate(annos_tumor_polygon) if idx not in annos_tumor_in_normal_idx}
    exclusion_polygons = {poly: True for idx, poly in enumerate(annos_tumor_polygon) if idx in annos_tumor_in_normal_idx}

    for i, coord in enumerate(coords):
        patch = Polygon([
            [coord[0], coord[1]],
            [coord[0] + 512, coord[1]],
            [coord[0] + 512, coord[1] + 512],
            [coord[0], coord[1] + 512]
        ])

        intersects_tumor = any(patch.intersects(poly) for poly in tumor_polygons)
        intersects_exclusion = any(patch.intersects(poly) for poly in exclusion_polygons)

        if intersects_tumor and not intersects_exclusion:
            if _l is not None:
                label[i] = 0
            else:
                label[i] = 1
        else:
            label[i] = 0

    if _l is not None:
        label[_l > 0] = 1

    return label

def plot(x, y, ax=None, title=None, draw_legend=True, draw_centers=False, draw_cluster_labels=False, colors=None, legend_kwargs=None, label_order=None, **kwargs):
    import matplotlib

    if ax is None:
        _, ax = plt.subplots(figsize=(8, 8))

    if title is not None:
        ax.set_title(title)

    plot_params = {"alpha": kwargs.get("alpha", 0.8)}

    # Create main plot
    if label_order is not None:
        assert all(np.isin(np.unique(y), label_order))
        classes = [l for l in label_order if l in np.unique(y)]
    else:
        classes = np.unique(y)
    if colors is None:
        default_colors = matplotlib.rcParams["axes.prop_cycle"]
        colors = {k: v["color"] for k, v in zip(classes, default_colors())}

    point_colors = list(map(colors.get, y))

    size = deepcopy(y)
    point_size = deepcopy(y)
    point_size[size != 1] = 1
    point_size[size == 1] = 1
    point_size[size == 2] = 1

    fig = ax.scatter(x[:, 0], x[:, 1], c=point_colors, rasterized=True, s=point_size, **plot_params)

    # Plot mediods
    if draw_centers:
        centers = []
        for yi in classes:
            mask = yi == y
            centers.append(np.median(x[mask, :2], axis=0))
        centers = np.array(centers)

        center_colors = list(map(colors.get, classes))
        ax.scatter(
            centers[:, 0], centers[:, 1], c=center_colors, s=48, alpha=1, edgecolor="k"
        )

        # Draw mediod labels
        if draw_cluster_labels:
            for idx, label in enumerate(classes):
                ax.text(
                    centers[idx, 0],
                    centers[idx, 1] + 2.2,
                    label + ': ' + str(len(x)),
                    fontsize=kwargs.get("fontsize", 6),
                    horizontalalignment="center",
                )

    # Hide ticks and axis
    ax.set_xticks([]), ax.set_yticks([]), ax.axis("off")

    if draw_legend:
        legend_handles = [
            mlines.Line2D(
                [],
                [],
                marker="s",
                color="w",
                markerfacecolor=colors[yi],
                ms=10,
                alpha=1,
                linewidth=0,
                label=str(yi) + ': ' + str(len(y[y == yi])),
                markeredgecolor="k",
            )
            for yi in classes
        ]
        legend_kwargs_ = dict(loc="center left", bbox_to_anchor=(1, 0.5), frameon=False, )
        if legend_kwargs is not None:
            legend_kwargs_.update(legend_kwargs)
        ax.legend(handles=legend_handles, **legend_kwargs_)

    return fig

def umap_embedding(feat, coords=None, anno_file=None, _l=None, n_nbs = 100, **kwargs):

    """使用UMAP代替t-SNE进行降维"""
    n_nbs = n_nbs
    try:
        label = get_label(coords, anno_file, _l)
    except Exception as e:
        print(f"Error in get_label: {e}")
        label = None

    feat = feat.cpu().numpy()
    try:
        # 使用UMAP进行降维
        reducer = umap.UMAP(
            n_neighbors=100,  # 邻居数量，控制局部与全局结构的平衡
            min_dist=0,    # 点之间的最小距离，控制点的聚集程度
            n_components=2,  # 降维到2维
            random_state=42,  # 设置随机种子保证可重复性
            metric='cosine'  # 距离度量 #cosine
        )
        embedding = reducer.fit_transform(feat)
    except Exception as e:
        print(f"Error in UMAP: {e}")
        return None

    y = label if label is not None else np.array([1 for _ in range(len(embedding))])
    return plot(embedding, y, **kwargs)

# 如果需要保持函数名不变，可以重命名函数
def tsne(feat, coords=None, anno_file=None, _l=None, **kwargs):
    """为了向后兼容，保留tsne函数名但实际使用UMAP"""
    return umap_embedding(feat, coords, anno_file, _l, **kwargs)

# directory_1 = rf"E:\DATA\ABMILX\draw_pics\weights\abmil_feat"
# for root, dirs, files in os.walk(directory_1):
#     for file in files:
#         try:
#             if file.endswith('.pt'):
#                 file_path = os.path.join(root, file)
#                 name = 'tumor'+'_'+file.split('.')[-2].split('_')[0].split('r')[:][1]
#                 str_id = 'tumor'+file.split('.')[-2].split('_')[0].split('r')[:][1]
#                 feat = torch.load(file_path)
#                 feat = feat.squeeze(0)
#                 # data_1 = torch.load(fr"D:\DATA\ABMILX\draw_pics\weights\abmil_attention\{str_id}_abmil_attention.pt")
#                 # data_1 = data_1.squeeze()
#                 # data_1 = data_1.mean(axis=0)
#                 # features = torch.einsum('ns,n->ns', feat, data_1)
#                 patch = h5py.File(rf"E:\DATA\PLIP_MIL\draw_pictures\h5\{name}.h5", "r")
#                 coords = patch['coords'][:]
#                 plip_umap = umap_embedding(feat, coords, rf"E:\DATA\PLIP_MIL\draw_pictures\lesion_annotations\{name}.xml", draw_legend=False)
#                 if plip_umap is not None:
#                     plip_umap.get_figure().savefig(rf'E:\DATA\ABMILX\draw_pics\pics\lowdim\temp\abmil_umap\{name}.png', dpi=200, bbox_inches='tight')
#                     print(f"Saved {name}")
#         except:
#             continue

# directory_2 = rf"E:\DATA\ABMILX\draw_pics\weights\fhabmil_feat"
# lst_nbs = [90, 100]

# for nbs in lst_nbs:
#     # 根据nbs值创建对应的输出文件夹
#     output_dir = rf'E:\DATA\ABMILX\draw_pics\pics\lowdim\temp\fhabmil_0.3_umap_n{nbs}'
#     os.makedirs(output_dir, exist_ok=True)  # 如果文件夹不存在则创建
    
#     print(f"\n{'='*50}")
#     print(f"Processing with n_neighbors = {nbs}")
#     print(f"Output directory: {output_dir}")
#     print(f"{'='*50}\n")
    
#     for root, dirs, files in os.walk(directory_2):
#         for file in files:
#             try:
#                 if file.endswith('.pt'):
#                     file_path = os.path.join(root, file)
#                     name = 'tumor' + '_' + file.split('.')[-2].split('_')[0].split('r')[:][1]
#                     str_id = 'tumor' + file.split('.')[-2].split('_')[0].split('r')[:][1]
                    
#                     feat = torch.load(file_path)
#                     feat = feat.squeeze(0)
#                     n, m = feat.shape
#                     feat = feat.reshape(n, 8, m//8)
#                     data_1 = torch.load(fr"E:\DATA\ABMILX\draw_pics\weights\fhabmil_attention\{str_id}_fhabmil_attention.pt")
#                     data_1 = data_1.squeeze()
#                     data_1 = data_1.T
#                     features = torch.einsum('nbs,nb->nbs', feat, data_1)
#                     features = features.reshape(n, m)
                    
#                     patch = h5py.File(rf"E:\DATA\PLIP_MIL\draw_pictures\h5\{name}.h5", "r")
#                     coords = patch['coords'][:]
                    
#                     # 使用当前的nbs值进行UMAP降维
#                     plip_umap = umap_embedding(features, coords, 
#                                                rf"E:\DATA\PLIP_MIL\draw_pictures\lesion_annotations\{name}.xml", 
#                                                draw_legend=False, 
#                                                n_nbs=nbs)  # 注意参数名应该是n_neighbors
                    
#                     if plip_umap is not None:
#                         # 保存到对应的nbs文件夹中
#                         save_path = os.path.join(output_dir, f'{name}.png')
#                         plip_umap.get_figure().savefig(save_path, dpi=200, bbox_inches='tight')
#                         print(f"Saved {name} (n_neighbors={nbs})")
                        
#             except Exception as e:
#                 print(f"Error processing {file} with n_neighbors={nbs}: {e}")
#                 continue
#
directory_3 = rf"E:\DATA\ABMILX\draw_pics\weights\abmilx_feat"
for root, dirs, files in os.walk(directory_3):
    for file in files:
        try:
            if file.endswith('.pt'):
                file_path = os.path.join(root, file)
                name = 'tumor'+'_'+file.split('.')[-2].split('_')[0].split('r')[:][1]
                str_id = 'tumor'+file.split('.')[-2].split('_')[0].split('r')[:][1]
                feat = torch.load(file_path)
                feat = feat.squeeze(0)
                n, m = feat.shape
                feat = feat.reshape(n, 8, m//8)
                data_1 = torch.load(fr"E:\DATA\ABMILX\draw_pics\weights\abmilx_attention\{str_id}_abmilx_attention.pt")
                data_1 = data_1.squeeze()
                data_1 = data_1.T
                # data_1 = data_1.max(axis=0)[0]
                features = torch.einsum('nbs,nb->nbs', feat, data_1)
                features = features.reshape(n, m)
                patch = h5py.File(rf"E:\DATA\PLIP_MIL\draw_pictures\h5\{name}.h5", "r")
                coords = patch['coords'][:]
                plip_umap = umap_embedding(features, coords, rf"E:\DATA\PLIP_MIL\draw_pictures\lesion_annotations\{name}.xml", draw_legend=False)
                if plip_umap is not None:
                    plip_umap.get_figure().savefig(rf'E:\DATA\ABMILX\draw_pics\pics\lowdim\temp\abmilx_umap\{name}.png', dpi=200, bbox_inches='tight')
                    print(f"Saved {name}")
        except:
            continue

# directory_3 = r"D:\DATA\ABMILX\draw_pics\weights\fhabmil_feat"
# for root, dirs, files in os.walk(directory_3):
#     for file in files:
#         try:
#             if file.endswith('.pt'):
#                 file_path = os.path.join(root, file)
#                 name = file.split('.')[-2].split('_')[0].split('0')[:][0]+'_'+file.split('.')[-2].split('_')[0].split('r')[:][1]
#                 str_id = file.split('.')[-2].split('_')[0].split('0')[:][0]+file.split('.')[-2].split('_')[0].split('r')[:][1]
#                 feat = torch.load(file_path)
#                 feat = feat.squeeze(0)
#                 data_1 = torch.load(fr"D:\DATA\ABMILX\draw_pics\weights\fhabmil_attention\{str_id}_fhabmil_attention.pt")
#                 data_1 = data_1.squeeze()
#                 data_1 = data_1.max(axis=0)[0]
#                 features = torch.einsum('ns,n->ns', feat, data_1)
#                 patch = h5py.File(rf"D:\DATA\PLIP_MIL\draw_pictures\h5\{name}.h5", "r")
#                 coords = patch['coords'][:]
#                 plip_umap = umap_embedding(feat, coords, rf"D:\DATA\PLIP_MIL\draw_pictures\lesion_annotations\{name}.xml", draw_legend=False)
#                 if plip_umap is not None:
#                     plip_umap.get_figure().savefig(rf'D:\DATA\ABMILX\draw_pics\pics\lowdim\temp\fhabmil_umap\{name}.png', dpi=600, bbox_inches='tight')
#                     print(f"Saved {name}")
#         except:
#             continue