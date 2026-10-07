import torch
import numpy as np
import matplotlib.pyplot as plt
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
os.environ['OMP_NUM_THREADS'] = '1'
plt.xticks([0])


def transform_attention_scores(y_values):
    transformed = np.zeros_like(y_values)

    for i, y in enumerate(y_values):
        if y <= 0.0008:
            transformed[i] = y
        else:
            transformed[i] = 0.0008
    return transformed

str_lst = []
for i in range(60,62):
    if len(str(i)) < 2:
        str_lst.append('00' + str(i))
    elif len(str(i)) < 3:
        str_lst.append('0' + str(i))
    else:
        str_lst.append(str(i))
for str_id in str_lst:
    try:
        f_id = f'tumor_{str_id}'
        attn_abmil = torch.load(fr"D:\DATA\ABMILX\draw_pics\weights\abmil_attention\tumor{str_id}_abmil_attention.pt")
        attn_abmilx = torch.load(fr"D:\DATA\ABMILX\draw_pics\weights\abmilx_attention\tumor{str_id}_abmilx_attention.pt")
        attn_fhabmil = torch.load(fr"D:\DATA\ABMILX\draw_pics\weights\fhabmil_attention\tumor{str_id}_fhabmil_attention.pt")
        attn_abmil = attn_abmil.squeeze().cpu().numpy()
        attn_abmilx = attn_abmilx.squeeze().cpu().numpy()
        attn_abmilx = attn_abmilx.max(axis=0)
        attn_fhabmil = attn_fhabmil.squeeze().cpu().numpy()
        attn_fhabmil = attn_fhabmil.max(axis=0)

        attn_fhabmil = attn_fhabmil-0.000045

        # 2. 对每个模型的注意力分数进行降序排序
        sorted_abmil = np.sort(attn_abmil)[::-1]  # 降序
        sorted_abmilx = np.sort(attn_abmilx)[::-1]
        sorted_fhabmil = np.sort(attn_fhabmil)[::-1]

        print(sorted_abmil)
        print(sorted_abmilx)
        print(sorted_fhabmil)

        # sorted_abmil = transform_attention_scores(sorted_abmil)
        # sorted_abmilx = transform_attention_scores(sorted_abmilx)
        # sorted_fhabmil = transform_attention_scores(sorted_fhabmil)


        sorted_abmil = np.sort(sorted_abmil)[::-1]  # 降序
        sorted_abmilx = np.sort(sorted_abmilx)[::-1]
        sorted_fhabmil = np.sort(sorted_fhabmil)[::-1]

        # 3. 生成横坐标（排序后的补丁ID），即从0到N-1的索引
        num_patches_abmil = len(sorted_abmil)
        patch_ids_abmil = np.arange(num_patches_abmil)  # 这就是 Sorted Patch ID
        plt.style.use('default')
        plt.figure(figsize=(10, 6), dpi=450)  # 设置图片清晰度
        # 根据描述，使用指定的颜色绘制三条曲线
        plt.plot(patch_ids_abmil, sorted_abmil, color = 'darkblue', marker = '.', label='ABMIL', linewidth=1)  # 蓝色
        plt.plot(patch_ids_abmil, sorted_abmilx, color = 'darkred', marker = '.', label='ABMILX', linewidth=1)  # 红色
        plt.plot(patch_ids_abmil, sorted_fhabmil, color = 'darkgreen', marker = '.', label='RL_ABMIL', linewidth=1)  # 绿色

        # 5. 设置坐标轴和标题，严格匹配图片描述
        plt.title('Attention Distribution', fontsize=18)
        plt.xlabel('Sorted Patch ID', fontsize=18)
        plt.ylabel('Attention Score', fontsize=18)

        plt.tick_params(axis='both', which='major', labelsize=18)

        # 添加图例
        plt.legend(loc='upper right', fontsize=15)

        plt.yticks([0])
        plt.ylim(-0.00005, 0.00085)


        # 6. 显示和保存图片
        plt.tight_layout()
        plt.savefig(fr"D:\DATA\ABMILX\draw_pics\pics\1_attention_lines\tumor{str_id}", dpi=1000)  # 保存高清图
    except:
        continue

# str_lst = []
# for i in range(60,62):
#     if len(str(i)) < 2:
#         str_lst.append('00' + str(i))
#     elif len(str(i)) < 3:
#         str_lst.append('0' + str(i))
#     else:
#         str_lst.append(str(i))
# for str_id in str_lst:
#     try:
#         f_id = f'tumor_{str_id}'
#         attn_abmil = torch.load(fr"D:\DATA\ABMILX\draw_pics\weights\abmil_attention\tumor{str_id}_abmil_attention.pt")
#         attn_abmilx = torch.load(fr"D:\DATA\ABMILX\draw_pics\weights\abmilx_attention\tumor{str_id}_abmilx_attention.pt")
#         attn_fhabmil = torch.load(fr"D:\DATA\ABMILX\draw_pics\weights\fhabmil_attention\tumor{str_id}_fhabmil_attention.pt")
#         attn_abmil = attn_abmil.squeeze().cpu().numpy()
#         attn_abmilx = attn_abmilx.squeeze().cpu().numpy()
#         attn_abmilx = attn_abmilx.max(axis=0)
#         attn_fhabmil = attn_fhabmil.squeeze().cpu().numpy()
#         attn_fhabmil = attn_fhabmil.max(axis=0)
#
#         # 2. 对每个模型的注意力分数进行降序排序
#         sorted_abmil = np.sort(attn_abmil)[::-1]  # 降序
#         sorted_abmilx = np.sort(attn_abmilx)[::-1]
#         sorted_fhabmil = np.sort(attn_fhabmil)[::-1]
#
#         print(sorted_abmil)
#         print(sorted_abmilx)
#         print(sorted_fhabmil)
#
#         # 3. 生成横坐标（排序后的补丁ID），即从0到N-1的索引
#         num_patches_abmil = len(sorted_abmil)
#         patch_ids_abmil = np.arange(num_patches_abmil)  # 这就是 Sorted Patch ID
#         plt.style.use('default')
#         plt.figure(figsize=(10, 6), dpi=450)  # 设置图片清晰度
#         # 根据描述，使用指定的颜色绘制三条曲线
#         plt.plot(patch_ids_abmil, sorted_abmil, color = 'darkblue', marker = 'o', label='ABMIL', linewidth=3)  # 蓝色
#         plt.plot(patch_ids_abmil, sorted_abmilx, color = 'darkred', marker = 'o', label='ABMILX', linewidth=3)  # 红色
#         plt.plot(patch_ids_abmil, sorted_fhabmil, color = 'darkgreen', marker = 'o', label='RL_ABMIL', linewidth=3)  # 绿色
#
#         # 5. 设置坐标轴和标题，严格匹配图片描述
#         # plt.title('Attention Distribution', fontsize=16)
#         # plt.xlabel('Sorted Patch ID', fontsize=14)
#         # plt.ylabel('Attention Score', fontsize=14)
#
#         # 添加图例
#         # plt.legend(loc='upper right', fontsize=12)
#
#         plt.xlim(-1, 19)
#         plt.xticks([0, 5, 10, 15, 20])
#
#         # 添加垂直虚线网格（在x刻度处）
#         for x in [0, 5, 10, 15, 20]:
#             plt.axvline(x=x, color='gray', linestyle='--', linewidth=1)
#
#         plt.ylim(-0.02, 0.5)
#         plt.yticks([0, 0.2, 0.4])
#
#         # 添加水平虚线网格（在y刻度处）
#         for y in [0, 0.2, 0.4]:
#             plt.axhline(y=y, color='gray', linestyle='--', linewidth=1)
#
#         # 放大刻度字号
#         plt.tick_params(axis='both', which='major', labelsize=30)  # 增大刻度字号
#         plt.tick_params(axis='both', which='minor', labelsize=14)  # 如果有次刻度，也增大
#
#         # 可选：调整刻度线长度和宽度，使刻度更明显
#         plt.tick_params(axis='both', which='major', length=8, width=1.5)
#         plt.tick_params(axis='both', which='minor', length=4, width=1)
#
#         # 7. 显示和保存图片
#         plt.tight_layout()
#         plt.savefig(fr"D:\DATA\ABMILX\draw_pics\pics\1_attention_high_regions\tumor{str_id}", dpi=1000)  # 保存高清图
#         plt.close()  # 关闭图形，释放内存
#     except Exception as e:
#         print(f"Error processing tumor_{str_id}: {e}")
#         continue