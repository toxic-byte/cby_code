import torch
import os
import sys

path = "/e/cuiby/paper/cby_code/embeddings_cache/adj/adj_matrix_biological_process_full.pt"

if not os.path.isfile(path):
    print(f"文件不存在：{path}")
    sys.exit(1)

mat = torch.load(path, map_location="cpu")
if not mat.is_sparse:
    print("张量不是稀疏布局，意外：", mat.layout)
    sys.exit(1)

# 统一成coo便于索引
mat = mat.coalesce()  # 确保 indices 唯一、排序
indices = mat.indices()  # 形状 [2, nnz]
values = mat.values()    # 形状 [nnz]

rows = indices[0]        # 每个非零的行索引
rows_to_check = min(5, mat.size(0))

# 定义“等于1”的判定（浮点容差）
if values.dtype.is_floating_point:
    ones_mask = (values == 1) | (values.sub(1).abs() < 1e-8)
else:
    ones_mask = (values == 1)

rows_with_one = rows[ones_mask]

# 用 bincount 统计每行中“值为1”的非零个数
counts = torch.bincount(rows_with_one, minlength=rows_to_check)

print(f"矩阵形状：{tuple(mat.shape)}")
for i in range(rows_to_check):
    print(f"第 {i} 行中 1 的数量：{int(counts[i].item())}")