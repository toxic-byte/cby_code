import pickle
import pprint

def load_and_inspect_pkl(file_path):
    """
    加载.pkl文件并查看其结构
    
    参数:
        file_path (str): .pkl文件的路径
        
    返回:
        加载的数据内容
    """
    try:
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
            
        print("文件加载成功！")
        print("\n数据结构类型:", type(data))
        
        # 如果是字典，打印键和样本值
        if isinstance(data, dict):
            print("\n字典键:", data.keys())
            print("\n前3个键值对样本:")
            for i, (k, v) in enumerate(data.items()):
                if i >= 3:
                    break
                print(f"键: {k}")
                print(f"值: {type(v)}")
                print(f"值形状: {len(v)}")
                if hasattr(v, 'shape'):
                    print(f"值形状: {v.shape}")
                print("---")
                
        # 如果是列表或元组，打印长度和前几个元素
        elif isinstance(data, (list, tuple)):
            print("\n长度:", len(data))
            print("\n前3个元素:")
            for i, item in enumerate(data[:3]):
                print(f"索引 {i}: 类型 {type(item)}")
                if hasattr(item, 'shape'):
                    print(f"形状: {item.shape}")
                print("---")
                
        # 如果是numpy数组或torch张量
        elif hasattr(data, 'shape'):
            print("\n数组/张量形状:", data.shape)
            print("数据类型:", data.dtype)
            
        return data
        
    except Exception as e:
        print(f"加载文件时出错: {e}")
        return None

# 使用示例
file_path = '/e/cuiby/paper/cby_code/embeddings_cache/labels/label_processed_biological_process.pkl'
data = load_and_inspect_pkl(file_path)

# 获取数据
label_list = data['label_list']
training_labels_binary = data['training_labels_binary']

print(f"\n{'='*50}")
print("标签详细信息:")
print(f"{'='*50}")
print(f"标签总数: {len(label_list)}")
print(f"训练样本数: {len(training_labels_binary)}")
print(f"每个样本的标签维度: {len(training_labels_binary[0]) if training_labels_binary else 0}")

# 查看前五个样本中值为1的标签对应的元素
print(f"\n{'='*50}")
print("前五个样本中值为1的标签对应的元素:")
print(f"{'='*50}")

for i in range(5):
    print(f"\n样本 {i}:")
    sample = training_labels_binary[i]
    
    # 找出该样本中值为1的索引位置
    one_indices = [idx for idx, value in enumerate(sample) if value == 1]
    
    print(f"  有 {len(one_indices)} 个标签为1")
    print(f"  占总标签数的 {len(one_indices)/len(sample)*100:.2f}%")
    
    if one_indices:
        # 获取对应的标签名称
        corresponding_labels = [label_list[idx] for idx in one_indices]
        
        # 显示前10个标签，避免输出过长
        max_display = min(10, len(corresponding_labels))
        print(f"  显示前{max_display}个标签:")
        
        for j in range(max_display):
            print(f"    位置 {one_indices[j]}: {corresponding_labels[j]}")
        
        if len(corresponding_labels) > max_display:
            print(f"    ... 还有 {len(corresponding_labels) - max_display} 个标签")
    else:
        print("  没有标签为1")
    
    print("-" * 40)

# 统计前五个样本的总体信息
print(f"\n{'='*50}")
print("前五个样本的总体统计:")
print(f"{'='*50}")

total_ones = 0
for i in range(5):
    sample = training_labels_binary[i]
    one_count = sum(sample)
    total_ones += one_count
    print(f"样本 {i}: {one_count} 个标签为1")

print(f"\n前五个样本平均每个样本有 {total_ones/5:.1f} 个标签为1")
print(f"标签稀疏度: {(1 - total_ones/(5*len(label_list)))*100:.2f}%")

# 可选：查看标签列表的前几个元素，了解标签的格式
print(f"\n{'='*50}")
print("标签列表的前10个元素:")
print(f"{'='*50}")
for i in range(min(10, len(label_list))):
    print(f"{i}: {label_list[i]}")