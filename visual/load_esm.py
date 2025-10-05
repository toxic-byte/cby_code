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
                # print(f"值: {v}")
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
# file_path = '/e/cuiby/paper/cby_code/embeddings_cache/esm/train_esm_embeddings_mean.pkl'
file_path = '/e/cuiby/paper/cby_code/embeddings_cache/labels/label_processed_biological_process.pkl'
data = load_and_inspect_pkl(file_path)
