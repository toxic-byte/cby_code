import pickle
import torch
import numpy as np

def compare_pkl_files(file1_path, file2_path):
    """
    比较两个pkl文件的所有内容
    """
    try:
        # 加载两个文件
        with open(file1_path, 'rb') as f1:
            data1 = pickle.load(f1)
        
        with open(file2_path, 'rb') as f2:
            data2 = pickle.load(f2)
        
        print("文件加载成功！")
        print(f"文件1类型: {type(data1)}")
        print(f"文件2类型: {type(data2)}")
        print()
        
        # 检查数据类型是否相同
        if type(data1) != type(data2):
            print("❌ 数据类型不同！")
            return False
        
        # 如果都是字典，比较所有键值对
        if isinstance(data1, dict):
            print("=== 字典键比较 ===")
            keys1 = set(data1.keys())
            keys2 = set(data2.keys())
            
            if keys1 == keys2:
                print("✅ 字典键相同")
                print(f"共有键: {list(keys1)}")
            else:
                print("❌ 字典键不同")
                print(f"文件1独有的键: {keys1 - keys2}")
                print(f"文件2独有的键: {keys2 - keys1}")
                return False
            
            print("\n=== 键值对详细比较 ===")
            all_same = True
            
            for key in keys1:
                print(f"\n比较键: '{key}'")
                value1 = data1[key]
                value2 = data2[key]
                
                print(f"  文件1值类型: {type(value1)}")
                print(f"  文件2值类型: {type(value2)}")
                
                # 类型检查
                if type(value1) != type(value2):
                    print(f"  ❌ 类型不同: {type(value1)} vs {type(value2)}")
                    all_same = False
                    continue
                
                # 张量比较
                if hasattr(value1, 'shape') and hasattr(value2, 'shape'):
                    print(f"  文件1形状: {value1.shape}")
                    print(f"  文件2形状: {value2.shape}")
                    
                    if value1.shape != value2.shape:
                        print(f"  ❌ 形状不同: {value1.shape} vs {value2.shape}")
                        all_same = False
                        continue
                    
                    # 将张量移动到CPU进行比较
                    if hasattr(value1, 'cpu'):
                        value1_cpu = value1.cpu()
                        value2_cpu = value2.cpu()
                        
                        # 检查是否完全相等
                        if torch.equal(value1_cpu, value2_cpu):
                            print(f"  ✅ 张量内容完全相同")
                        else:
                            print(f"  ❌ 张量内容不同")
                            # 计算差异
                            diff = torch.abs(value1_cpu - value2_cpu)
                            max_diff = torch.max(diff).item()
                            mean_diff = torch.mean(diff).item()
                            print(f"      最大差异: {max_diff:.6f}")
                            print(f"      平均差异: {mean_diff:.6f}")
                            all_same = False
                
                # 列表比较
                elif isinstance(value1, (list, tuple)):
                    print(f"  文件1长度: {len(value1)}")
                    print(f"  文件2长度: {len(value2)}")
                    
                    if len(value1) != len(value2):
                        print(f"  ❌ 长度不同: {len(value1)} vs {len(value2)}")
                        all_same = False
                        continue
                    
                    if value1 == value2:
                        print(f"  ✅ 列表内容完全相同")
                    else:
                        print(f"  ❌ 列表内容不同")
                        # 找出不同的元素
                        for i, (v1, v2) in enumerate(zip(value1, value2)):
                            if v1 != v2:
                                print(f"      第{i}个元素不同: {v1} vs {v2}")
                                if i >= 3:  # 只显示前3个不同的元素
                                    print(f"      ... 还有更多不同")
                                    break
                        all_same = False
                
                # 标量或其他类型比较
                else:
                    if value1 == value2:
                        print(f"  ✅ 值相同: {value1}")
                    else:
                        print(f"  ❌ 值不同: {value1} vs {value2}")
                        all_same = False
            
            print(f"\n=== 总体比较结果 ===")
            if all_same:
                print("🎉 两个文件内容完全相同！")
            else:
                print("⚠️ 两个文件内容有差异")
            
            return all_same
            
        else:
            print("文件不是字典类型，使用直接比较")
            if data1 == data2:
                print("🎉 两个文件内容完全相同！")
                return True
            else:
                print("❌ 两个文件内容不同")
                return False
                
    except Exception as e:
        print(f"比较文件时出错: {e}")
        return False

def analyze_individual_file(file_path, file_name):
    """
    分析单个文件的详细信息
    """
    print(f"\n{'='*50}")
    print(f"分析文件: {file_name}")
    print(f"{'='*50}")
    
    try:
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
        
        print(f"数据类型: {type(data)}")
        
        if isinstance(data, dict):
            print(f"字典键: {list(data.keys())}")
            
            for key, value in data.items():
                print(f"\n{key}:")
                print(f"  类型: {type(value)}")
                if hasattr(value, 'shape'):
                    print(f"  形状: {value.shape}")
                    if hasattr(value, 'device'):
                        print(f"  设备: {value.device}")
                    
                    # 如果是embeddings，统计零向量
                    if key == 'embeddings':
                        total_vectors = value.shape[0]
                        # 使用向量化方法检查零向量
                        norms = torch.norm(value, dim=1)
                        zero_vectors = torch.sum(norms == 0).item()
                        print(f"  总向量数: {total_vectors}")
                        print(f"  零向量数: {zero_vectors}")
                        print(f"  零向量比例: {zero_vectors/total_vectors*100:.2f}%")
                
                elif isinstance(value, (list, tuple)):
                    print(f"  长度: {len(value)}")
                    if len(value) > 0:
                        print(f"  前3个元素: {value[:3]}")
                
                else:
                    print(f"  值: {value}")
                    
    except Exception as e:
        print(f"分析文件时出错: {e}")

# 文件路径
file1_path = "/e/cuiby/paper/cby_code/embeddings_cache/nlp/train_nlp_embeddings_biological_process_def.pkl"
file2_path = "/e/cuiby/paper/cby_code/embeddings_cache/nlp/train_nlp_embeddings_biological_process_name.pkl"

# 首先分别分析每个文件
analyze_individual_file(file1_path, "biological_process_def.pkl")
analyze_individual_file(file2_path, "biological_process_name.pkl")

print(f"\n{'='*80}")
print("开始比较两个文件...")
print(f"{'='*80}")

# 然后比较两个文件
compare_pkl_files(file1_path, file2_path)