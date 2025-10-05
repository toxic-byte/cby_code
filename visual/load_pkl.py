import pickle
import numpy as np
import torch

def analyze_embeddings_file(file_path):
    try:
        # 加载pkl文件
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
        
        print(f"文件加载成功！")
        print(f"数据类型: {type(data)}")
        
        if isinstance(data, dict):
            print(f"字典键: {list(data.keys())}")
            print(f"字典长度: {len(data)}")
            
            # 检查每个键的值类型
            for key, value in data.items():
                print(f"\n{key}:")
                print(f"  类型: {type(value)}")
                if hasattr(value, 'shape'):
                    print(f"  形状: {value.shape}")
                if hasattr(value, 'device'):
                    print(f"  设备: {value.device}")
            
            # 分析embeddings
            if 'embeddings' in data:
                embeddings = data['embeddings']
                print(f"\n=== Embeddings分析 ===")
                print(f"Embeddings类型: {type(embeddings)}")
                
                if hasattr(embeddings, 'shape'):
                    print(f"Embeddings形状: {embeddings.shape}")
                    total_vectors = embeddings.shape[0]
                    
                    # 将张量移动到CPU并转换为numpy
                    if hasattr(embeddings, 'cpu'):
                        embeddings_cpu = embeddings.cpu()
                        if hasattr(embeddings_cpu, 'numpy'):
                            embeddings_np = embeddings_cpu.numpy()
                            
                            # 检查零向量和非零向量
                            zero_vector_indices = []
                            non_zero_vector_indices = []
                            
                            for i in range(total_vectors):
                                if np.all(embeddings_np[i] == 0):
                                    zero_vector_indices.append(i)
                                else:
                                    non_zero_vector_indices.append(i)
                            
                            zero_vectors = len(zero_vector_indices)
                            non_zero_vectors = len(non_zero_vector_indices)
                            
                            print(f"\n统计结果:")
                            print(f"总向量数量: {total_vectors}")
                            print(f"零向量数量: {zero_vectors}")
                            print(f"非零向量数量: {non_zero_vectors}")
                            print(f"零向量比例: {zero_vectors/total_vectors*100:.2f}%")
                            
                            # 显示零向量和非零向量的ID
                            print(f"\n零向量ID (前5个): {zero_vector_indices[:5] if zero_vectors > 0 else '无'}")
                            print(f"非零向量ID (前5个): {non_zero_vector_indices[:5] if non_zero_vectors > 0 else '无'}")
                            
                            # 如果有sample_ids，显示对应的样本ID
                            if 'sample_ids' in data:
                                sample_ids = data['sample_ids']
                                print(f"\n样本ID信息:")
                                print(f"样本ID数量: {len(sample_ids)}")
                                
                                if zero_vectors > 0:
                                    zero_sample_ids = [sample_ids[i] for i in zero_vector_indices[:5]]
                                    print(f"零向量对应的样本ID (前5个): {zero_sample_ids}")
                                
                                if non_zero_vectors > 0:
                                    non_zero_sample_ids = [sample_ids[i] for i in non_zero_vector_indices[:5]]
                                    print(f"非零向量对应的样本ID (前5个): {non_zero_sample_ids}")
                            
                            # 显示一些样本信息
                            print(f"\n其他样本信息:")
                            if 'sample_ids' in data:
                                sample_ids = data['sample_ids']
                                print(f"前5个样本ID: {sample_ids[:5] if len(sample_ids) > 5 else sample_ids}")
                            
                            if 'embedding_dim' in data:
                                print(f"嵌入维度: {data['embedding_dim']}")
                            
                            if 'num_samples' in data:
                                print(f"样本数量: {data['num_samples']}")
                            
                            return {
                                'total_vectors': total_vectors,
                                'zero_vectors': zero_vectors,
                                'non_zero_vectors': non_zero_vectors,
                                'zero_ratio': zero_vectors/total_vectors,
                                'zero_vector_indices': zero_vector_indices,
                                'non_zero_vector_indices': non_zero_vector_indices,
                                'embedding_dim': data.get('embedding_dim', embeddings.shape[1] if len(embeddings.shape) > 1 else None)
                            }
                
        return None
        
    except Exception as e:
        print(f"加载文件时出错: {e}")
        return None

# 文件路径
# file_path = "/e/cuiby/paper/cby_code/embeddings_cache/nlp/train_nlp_embeddings_biological_process_def.pkl"
file_path = "/e/cuiby/paper/cby_code/embeddings_cache/nlp/train_nlp_embeddings_biological_process_name.pkl"
# file_path = "/e/cuiby/paper/cby_code/embeddings_cache/nlp/train_nlp_embeddings_molecular_function_name.pkl"
# file_path = "/e/cuiby/paper/cby_code/embeddings_cache/nlp/train_nlp_embeddings_cellular_component_name.pkl"

# 分析文件
result = analyze_embeddings_file(file_path)