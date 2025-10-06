# =====================
# 每次运行注意修改run_mode（full/sample）和text_mode(def/name)
# =====================

from datetime import datetime
from sklearn import preprocessing
import sys

sys.path.append(r"utils")
sys.path.append(r"models")
sys.path.append(r"trainer")
from dataset import obo_graph,load_datasets,verify_alignment,process_labels_for_ontology,create_dataloaders
from config import setup_environment, get_config
from embed import load_nlp_model,compute_esm_embeddings,compute_nlp_embeddings_with_test
from trainer_combine import train_model_for_ontology,save_results


def main():
    """主函数"""
    # 初始化
    seed = setup_environment()
    config = get_config(run_mode="full", text_mode="def")
    ctime = datetime.now().strftime("%Y%m%d%H%M%S")
    print('Start running date:{}'.format(ctime))
    
    # 加载模型
    nlp_tokenizer, nlp_model = load_nlp_model(config['nlp_path'])
    
    # 初始化标签空间
    label_space = {
        'biological_process': [],
        'molecular_function': [],
        'cellular_component': []
    }
    enc = preprocessing.LabelEncoder()
    
    # 加载本体和IA字典
    onto, ia_dict = obo_graph(config['obo_path'], config['ia_path'])
    
    # 加载数据集
    train_id, training_sequences, training_labels, test_id, test_sequences, test_labels = load_datasets(
        config, onto, label_space)
    
    # 计算ESM embeddings
    train_esm_embeddings, test_esm_embeddings = compute_esm_embeddings(
        config, training_sequences, test_sequences)
    
    # 存储所有本体的预测结果和指标
    metrics_output_test = {}
    
    # 为每个本体训练模型
    for key in label_space.keys():
        print(f"\n{'='*50}")
        print(f"Training for ontology: {key}")
        print(f"{'='*50}")
        
        # 处理标签
        label_list, training_labels_binary, test_labels_binary, enc, ia_list, onto_parent, label_num = process_labels_for_ontology(
            config, key, label_space, training_labels, test_labels, onto, enc, ia_dict)
        
        # 计算NLP embeddings
        train_nlp, test_nlp = compute_nlp_embeddings_with_test(
            config, nlp_model, key, train_id, test_id, training_labels, test_labels,label_list, onto)
        
        # 验证对齐性
        verify_alignment(train_esm_embeddings, train_nlp, train_id, 
                        test_esm_embeddings, test_nlp, test_id)
        
        # 创建DataLoader
        train_dataloader, test_dataloader = create_dataloaders(
            config, training_sequences, training_labels_binary, train_esm_embeddings, train_nlp,
            test_sequences, test_labels_binary, test_esm_embeddings, test_nlp)
        
        # 训练模型
        model = train_model_for_ontology(
            config, key, train_dataloader, test_dataloader, label_num, ia_list, ctime, metrics_output_test)
    
    # 保存结果
    save_results(config, metrics_output_test, seed, ctime)
    print('End running date:{}'.format(datetime.now().strftime("%Y%m%d%H%M%S")))


if __name__ == "__main__":
    main()