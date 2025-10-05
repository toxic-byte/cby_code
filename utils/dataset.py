from sklearn import preprocessing
import torch
from myparser import obo_parser, ia_parser
from graph import Graph
from torch.utils.data import Dataset,DataLoader
import os
import pickle
from collections import Counter
from tqdm import tqdm

#taxonomy inputs
def extract_know(filepath):
    path = filepath
    train_id = []
    train_tax = []
    print(f"Start reading {path}")
    with open(path, "r") as lines:
        for _line in lines:
            if _line.startswith('EntryID'):
                continue
            _line = _line.strip()
            seqs = _line.split()
            _id = seqs[0]
            train_id.append(_id)
            _tax = seqs[1]
            train_tax.append(_tax)
    print("Read input complete")
    return train_id, train_tax

# taxonomy encoding
def embed_know(ground, extract, taxon):
    tax_set = set(taxon)
    tax_list = list(tax_set)
    enctax = preprocessing.LabelEncoder()
    taxspace = enctax.fit_transform(tax_list)
    tax_num = len(enctax.classes_)
    ground_taxon = [taxon[extract.index(_id)] for _id in ground if _id in extract]
    tax_list = []
    tax_fin = []
    for i in range(len(ground_taxon)):
        extract_tax = torch.zeros(tax_num)
        tax_list.append(extract_tax)
    for i, _taxon in enumerate(ground_taxon):
        temp_tax = enctax.transform([_taxon])
        tax_list[i][temp_tax[0]] = 1.0
    for _tax in tax_list:
        tax_fin.append(_tax)
    return tax_fin

#good_version
def embed_know_good(ground, extract, taxon):
    tax_set = set(taxon)
    tax_list = list(tax_set)
    enctax = preprocessing.LabelEncoder()
    taxspace = enctax.fit_transform(tax_list)
    tax_num = len(enctax.classes_)
    ground_taxon = [taxon[extract.index(_id)] for _id in ground if _id in extract]
    tax_list = []
    tax_fin = []
    for i in range(len(ground_taxon)):
        extract_tax = torch.zeros(tax_num)
        tax_list.append(extract_tax)
    for i, _taxon in enumerate(ground_taxon):
        temp_tax = enctax.transform([_taxon])
        tax_list[i][temp_tax[0]] = 1.0
    for _tax in tax_list:
        # trans_tax = model_tax(_tax)
        tax_fin.append(_tax)
    return tax_fin, tax_num

# Parse the OBO file and creates a different graph for each namespace
def obo_graph(filepath, dict_path):
    ia_dict = None
    if dict_path is not None:
        ia_dict = ia_parser(dict_path)

    ontologies = []
    no_orphans = False
    for ns, terms_dict in obo_parser(filepath).items():
        ontologies.append(Graph(ns, terms_dict, ia_dict, not no_orphans))
    return ontologies, ia_dict

def parent(enc, key, label_list,onto,label_space):
    onto_parent = {}
    label_num = len(enc.classes_)
    for i in range(label_num):
        _label = enc.inverse_transform([i])
        _tag = 'GO:' + str(_label[0])
        if i not in onto_parent.keys():
            onto_parent[i] = {
                'size': 0,
                'pos': []
            }
        for ont in onto:
            if ont.namespace != key:
                continue
            for term in ont.terms_list:
                if term['id'] == _tag:
                    ns = ont.namespace
                    parent_ids = term['adj']
                    if len(parent_ids) == 0:
                        continue
                    else:
                        for _parent in parent_ids:
                            for _key, val in ont.terms_dict.items():
                                if 'index' in val and val['index'] == _parent:
                                    poss_tags = _key[3:]
                                    if poss_tags not in label_space[
                                        key]:  # 'alt_id' is used in this version, has to exclude from ground-truth label space
                                        continue
                                    if poss_tags not in label_list:
                                        continue
                                    _pos = enc.transform([poss_tags])
                                    onto_parent[i]['size'] += 1
                                    onto_parent[i]['pos'].extend(_pos)
    return onto_parent

def preprocess_dataset(filepath, MAXLEN,onto,label_space):
    '''
        Args:
            sequences: list, the list which contains the protein primary sequences.
            labels: list, the list which contains the dataset labels.
            max_length, Integer, the maximum sequence length,
            if there is a sequence that is larger than the specified sequence length will be post-truncated.
    '''
    pro_id = []
    sequences = []
    labels = {
        'biological_process': [],
        'molecular_function': [],
        'cellular_component': []
    }
    multi_labels = {
        'biological_process': [],
        'molecular_function': [],
        'cellular_component': []
    }
    path = filepath
    print(f"Start reading {path}")
    with open(path, "r") as lines:
        for _line in lines:
            if _line.startswith('>'):
                _line = _line.strip()
                seqs = _line.split()
                _id = seqs[0][1:]
                pro_id.append(_id)
                tags = seqs[1].split(';')
                for tag in tags:
                    gene = 'GO:' + tag
                    for ont in onto:
                        ns = ont.namespace
                        if gene in ont.terms_dict.keys():
                            multi_labels[ns].append(tag)
                            label_space[ns].append(tag)
                            continue
                for key in multi_labels.keys():
                    labels[key].append(multi_labels[key])
                multi_labels = {
                    'biological_process': [],
                    'molecular_function': [],
                    'cellular_component': []
                }
            else:
                _line = _line.strip()
                if len(_line) > MAXLEN:
                    _line = _line[:MAXLEN]
                sequences.append(_line)
    print("Read input complete")
    return pro_id, sequences, labels

# Organize sequences embeddings
class StabilitylandscapeDataset(Dataset):
    def __init__(self, sequences, labels):
        assert len(sequences) == len(labels), \
            f"初始化时长度不匹配！sequences: {len(sequences)}, labels: {len(labels)}"
        self.sequences = sequences
        self.labels = labels

    def __getitem__(self, idx):
        embedding = self.sequences[idx]
        label = self.labels[idx]
        return {'embed': embedding, 'labels': torch.as_tensor(label, dtype=torch.float32).clone().detach()}

    def __len__(self):
        return len(self.sequences)
    
class IndexedStabilitylandscapeDataset(StabilitylandscapeDataset):
    def __init__(self, sequences, labels, embeddings=None, nlp_embeddings=None):
        super().__init__(sequences, labels)
        self.embeddings = embeddings
        self.nlp_embeddings = nlp_embeddings  # 添加NLP embeddings
    
    def __getitem__(self, idx):
        data = super().__getitem__(idx)
        data['index'] = idx  # 添加索引信息
        
        # 如果有预计算的ESM embeddings,直接使用池化后的结果
        if self.embeddings is not None:
            data['embedding'] = self.embeddings[idx]  # 已经是池化后的 [embed_dim]
        
        # 如果有预计算的NLP embeddings,添加到data中
        if self.nlp_embeddings is not None:
            data['nlp_embedding'] = self.nlp_embeddings[idx]  #已经是池化后的[nlp_dim]
        
        return data
    
    def __len__(self):
        return len(self.sequences)


# =====================
# 数据加载和预处理
# =====================
def load_datasets(config, onto, label_space):
    """加载训练和测试数据集"""
    print("Loading datasets...")
    train_id, training_sequences, training_labels = preprocess_dataset(
        config['train_path'], config['MAXLEN'], onto, label_space
    )
    test_id, test_sequences, test_labels = preprocess_dataset(
        config['test_path'], config['MAXLEN'], onto, label_space
    )
    
    print("Train IDs (first 5):", train_id[:5])
    print("Test IDs (first 5):", test_id[:5])
    print(f"Total train samples: {len(train_id)}, Total test samples: {len(test_id)}")
    
    return train_id, training_sequences, training_labels, test_id, test_sequences, test_labels

# =====================
# 标签处理
# =====================
def process_labels_for_ontology(config, key, label_space, training_labels, test_labels, onto, enc, ia_dict):
    """处理特定本体的标签"""
    print(f"\n{'='*50}")
    print(f"Processing labels for ontology: {key}")
    print(f"{'='*50}")
    
    if config['run_mode'] == "sample":
        label_processing_cache = os.path.join(config['cache_dir'], f"labels/label_processed_{key}_sample.pkl")
    elif config['run_mode'] == "full":
        label_processing_cache = os.path.join(config['cache_dir'], f"labels/label_processed_{key}.pkl")
    
    if os.path.exists(label_processing_cache):
        print(f"Loading preprocessed labels for {key} from cache...")
        with open(label_processing_cache, 'rb') as f:
            cached = pickle.load(f)
        
        return (cached['label_list'], cached['training_labels_binary'], 
                cached['test_labels_binary'], cached['encoder'], 
                cached['ia_list'], cached['onto_parent'], cached['label_num'])
    
    print(f"Processing labels for {key} from scratch...")
    
    # 筛选高频标签
    label_tops = Counter(label_space[key])
    top_labels = sorted([label for label in set(label_space[key]) if label_tops[label] > 21])
    print(f'Top label numbers: {len(top_labels)}')
    label_list = top_labels
    print("Top labels (first 10):", label_list[:10])
    
    # 标签编码
    labspace = enc.fit_transform(label_list)
    onto_parent = parent(enc, key, label_list, onto, label_space)
    label_num = len(enc.classes_)
    print(f'Number of classes: {label_num}')
    
    # 转换标签为二进制格式
    label_set = set(label_list)
    training_labels_binary = convert_labels_to_binary(training_labels[key], label_set, enc, label_num)
    test_labels_binary = convert_labels_to_binary(test_labels[key], label_set, enc, label_num)
    
    # 构建IA权重矩阵
    ia_list = build_ia_weight_matrix(ia_dict, label_set, enc, label_num)
    
    # 保存处理结果
    print(f"Saving processed labels to {label_processing_cache}")
    with open(label_processing_cache, 'wb') as f:
        pickle.dump({
            'label_list': label_list,
            'training_labels_binary': training_labels_binary,
            'test_labels_binary': test_labels_binary,
            'encoder': enc,
            'ia_list': ia_list,
            'onto_parent': onto_parent,
            'label_num': label_num,
        }, f)
    print("✓ Saved processed labels")
    
    return label_list, training_labels_binary, test_labels_binary, enc, ia_list, onto_parent, label_num

def convert_labels_to_binary(labels, label_set, enc, label_num):
    """将标签转换为二进制格式"""
    print("Converting labels to binary format...")
    labels_binary = []
    for label in tqdm(labels, desc="Processing labels"):
        filtered_label = [item for item in label if item in label_set]
        if len(filtered_label) == 0:
            labels_binary.append([0] * label_num)
        else:
            temp_labels = enc.transform(filtered_label)
            binary_label = [0] * label_num
            for idx in temp_labels:
                binary_label[idx] = 1
            labels_binary.append(binary_label)
    return labels_binary


def build_ia_weight_matrix(ia_dict, label_set, enc, label_num):
    """构建IA权重矩阵"""
    print("Building IA weight matrix...")
    ia_list = torch.ones(1, label_num).cuda()
    for _tag, _value in ia_dict.items():
        _tag = _tag[3:]
        if _tag not in label_set:
            continue
        ia_id = enc.transform([_tag])
        if _value == 0.0:
            _value = 1.0
        ia_list[0, ia_id[0]] = _value
    return ia_list

def verify_alignment(train_esm_embeddings, train_nlp, train_id, test_esm_embeddings, test_nlp, test_id):
    """验证数据对齐性"""
    print(f"\nAlignment verification:")
    print(f"Train samples: ESM={len(train_esm_embeddings)}, NLP={len(train_nlp)}, IDs={len(train_id)}")
    print(f"Test samples: ESM={len(test_esm_embeddings)}, NLP={len(test_nlp)}, IDs={len(test_id)}")
    assert len(train_esm_embeddings) == len(train_nlp) == len(train_id), "Train data alignment error!"
    assert len(test_esm_embeddings) == len(test_nlp) == len(test_id), f"Test data alignment error!"

# =====================
# 数据集和DataLoader创建
# =====================
def create_dataloaders(config, training_sequences, training_labels_binary, train_esm_embeddings, train_nlp,
                       test_sequences, test_labels_binary, test_esm_embeddings, test_nlp):
    """创建训练和测试DataLoader"""
    training_dataset = IndexedStabilitylandscapeDataset(
        training_sequences, 
        training_labels_binary, 
        embeddings=train_esm_embeddings,
        nlp_embeddings=train_nlp
    )
    test_dataset = IndexedStabilitylandscapeDataset(
        test_sequences, 
        test_labels_binary, 
        embeddings=test_esm_embeddings,
        nlp_embeddings=test_nlp
    )
    
    train_dataloader = DataLoader(
        training_dataset, 
        batch_size=config['batch_size_train'], 
        shuffle=True
    )
    test_dataloader = DataLoader(
        test_dataset, 
        batch_size=config['batch_size_test'], 
        shuffle=False
    )
    
    return train_dataloader, test_dataloader