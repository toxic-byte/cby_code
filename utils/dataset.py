from sklearn import preprocessing
import torch
from myparser import obo_parser, ia_parser
from graph import Graph
from torch.utils.data import Dataset

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
            data['nlp_embedding'] = self.nlp_embeddings[idx]  # [seq_len, nlp_dim]
        
        return data
    
    def __len__(self):
        return len(self.sequences)