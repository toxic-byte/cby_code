from math import e
import os
import pickle
import torch
from tqdm import tqdm
import esm
from transformers import AutoTokenizer, AutoModel
import re

model, tokenizer = esm.pretrained.esm2_t30_150M_UR50D()
num_layers = 30

model.cuda()
model.eval()

def precompute_esm_embeddings(sequences, cache_file, pooling='mean'):
    if os.path.exists(cache_file):
        print(f"Loading cached ESM embeddings from {cache_file}")
        with open(cache_file, 'rb') as f:
            return pickle.load(f)
    
    print(f"Computing ESM embeddings for {len(sequences)} sequences...")
    batch_converter = tokenizer.get_batch_converter()
    embeddings = []
    
    for i, seq in enumerate(tqdm(sequences, desc="Computing ESM embeddings")):
        batch_labels, batch_strs, batch_tokens = batch_converter([("x", seq)])
        with torch.no_grad():
            batch_tokens = batch_tokens.cuda()
            results = model(batch_tokens, repr_layers=[num_layers])
            token_representations = results["representations"][num_layers]
            
            # 提取去掉 <cls>/<eos> 的部分
            plm_embed = token_representations[0, 1:1 + len(seq), :].cpu()
            
            # 池化处理
            if pooling == 'mean':
                pooled_embed = plm_embed.mean(dim=0)  # [embed_dim]
            elif pooling == 'max':
                pooled_embed, _ = plm_embed.max(dim=0)  # [embed_dim]
            elif pooling == 'cls':
                pooled_embed = token_representations[0, 0, :].cpu()  # [embed_dim]
            else:
                raise ValueError(f"Unsupported pooling method: {pooling}")
            
            embeddings.append(pooled_embed)
    
    # 保存缓存
    print(f"Saving ESM embeddings to {cache_file}")
    with open(cache_file, 'wb') as f:
        pickle.dump(embeddings, f)
    
    return embeddings

def compute_esm_embeddings(config, training_sequences, test_sequences):
    """预计算ESM embeddings"""
    print("Computing ESM embeddings...")
    
    if config['run_mode'] == "sample":
        train_esm_cache = os.path.join(config['cache_dir'], "esm/train_esm_embeddings_sample.pkl")
        test_esm_cache = os.path.join(config['cache_dir'], "esm/test_esm_embeddings_sample.pkl")
    elif config['run_mode'] == "full":
        train_esm_cache = os.path.join(config['cache_dir'], "esm/train_esm_embeddings_mean.pkl")
        test_esm_cache = os.path.join(config['cache_dir'], "esm/test_esm_embeddings_mean.pkl")
    
    train_esm_embeddings = precompute_esm_embeddings(training_sequences, train_esm_cache, pooling='mean')
    test_esm_embeddings = precompute_esm_embeddings(test_sequences, test_esm_cache, pooling='mean')
    
    return train_esm_embeddings, test_esm_embeddings


# =====================
# NLP 模型 (BiomedBERT)
# =====================
nlp_path = '/e/cuiby/huggingface/hub/models--microsoft--BiomedNLP-BiomedBERT-base-uncased-abstract/snapshots/d673b8835373c6fa116d6d8006b33d48734e305d'
nlp_tokenizer = AutoTokenizer.from_pretrained(nlp_path)
nlp_model = AutoModel.from_pretrained(nlp_path)
nlp_dim = 768
nlp_model.cuda()
nlp_model.eval()

MAXLEN = 2048


def load_nlp_model(nlp_path):
    """加载NLP模型和tokenizer"""
    nlp_tokenizer = AutoTokenizer.from_pretrained(nlp_path)
    nlp_model = AutoModel.from_pretrained(nlp_path)
    nlp_model.cuda()
    nlp_model.eval()
    return nlp_tokenizer, nlp_model

#对每一个蛋白质生成文本描述向量
def nlp_embedding(nlp_model, sample_ids, label_list, key, top_list, cache_path=None, onto=None, pooling='mean', name_flag="name"):
    """
    生成或加载NLP文本嵌入
    
    参数:
        nlp_model: BiomedBERT模型
        sample_ids: 样本ID列表,用于对齐
        label_list: 每个样本的GO标签列表
        key: GO命名空间 (biological_process, molecular_function, cellular_component)
        top_list: 高频标签列表
        cache_path: 缓存文件路径,如果提供则尝试加载/保存
        onto: GO本体对象
        pooling: 池化方式 ('mean', 'max', 'cls', None)
    
    返回:
        match_embedding: 文本嵌入列表,与sample_ids对齐
    """
    # 如果提供了缓存路径且文件存在,直接加载
    if cache_path and os.path.exists(cache_path):
        print(f"Loading NLP embeddings from cache: {cache_path}")
        with open(cache_path, 'rb') as f:
            cached_data = pickle.load(f)
        
        # 验证缓存数据的对齐性和池化方式
        if (len(cached_data['embeddings']) == len(sample_ids) and 
            cached_data.get('pooling') == pooling):
            if cached_data['sample_ids'] == sample_ids:
                print(f"Cache loaded successfully. {len(cached_data['embeddings'])} embeddings loaded.")
                return cached_data['embeddings']
            else:
                print("Warning: Sample IDs mismatch. Regenerating embeddings...")
        else:
            print(f"Warning: Cache mismatch (pooling: {cached_data.get('pooling')} vs {pooling}). Regenerating embeddings...")
    
    # 生成新的嵌入
    print(f"Generating NLP embeddings using model: {nlp_path}")
    print(f"Processing {len(label_list)} samples for ontology: {key}")
    print(f"Pooling method: {pooling}")
    
    match_embedding = []
    
    for index, multi_tag in enumerate(tqdm(label_list, desc="Generating NLP embeddings")):
        if multi_tag == []:
            # 对于没有标签的样本,创建零向量
            if pooling is None:
                match_embedding.append(torch.zeros(1, nlp_dim).cuda())
            else:
                match_embedding.append(torch.zeros(nlp_dim).cuda())
            continue
        
        context = ''
        for _tag in multi_tag:
            if _tag not in top_list:
                continue
            for ont in onto:
                if ont.namespace != key:
                    continue
                _tag = 'GO:' + _tag
                if _tag in ont.terms_dict.keys():
                    if name_flag=="name":
                        # print("Using GO names as context.")
                        tag_context = ont.terms_dict[_tag]['name']  # 只要名字,并且将一个蛋白质的所有GO名字拼接
                        context = context + tag_context + ' '
                    elif name_flag=="def":
                        # print("Using GO definitions as context.")
                        tag_context = ont.terms_dict[_tag]['def']
                        tag_contents = re.findall(r'"(.*?)"', tag_context)
                        if context == '':
                            context = context + tag_contents[0]
                        else:
                            context = context + ' ' + tag_contents[0]
        
        # 如果没有找到任何上下文,使用零向量
        if context == '':
            if pooling is None:
                match_embedding.append(torch.zeros(1, nlp_dim).cuda())
            else:
                match_embedding.append(torch.zeros(nlp_dim).cuda())
            continue
        
        # 分段处理长文本
        seq_len = 512
        max_len = MAXLEN // 2
        if len(context) > max_len:
            context = context[:max_len]
        
        num_seqs = len(context) // seq_len + (1 if len(context) % seq_len != 0 else 0)
        last_embed = []
        
        with torch.no_grad():
            for i in range(num_seqs):
                start_index = i * seq_len
                end_index = min((i + 1) * seq_len, len(context))
                context_sample = context[start_index:end_index]
                inputs = nlp_tokenizer(context_sample, return_tensors="pt")
                inputs = {k: v.cuda() for k, v in inputs.items()}
                outputs = nlp_model(**inputs)
                last_hidden_states = outputs.last_hidden_state.squeeze(0).detach()
                last_embed.append(last_hidden_states)
        
        # 合并所有段落的embeddings
        embed = torch.cat(last_embed, dim=0)  # [total_seq_len, nlp_dim]
        
        # 应用池化
        if pooling == 'mean':
            # 平均池化
            pooled_embed = embed.mean(dim=0)  # [nlp_dim]
        elif pooling == 'max':
            # 最大池化
            pooled_embed = embed.max(dim=0)[0]  # [nlp_dim]
        elif pooling == 'cls':
            # 使用CLS token (第一个token)
            pooled_embed = embed[0]  # [nlp_dim]
        elif pooling is None:
            # 不池化,保持原样
            pooled_embed = embed.unsqueeze(0)  # [1, seq_len, nlp_dim]
        else:
            raise ValueError(f"Unknown pooling method: {pooling}")
        
        match_embedding.append(pooled_embed)
    
    # 如果使用了池化,转换为tensor
    if pooling is not None:
        match_embedding = torch.stack(match_embedding)  # [num_samples, nlp_dim]
    
    # 保存到缓存
    if cache_path:
        print(f"Saving NLP embeddings to cache: {cache_path}")
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        cache_data = {
            'sample_ids': sample_ids,
            'embeddings': match_embedding,
            'key': key,
            'num_samples': len(sample_ids),
            'pooling': pooling,
            'embedding_dim': nlp_dim
        }
        with open(cache_path, 'wb') as f:
            pickle.dump(cache_data, f)
        print(f"Cache saved successfully.")
    
    print(f"NLP embedding generation completed. Total samples: {len(match_embedding)}")
    if pooling is not None:
        print(f"Embedding shape: {match_embedding.shape}")
    
    return match_embedding

# =====================
# NLP Embedding处理
# =====================
def compute_nlp_embeddings(config, nlp_model, key, train_id, test_id, training_labels, label_list, onto):
    """计算NLP embeddings"""
    if config['run_mode'] == "sample":
        train_nlp_cache = os.path.join(config['cache_dir'], f"nlp/train_nlp_embeddings_{key}_{config['text_mode']}_sample.pkl")
    elif config['run_mode'] == "full":
        train_nlp_cache = os.path.join(config['cache_dir'], f"nlp/train_nlp_embeddings_{key}_{config['text_mode']}.pkl")
    
    print(f"\n--- Processing Train NLP Embeddings for {key} ---")
    train_nlp = nlp_embedding(
        nlp_model, 
        train_id,
        training_labels[key], 
        key, 
        label_list,
        cache_path=train_nlp_cache,
        onto=onto,
        pooling='mean',
        name_flag=config['text_mode']
    )
    
    # 测试集使用零向量作为占位符
    print(f"\n--- Creating placeholder NLP Embeddings for Test Set ---")
    test_nlp = torch.zeros(len(test_id), config['nlp_dim'])
    print(f"Test NLP embeddings shape: {test_nlp.shape}")
    
    return train_nlp, test_nlp

#测试集也加上文本向量
def compute_nlp_embeddings_with_test(config, nlp_model, key, train_id, test_id, training_labels,test_labels, label_list, onto):
    """计算NLP embeddings"""
    if config['run_mode'] == "sample":
        train_nlp_cache = os.path.join(config['cache_dir'], f"nlp/train_nlp_embeddings_{key}_{config['text_mode']}_sample.pkl")
        test_nlp_cache = os.path.join(config['cache_dir'], f"nlp/test_nlp_embeddings_{key}_{config['text_mode']}_sample.pkl")
    elif config['run_mode'] == "full":
        train_nlp_cache = os.path.join(config['cache_dir'], f"nlp/train_nlp_embeddings_{key}_{config['text_mode']}.pkl")
        test_nlp_cache = os.path.join(config['cache_dir'], f"nlp/test_nlp_embeddings_{key}_{config['text_mode']}.pkl")
    
    print(f"\n--- Processing Train NLP Embeddings for {key} ---")
    train_nlp = nlp_embedding(
        nlp_model, 
        train_id,
        training_labels[key], 
        key, 
        label_list,
        cache_path=train_nlp_cache,
        onto=onto,
        pooling='mean',
        name_flag=config['text_mode']
    )

    test_nlp = nlp_embedding(
        nlp_model, 
        test_id,
        test_labels[key], 
        key, 
        label_list,
        cache_path=test_nlp_cache,
        onto=onto,
        pooling='mean',
        name_flag=config['text_mode']
    )

    
    return train_nlp, test_nlp


def list_embedding(nlp_model, key, top_list, cache_path=None, onto=None, pooling='mean', name_flag="name"):
    """
    生成或加载NLP文本嵌入(为top_list中的每个标签生成向量)
    
    参数:
        nlp_model: BiomedBERT模型
        key: GO命名空间 (biological_process, molecular_function, cellular_component)
        top_list: 高频标签列表
        cache_path: 缓存文件路径,如果提供则尝试加载/保存
        onto: GO本体对象
        pooling: 池化方式 ('mean', 'max', 'cls')
        name_flag: 使用GO的名称("name")或定义("def")
    
    返回:
        embeddings: 所有top_list标签的向量 [len(top_list), nlp_dim]
    """
    # 如果提供了缓存路径且文件存在,直接加载
    if cache_path and os.path.exists(cache_path):
        print(f"Loading NLP embeddings from cache: {cache_path}")
        with open(cache_path, 'rb') as f:
            cached_data = pickle.load(f)
        
        # 验证缓存数据
        if (cached_data.get('pooling') == pooling and 
            cached_data.get('key') == key and
            cached_data.get('top_list') == top_list):
            print(f"Cache loaded successfully. Shape: {cached_data['embeddings'].shape}")
            return cached_data['embeddings']
        else:
            print("Warning: Cache mismatch. Regenerating embeddings...")
    
    # 生成新的嵌入
    print(f"Generating NLP embeddings for {len(top_list)} GO terms")
    print(f"Ontology namespace: {key}")
    print(f"Pooling method: {pooling}")
    print(f"Using GO {name_flag}s as context")
    
    all_embeddings = []
    
    for _tag in tqdm(top_list, desc="Processing GO terms"):
        context = ''
        
        # 查找该标签的文本描述
        for ont in onto:
            if ont.namespace != key:
                continue
            
            _tag_with_prefix = 'GO:' + _tag if not _tag.startswith('GO:') else _tag
            
            if _tag_with_prefix in ont.terms_dict.keys():
                if name_flag == "name":
                    context = ont.terms_dict[_tag_with_prefix]['name']
                elif name_flag == "def":
                    tag_context = ont.terms_dict[_tag_with_prefix]['def']
                    tag_contents = re.findall(r'"(.*?)"', tag_context)
                    if tag_contents:
                        context = tag_contents[0]
                break
        
        # 如果没有找到上下文,使用零向量
        if context == '':
            print(f"Warning: No context found for {_tag}, using zero vector...")
            all_embeddings.append(torch.zeros(nlp_dim).cuda())
            continue
        
        # 分段处理长文本
        seq_len = 512
        max_len = MAXLEN // 2
        if len(context) > max_len:
            context = context[:max_len]
        
        num_seqs = len(context) // seq_len + (1 if len(context) % seq_len != 0 else 0)
        last_embed = []
        
        with torch.no_grad():
            for i in range(num_seqs):
                start_index = i * seq_len
                end_index = min((i + 1) * seq_len, len(context))
                context_sample = context[start_index:end_index]
                inputs = nlp_tokenizer(context_sample, return_tensors="pt")
                inputs = {k: v.cuda() for k, v in inputs.items()}
                outputs = nlp_model(**inputs)
                last_hidden_states = outputs.last_hidden_state.squeeze(0).detach()
                last_embed.append(last_hidden_states)
        
        # 合并所有段落的embeddings
        embed = torch.cat(last_embed, dim=0)  # [total_seq_len, nlp_dim]
        
        # 应用池化
        if pooling == 'mean':
            pooled_embed = embed.mean(dim=0)  # [nlp_dim]
        elif pooling == 'max':
            pooled_embed = embed.max(dim=0)[0]  # [nlp_dim]
        elif pooling == 'cls':
            pooled_embed = embed[0]  # [nlp_dim]
        else:
            raise ValueError(f"Unknown pooling method: {pooling}")
        
        all_embeddings.append(pooled_embed)
    
    # 堆叠所有向量
    embeddings = torch.stack(all_embeddings)  # [len(top_list), nlp_dim]
    
    print(f"NLP embedding generation completed.")
    print(f"Embedding shape: {embeddings.shape}")
    
    # 保存到缓存
    if cache_path:
        print(f"Saving NLP embeddings to cache: {cache_path}")
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        cache_data = {
            'top_list': top_list,
            'embeddings': embeddings,
            'key': key,
            'num_terms': len(top_list),
            'pooling': pooling,
            'embedding_dim': nlp_dim,
            'name_flag': name_flag
        }
        with open(cache_path, 'wb') as f:
            pickle.dump(cache_data, f)
        print(f"Cache saved successfully.")
    
    return embeddings

def compute_nlp_embeddings_list(config, nlp_model, key, label_list, onto):
    """计算NLP embeddings"""
    if config['run_mode'] == "sample":
        list_nlp_cache = os.path.join(config['cache_dir'], f"description/train_nlp_embeddings_{key}_{config['text_mode']}_sample.pkl")
    elif config['run_mode'] == "full":
        list_nlp_cache = os.path.join(config['cache_dir'], f"description/train_nlp_embeddings_{key}_{config['text_mode']}.pkl")
    
    print(f"\n--- Processing Train NLP Embeddings for {key} ---")
    embeddings  = list_embedding(
        nlp_model, 
        key, 
        label_list,
        cache_path=list_nlp_cache,
        onto=onto,
        pooling='mean',
        name_flag=config['text_mode']
    )

    return embeddings 