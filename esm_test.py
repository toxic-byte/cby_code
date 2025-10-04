import esm
model, tokenizer = esm.pretrained.esm2_t30_150M_UR50D()
num_layers = 30
embed_dim = 640

model.cuda()
model.eval()
batch_converter=tokenizer.get_batch_converter()
def compare_padding():
    test_sequences = ["MKTLVV", "MKTLVVAAACDEFGH", "MK"]
    
    print("="*80)
    print("方法1: 单序列处理 (第一个代码)")
    print("="*80)
    
    total_tokens_method1 = 0
    for i, seq in enumerate(test_sequences):
        batch_labels, batch_strs, batch_tokens = batch_converter([("x", seq)])
        n_tokens = batch_tokens.shape[1]
        n_padding = (batch_tokens == tokenizer.padding_idx).sum().item()
        total_tokens_method1 += n_tokens
        
        print(f"Seq {i}: '{seq}'")
        print(f"  Shape: {batch_tokens.shape}")
        print(f"  Tokens: {n_tokens}, Padding: {n_padding}")
        print(f"  Actual: {batch_tokens.tolist()[0][:10]}")
        print()
    
    print(f"总计tokens: {total_tokens_method1}\n")
    
    print("="*80)
    print("方法2: 批处理 (第二个代码)")
    print("="*80)
    
    batch_data = [(f"seq_{i}", seq) for i, seq in enumerate(test_sequences)]
    batch_labels, batch_strs, batch_tokens = batch_converter(batch_data)
    
    total_tokens_method2 = batch_tokens.numel()
    
    for i, seq in enumerate(test_sequences):
        n_tokens = batch_tokens.shape[1]
        n_padding = (batch_tokens[i] == tokenizer.padding_idx).sum().item()
        
        print(f"Seq {i}: '{seq}'")
        print(f"  Shape: {batch_tokens[i].shape}")
        print(f"  Tokens: {n_tokens}, Padding: {n_padding} ({100*n_padding/n_tokens:.1f}%)")
        print(f"  Actual: {batch_tokens[i].tolist()[:10]}")
        print()
    
    print(f"总计tokens: {total_tokens_method2}")
    print(f"\n浪费比例: {100*(total_tokens_method2 - total_tokens_method1)/total_tokens_method2:.1f}%")

compare_padding()