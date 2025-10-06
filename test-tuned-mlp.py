"""
Script tạo lệnh chạy train_tuned_mlp.py với embedding input
Ví dụ cho Fashion-MNIST với CLIP embeddings
"""

# Cấu hình cho Fashion-MNIST với CLIP base-16 embeddings
args = {
    'batch_size': 64,
    'lr': 0.02,
    'alpha': 4,
    'lambda_u': 25,
    'p_threshold': 0.5,
    'T': 0.5,
    'num_epochs': 300,
    'seed': 123,
    'gpuid': 0,
    'num_class': 10,
    'embedding_dim': 512,  # CLIP base-16 embedding dimension
    'warm_up': 10,
    'dataset': 'fashion-mnist',
    'noise_type': 'semi_supervise',
    'num_workers': 8,
    
    # Train set
    'train_csv_path': './Data/Fashion-MNIST-test/fashion_mnist.csv',
    'train_embedding_feather_path': './Data/Fashion-MNIST-test/fashion-mnist-test-clip-b16-noise/fashion-mnist-test_auto.feather',  # Embedding file
    'train_noisy_label_feather_path': './Data/Fashion-MNIST-test/fashion-mnist-test-clip-b16-noise/fashion-mnist-test_semi_supervise.feather',  # Noisy labels
    'train_label_column': 'label',
    
    # Test set
    'test_csv_path': './Data/fashion-mnist-2k5-testset/fashion-mnist-test-2k5.csv',
    'test_embedding_feather_path': './Data/fashion-mnist-2k5-testset/embed/fashion-mnist-2k5-testset_embedding_clip-base-16.feather',
    'test_label_column': 'label',
}

# Tạo chuỗi đối số từ dict
arg_str = ' '.join(f'--{k} "{v}"' if isinstance(v, str) else f'--{k} {v}' for k, v in args.items())

# In ra chuỗi lệnh python để chạy train_tuned_mlp.py
print(f'python train_tuned_mlp.py {arg_str}')

print("\n" + "="*80)
print("CÁC VÍ DỤ KHÁC:")
print("="*80)

# Ví dụ cho AG News (text embeddings với BERT)
print("\n# AG News với BERT embeddings:")
agnews_args = {
    'batch_size': 64,
    'lr': 0.02,
    'num_epochs': 300,
    'warm_up': 10,
    'num_class': 4,
    'embedding_dim': 768,  # BERT embedding dimension
    'dataset': 'agnews',
    'noise_type': 'llm',
    'train_csv_path': './Data/Agnews-12k-train/ag_news_12k.csv',
    'train_embedding_feather_path': './Data/Agnews-12k-train/agnews-12k-train-bert-noise/agnews-12k-train_auto.feather',
    'train_noisy_label_feather_path': './Data/Agnews-12k-train/agnews-12k-train-bert-noise/agnews-12k-train_LLM.feather',
    'train_label_column': 'label',
    'test_csv_path': './Data/agnews-3k-testset/agnews-test-3k.csv',
    'test_embedding_feather_path': './Data/agnews-3k-testset/embed/agnews-3k-testset_embedding_bert.feather',
    'test_label_column': 'label',
    'num_workers': 8,
}
agnews_str = ' '.join(f'--{k} "{v}"' if isinstance(v, str) else f'--{k} {v}' for k, v in agnews_args.items())
print(f'python train_tuned_mlp.py {agnews_str}')

# Ví dụ cho OrganAMNIST
print("\n# OrganAMNIST với CLIP embeddings:")
organ_args = {
    'batch_size': 64,
    'lr': 0.02,
    'num_epochs': 300,
    'warm_up': 10,
    'num_class': 11,  # OrganAMNIST có 11 classes
    'embedding_dim': 512,  # CLIP base-16
    'dataset': 'organamnist',
    'noise_type': 'llm',
    'train_csv_path': './Data/OrganAMNIST-test/organamnist.csv',
    'train_embedding_feather_path': './Data/OrganAMNIST-test/organamnist-test-clip-b16-noise/organamnist-test_auto.feather',
    'train_noisy_label_feather_path': './Data/OrganAMNIST-test/organamnist-test-clip-b16-noise/organamnist-test_LLM.feather',
    'train_label_column': 'label',
    'test_csv_path': './Data/organamnist-4k5-testset/organamnist-test-4k5.csv',
    'test_embedding_feather_path': './Data/organamnist-4k5-testset/embed/organamnist-4k5-testset_embedding_clip-base-16.feather',
    'test_label_column': 'label',
    'num_workers': 8,
}
organ_str = ' '.join(f'--{k} "{v}"' if isinstance(v, str) else f'--{k} {v}' for k, v in organ_args.items())
print(f'python train_tuned_mlp.py {organ_str}')

# Ví dụ cho Resisc45
print("\n# Resisc45 với CLIP embeddings:")
resisc_args = {
    'batch_size': 64,
    'lr': 0.02,
    'num_epochs': 300,
    'warm_up': 10,
    'num_class': 45,  # Resisc45 có 45 classes
    'embedding_dim': 512,  # CLIP base-16
    'dataset': 'resisc45',
    'noise_type': 'semi_supervise',
    'train_csv_path': './Data/Resisc45-train/resisc45-train.csv',
    'train_embedding_feather_path': './Data/Resisc45-train/resisc45-train-clip-b16-noise/resisc-45-train_auto.feather',
    'train_noisy_label_feather_path': './Data/Resisc45-train/resisc45-train-clip-b16-noise/resisc45-train_semi_supervise.feather',
    'train_label_column': 'label',
    'test_csv_path': './Data/resisc45-4k750-testset/resisc45-test-4k750.csv',
    'test_embedding_feather_path': './Data/resisc45-4k750-testset/embed/resisc45-4k750-testset_embedding_clip-base-16.feather',
    'test_label_column': 'label',
    'num_workers': 8,
}
resisc_str = ' '.join(f'--{k} "{v}"' if isinstance(v, str) else f'--{k} {v}' for k, v in resisc_args.items())
print(f'python train_tuned_mlp.py {resisc_str}')

print("\n" + "="*80)
print("LƯU Ý:")
print("="*80)
print("""
1. embedding_dim: 
   - CLIP base-16: 512
   - BERT: 768
   - Kiểm tra bằng: pd.read_feather('path').shape[1]

2. train_embedding_feather_path: file chứa embeddings (thường là *_auto.feather)
   train_noisy_label_feather_path: file chứa noisy labels (ví dụ: *_LLM.feather, *_semi_supervise.feather)

3. Các file embedding có sẵn trong Data/:
   - Fashion-MNIST: CLIP embeddings
   - AG News: BERT embeddings  
   - OrganAMNIST: CLIP embeddings
   - Resisc45: CLIP embeddings
   - Yahoo: BERT embeddings

4. Để chạy với đường dẫn tuyệt đối, thay './' bằng '/Users/linhnh/Documents/DivideMix-tuned/'
""")
