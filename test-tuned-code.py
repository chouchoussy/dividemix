# Tạo dict chứa các tham số như argparse trong train_tuned.py
args = {
    'batch_size': 64,
    'lr': 0.02,
    'alpha': 4,
    'lambda_u': 25,
    'p_threshold': 0.5,
    'T': 0.5,
    'num_epochs': 300,
    # 'id': '',
    'seed': 123,
    'gpuid': 0,
    'num_class': 10,
    'image_size': 28,
    'warm_up': 10,
    'train_csv_path': './Data/Fashion-MNIST-test/fashion_mnist.csv',
    'train_feather_path': './Data/Fashion-MNIST-test/fashion-mnist-test-clip-b16-noise/fashion-mnist-test_LLM.feather',
    'train_data_column': 'image_name',  # Sửa lại đúng tên cột dữ liệu ảnh trong csv
    'train_label_column': 'label',      # Sửa lại đúng tên cột label trong csv
    'train_image_dir': './Data/Fashion-MNIST-test/images',
    'test_csv_path': './Data/fashion-mnist-2k5-testset/fashion-mnist-test-2k5.csv',
    'test_data_column': 'image_name',   # Sửa lại đúng tên cột dữ liệu ảnh trong csv
    'test_label_column': 'label',       # Sửa lại đúng tên cột label trong csv
    'test_image_dir': './Data/fashion-mnist-2k5-testset/images',
    'num_workers': 4
}

# Bật/tắt chạy tập clean
use_clean = True  # đặt True để chạy
print(f"use_clean: {use_clean}")

out_path = "./Data/all-train-clean-labels/fashion-mnist-test_train-clean-labels.feather"
if use_clean:
    import pandas as pd
    df = pd.read_csv(args['train_csv_path'], usecols=[args['train_label_column']])
    df.to_feather(out_path)
    args['train_feather_path'] = out_path

# Tạo chuỗi đối số từ dict
arg_str = ' '.join(f'--{k} "{v}"' if isinstance(v, str) else f'--{k} {v}' for k, v in args.items())

# In ra chuỗi lệnh python để chạy train_tuned.py
print(f'python train_tuned.py {arg_str}')
