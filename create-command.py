"""
Script tạo dòng lệnh chạy train_tuned.py (DivideMix).
Tên file log = dataset + '_' + noise_type + '_' + timestamp.
VD: logs/agnews_llm_20250906_191530.log
"""

import os
import re

# ==== Sửa cấu hình theo bài của bạn ====
config = {
    # ----- DivideMix / Train args -----
    "seed": 42,
    "batch_size": 64,       # default: 64
    "num_epochs": 300,      # default: 300
    "warm_up": 10,          # default: 10

    # ----- Dataset meta -----
    "dataset": "fashion-mnist",
    "noise_type": "llm",
    "data_type": "image",
    "num_class": 10,
    "image_size": 224,

    # ----- Train set -----
    "train_csv_path": "Data/Fashion-MNIST-test/fashion_mnist.csv",
    "train_feather_path": "Data/Fashion-MNIST-test/fashion-mnist-test-clip-b16-noise/fashion-mnist-test_LLM.feather",
    "train_data_column": "image_name",
    "train_label_column": "label",
    "train_image_dir": "Data/Fashion-MNIST-test/images",

    # ----- Test set -----
    "test_csv_path": "Data/fashion-mnist-2k5-testset/fashion-mnist-test-2k5.csv",
    "test_data_column": "image_name",
    "test_label_column": "label",
    "test_image_dir": "Data/fashion-mnist-2k5-testset/images",

    # ----- Loader -----
    "num_workers": 4,
}

def _sanitize(name: str) -> str:
    # Chỉ giữ chữ/số/dấu chấm/gạch/underscore để an toàn khi đặt tên file
    return re.sub(r'[^A-Za-z0-9._-]+', '_', name.strip())

def _log_file_from_cfg(cfg) -> str:
    dataset = _sanitize(str(cfg.get("dataset", "run")))
    noise = _sanitize(str(cfg.get("noise_type", "noise")))
    return f'logs/{dataset}_{noise}_$(date +%Y%m%d_%H%M%S).log'

def _fmt_arg(k: str, v) -> str:
    # Quote giá trị chuỗi nếu có khoảng trắng hoặc dấu phẩy
    if isinstance(v, str):
        if (" " in v) or ("," in v):
            return f'--{k} "{v}"'
        else:
            return f"--{k} {v}"
    else:
        return f"--{k} {v}"

def build_command(cfg):
    cmd = ["python train_tuned.py"]
    for k, v in cfg.items():
        if v is None:
            continue
        cmd.append(_fmt_arg(k, v))
    log_file = _log_file_from_cfg(cfg)
    cmd.append(f'2>&1 | tee "{log_file}"')
    return " ".join(cmd)

if __name__ == "__main__":
    # Tạo thư mục logs trước khi chạy
    print("mkdir -p logs")
    print(build_command(config))
