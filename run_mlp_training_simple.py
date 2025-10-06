"""
Script đơn giản hơn - chỉ cần sửa config và chạy
Không có prompt xác nhận, chạy trực tiếp
"""
import subprocess
import sys
from pathlib import Path

# ============================================================================
# CẤU HÌNH - CHỈ CẦN SỬA PHẦN NÀY
# ============================================================================

# Chọn dataset
DATASET = "fashion-mnist"  # Có thể đổi: fashion-mnist, agnews, organamnist, resisc45, yahoo

# Loại noise
NOISE_TYPE = "LLM"  # Có thể đổi: llm, semi_supervise, clf, auto, weak_supervise

# Hyperparameters (có thể điều chỉnh)
BATCH_SIZE = 64
LEARNING_RATE = 0.02
NUM_EPOCHS = 300
WARM_UP = 10
NUM_WORKERS = 0  # Set to 0 for macOS to avoid multiprocessing issues (or try 2-4)
GPU_ID = 0

# ============================================================================
# ĐƯỜNG DẪN FILES - SỬA THEO BỘ DỮ LIỆU CỦA BẠN
# ============================================================================

PROJECT_ROOT = Path(__file__).parent.absolute()

# Mapping dataset -> config
DATASET_CONFIGS = {
    "fashion-mnist": {
        "num_class": 10,
        "embedding_dim": 512,
        "train_csv": "Data/Fashion-MNIST-test/fashion_mnist.csv",
        "train_embedding": "Data/Fashion-MNIST-test/fashion-mnist-test-clip-b16-noise/fashion-mnist-test_auto.feather",
        "train_noisy_label": f"Data/Fashion-MNIST-test/fashion-mnist-test-clip-b16-noise/fashion-mnist-test_{NOISE_TYPE}.feather",
        "test_csv": "Data/fashion-mnist-2k5-testset/fashion-mnist-test-2k5.csv",
        "test_embedding": "Data/fashion-mnist-2k5-testset/embed/fashion-mnist-2k5-testset_embedding_clip-base-16.feather",
    },
    "agnews": {
        "num_class": 4,
        "embedding_dim": 768,
        "train_csv": "Data/Agnews-12k-train/ag_news_12k.csv",
        "train_embedding": "Data/Agnews-12k-train/agnews-12k-train-bert-noise/agnews-12k-train_auto.feather",
        "train_noisy_label": f"Data/Agnews-12k-train/agnews-12k-train-bert-noise/agnews-12k-train_{NOISE_TYPE}.feather",
        "test_csv": "Data/agnews-3k-testset/agnews-test-3k.csv",
        "test_embedding": "Data/agnews-3k-testset/embed/agnews-3k-testset_embedding_bert.feather",
    },
    "organamnist": {
        "num_class": 11,
        "embedding_dim": 512,
        "train_csv": "Data/OrganAMNIST-test/organamnist.csv",
        "train_embedding": "Data/OrganAMNIST-test/organamnist-test-clip-b16-noise/organamnist-test_auto.feather",
        "train_noisy_label": f"Data/OrganAMNIST-test/organamnist-test-clip-b16-noise/organamnist-test_{NOISE_TYPE}.feather",
        "test_csv": "Data/organamnist-4k5-testset/organamnist-test-4k5.csv",
        "test_embedding": "Data/organamnist-4k5-testset/embed/organamnist-4k5-testset_embedding_clip-base-16.feather",
    },
    "resisc45": {
        "num_class": 45,
        "embedding_dim": 512,
        "train_csv": "Data/Resisc45-train/resisc45-train.csv",
        "train_embedding": "Data/Resisc45-train/resisc45-train-clip-b16-noise/resisc-45-train_auto.feather",
        "train_noisy_label": f"Data/Resisc45-train/resisc45-train-clip-b16-noise/resisc45-train_{NOISE_TYPE}.feather",
        "test_csv": "Data/resisc45-4k750-testset/resisc45-test-4k750.csv",
        "test_embedding": "Data/resisc45-4k750-testset/embed/resisc45-4k750-testset_embedding_clip-base-16.feather",
    },
    "yahoo": {
        "num_class": 10,
        "embedding_dim": 768,
        "train_csv": "Data/Yahoo-10k-train/yahoo-10k-train.csv",
        "train_embedding": "Data/Yahoo-10k-train/Yahoo-10k-train-bert-noise/yahoo-10k-train_auto.feather",
        "train_noisy_label": f"Data/Yahoo-10k-train/Yahoo-10k-train-bert-noise/yahoo-10k_{NOISE_TYPE}.feather",
        "test_csv": "Data/yahoo-2k5-testset/yahoo_answers-test-2k5.csv",
        "test_embedding": "Data/yahoo-2k5-testset/embed/yahoo-2k5-testset_embedding_bert.feather",
    },
}

# ============================================================================
# MAIN
# ============================================================================

def main():
    if DATASET not in DATASET_CONFIGS:
        print(f"❌ Dataset '{DATASET}' không hợp lệ!")
        print(f"Các dataset có sẵn: {', '.join(DATASET_CONFIGS.keys())}")
        sys.exit(1)
    
    cfg = DATASET_CONFIGS[DATASET]
    
    # Build command
    cmd = [
        sys.executable,
        str(PROJECT_ROOT / "train_tuned_mlp.py"),
        "--dataset", DATASET,
        "--noise_type", NOISE_TYPE,
        "--batch_size", str(BATCH_SIZE),
        "--lr", str(LEARNING_RATE),
        "--num_epochs", str(NUM_EPOCHS),
        "--warm_up", str(WARM_UP),
        "--num_class", str(cfg["num_class"]),
        "--embedding_dim", str(cfg["embedding_dim"]),
        "--num_workers", str(NUM_WORKERS),
        "--gpuid", str(GPU_ID),
        "--train_csv_path", str(PROJECT_ROOT / cfg["train_csv"]),
        "--train_embedding_feather_path", str(PROJECT_ROOT / cfg["train_embedding"]),
        "--train_noisy_label_feather_path", str(PROJECT_ROOT / cfg["train_noisy_label"]),
        "--train_label_column", "label",
        "--test_csv_path", str(PROJECT_ROOT / cfg["test_csv"]),
        "--test_embedding_feather_path", str(PROJECT_ROOT / cfg["test_embedding"]),
        "--test_label_column", "label",
    ]
    
    print("=" * 80)
    print(f"🚀 Training: {DATASET.upper()} | Noise: {NOISE_TYPE} | Epochs: {NUM_EPOCHS}")
    print("=" * 80)
    print(f"Embedding dim: {cfg['embedding_dim']} | Num classes: {cfg['num_class']}")
    print(f"Batch size: {BATCH_SIZE} | LR: {LEARNING_RATE}")
    print("=" * 80)
    print()
    
    try:
        subprocess.run(cmd, check=True)
        print()
        print("=" * 80)
        print("✅ Training hoàn thành!")
        print("=" * 80)
    except subprocess.CalledProcessError as e:
        print()
        print("=" * 80)
        print(f"❌ Training thất bại!")
        print("=" * 80)
        sys.exit(e.returncode)
    except KeyboardInterrupt:
        print()
        print("=" * 80)
        print("⚠️  Training bị ngắt!")
        print("=" * 80)
        sys.exit(1)


if __name__ == "__main__":
    # Important for macOS multiprocessing
    import multiprocessing
    multiprocessing.set_start_method('spawn', force=True)
    main()
