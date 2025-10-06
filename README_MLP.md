# DivideMix với MLP và Pre-computed Embeddings

## Tổng quan

Đây là phiên bản mở rộng của DivideMix cho phép huấn luyện trên **pre-computed embeddings** thay vì ảnh raw, sử dụng mô hình **MLP đơn giản** thay vì ResNet.

### Kiến trúc MLP

```
Input (embedding_dim) 
  ↓
Linear(in_dim → 512) → BatchNorm1d → ReLU → Dropout(0.3)
  ↓
Linear(512 → 256) → BatchNorm1d → ReLU → Dropout(0.3)
  ↓
Linear(256 → num_classes)
  ↓
Output (logits)
```

## Files mới

1. **`MLP.py`**: Định nghĩa kiến trúc MLP
2. **`dataloader_tuned_mlp.py`**: DataLoader đọc embeddings từ file `.feather`
3. **`train_tuned_mlp.py`**: Script huấn luyện DivideMix với MLP
4. **`test-tuned-mlp.py`**: Script tạo lệnh chạy mẫu

## Cài đặt

Giống như pipeline gốc:

```bash
cd /Users/linhnh/Documents/DivideMix-tuned
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Cách chạy

### 1. Chuẩn bị dữ liệu

Bạn cần 3 loại file:

- **CSV train**: chứa true labels (cột `label`)
- **Feather embeddings train**: embeddings đã tính sẵn (ví dụ: `*_auto.feather`)
- **Feather noisy labels train**: nhãn nhiễu (ví dụ: `*_LLM.feather`, `*_semi_supervise.feather`)
- **CSV test**: chứa labels
- **Feather embeddings test**: embeddings test set

### 2. Chạy huấn luyện

#### Ví dụ Fashion-MNIST với CLIP embeddings (dim=512):

```bash
python /Users/linhnh/Documents/DivideMix-tuned/train_tuned_mlp.py \
  --dataset fashion-mnist \
  --noise_type semi_supervise \
  --batch_size 64 \
  --num_epochs 300 \
  --warm_up 10 \
  --num_class 10 \
  --embedding_dim 512 \
  --num_workers 8 \
  --gpuid 0 \
  --train_csv_path "/Users/linhnh/Documents/DivideMix-tuned/Data/Fashion-MNIST-test/fashion_mnist.csv" \
  --train_embedding_feather_path "/Users/linhnh/Documents/DivideMix-tuned/Data/Fashion-MNIST-test/fashion-mnist-test-clip-b16-noise/fashion-mnist-test_auto.feather" \
  --train_noisy_label_feather_path "/Users/linhnh/Documents/DivideMix-tuned/Data/Fashion-MNIST-test/fashion-mnist-test-clip-b16-noise/fashion-mnist-test_semi_supervise.feather" \
  --train_label_column "label" \
  --test_csv_path "/Users/linhnh/Documents/DivideMix-tuned/Data/fashion-mnist-2k5-testset/fashion-mnist-test-2k5.csv" \
  --test_embedding_feather_path "/Users/linhnh/Documents/DivideMix-tuned/Data/fashion-mnist-2k5-testset/embed/fashion-mnist-2k5-testset_embedding_clip-base-16.feather" \
  --test_label_column "label"
```

#### Ví dụ AG News với BERT embeddings (dim=768):

```bash
python /Users/linhnh/Documents/DivideMix-tuned/train_tuned_mlp.py \
  --dataset agnews \
  --noise_type llm \
  --num_class 4 \
  --embedding_dim 768 \
  --train_csv_path "/Users/linhnh/Documents/DivideMix-tuned/Data/Agnews-12k-train/ag_news_12k.csv" \
  --train_embedding_feather_path "/Users/linhnh/Documents/DivideMix-tuned/Data/Agnews-12k-train/agnews-12k-train-bert-noise/agnews-12k-train_auto.feather" \
  --train_noisy_label_feather_path "/Users/linhnh/Documents/DivideMix-tuned/Data/Agnews-12k-train/agnews-12k-train-bert-noise/agnews-12k-train_LLM.feather" \
  --train_label_column "label" \
  --test_csv_path "/Users/linhnh/Documents/DivideMix-tuned/Data/agnews-3k-testset/agnews-test-3k.csv" \
  --test_embedding_feather_path "/Users/linhnh/Documents/DivideMix-tuned/Data/agnews-3k-testset/embed/agnews-3k-testset_embedding_bert.feather" \
  --test_label_column "label"
```

### 3. Tạo lệnh tự động

Chạy script helper để in ra lệnh mẫu:

```bash
python /Users/linhnh/Documents/DivideMix-tuned/test-tuned-mlp.py
```

## Tham số quan trọng

| Tham số | Mô tả | Mặc định |
|---------|-------|----------|
| `--embedding_dim` | Kích thước embedding (CLIP: 512, BERT: 768) | 512 |
| `--num_class` | Số lượng classes | 10 |
| `--train_embedding_feather_path` | File embeddings train | Required |
| `--train_noisy_label_feather_path` | File noisy labels train | Required |
| `--test_embedding_feather_path` | File embeddings test | Required |
| `--batch_size` | Batch size | 64 |
| `--lr` | Learning rate | 0.02 |
| `--num_epochs` | Số epochs | 300 |
| `--warm_up` | Số epochs warmup | 10 |
| `--p_threshold` | Ngưỡng clean probability | 0.5 |
| `--lambda_u` | Trọng số unsupervised loss | 25 |
| `--T` | Temperature sharpening | 0.5 |
| `--alpha` | Beta distribution parameter | 4 |

## Kết quả

- **Log huấn luyện**: In ra console mỗi epoch
- **Test accuracy**: In ra sau mỗi epoch
- **Predictions**: Lưu vào file `{dataset}_{noise_type}_mlp_test-predictions.npy` ở epoch cuối
- **Metrics**: Accuracy, F1-macro, F1-weighted, Classification report

## So sánh với pipeline gốc

| Aspect | Pipeline gốc (Image) | Pipeline MLP (Embedding) |
|--------|---------------------|--------------------------|
| Input | Ảnh raw (load từ disk) | Embeddings (pre-computed) |
| Model | ResNet18 | MLP (512→256→out) |
| Augmentation | RandomCrop, Flip | Không (embeddings cố định) |
| Training script | `train_tuned.py` | `train_tuned_mlp.py` |
| Dataloader | `dataloader_tuned.py` | `dataloader_tuned_mlp.py` |
| Speed | Chậm hơn (load + transform ảnh) | Nhanh hơn (chỉ load embeddings) |

## Datasets có sẵn

Trong thư mục `Data/` có các bộ dữ liệu với embeddings:

1. **Fashion-MNIST** (CLIP, 10 classes)
2. **AG News** (BERT, 4 classes)
3. **OrganAMNIST** (CLIP, 11 classes)
4. **Resisc45** (CLIP, 45 classes)
5. **Yahoo Answers** (BERT, 10 classes)

## Lưu ý

1. **CUDA required**: Code hiện dùng CUDA. Nếu chạy trên macOS/CPU cần sửa code.
2. **Embedding dimension**: Phải khớp với file embedding:
   - CLIP base-16: 512
   - BERT: 768
   - Kiểm tra: `pd.read_feather('path').shape[1]`
3. **File structure**:
   - Embedding file: ma trận NxD (N samples, D dimensions)
   - Noisy label file: phải có cột `label`
4. **Memory**: Embeddings được load toàn bộ vào RAM, phù hợp với datasets vừa/nhỏ.

## Troubleshooting

**Q: ModuleNotFoundError: No module named 'MLP'**
- A: Đảm bảo chạy từ thư mục gốc của project hoặc thêm vào PYTHONPATH

**Q: Shape mismatch error**
- A: Kiểm tra `--embedding_dim` khớp với dimension thực tế của file embedding

**Q: CUDA error**
- A: Chạy trên máy có GPU NVIDIA hoặc sửa code để dùng CPU/MPS

**Q: File not found**
- A: Dùng đường dẫn tuyệt đối như trong ví dụ

## Citation

Nếu sử dụng code này, vui lòng cite paper gốc DivideMix:

```bibtex
@inproceedings{
    li2020dividemix,
    title={DivideMix: Learning with Noisy Labels as Semi-supervised Learning},
    author={Junnan Li and Richard Socher and Steven C.H. Hoi},
    booktitle={International Conference on Learning Representations},
    year={2020},
}
```

%
%
%
%


cd /path/to/DivideMix-tuned

# Tạo venv (nếu chưa có)
python3 -m venv .venv
source .venv/bin/activate

# Cài đặt PyTorch với CUDA support
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
# Hoặc cu121 tùy CUDA version của bạn

# Cài các thư viện khác
pip install -r requirements.txt

# Chạy trainning 
source .venv/bin/activate
python run_mlp_training_simple.py