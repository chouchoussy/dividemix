"""
Dataloader for DivideMix with pre-computed embeddings
Reads embeddings from .feather files instead of loading images
"""
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader


class BaseEmbeddingDataset(Dataset):
    """Dataset for test set with embeddings"""
    def __init__(self, csv_path, embedding_feather_path, label_column):
        self.df = pd.read_csv(csv_path)
        self.embeddings = pd.read_feather(embedding_feather_path).values
        self.label_column = label_column
        
        # Verify dimensions match
        assert len(self.df) == len(self.embeddings), \
            f"CSV rows ({len(self.df)}) != embedding rows ({len(self.embeddings)})"
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        embedding = torch.tensor(self.embeddings[idx], dtype=torch.float32)
        label = int(self.df.iloc[idx][self.label_column])
        return embedding, label


class dataset_tuned_mlp(Dataset):
    """Dataset for training with embeddings and noisy labels"""
    def __init__(self, mode, 
                 train_csv_path=None, 
                 train_embedding_feather_path=None,
                 train_noisy_label_feather_path=None, 
                 train_label_column=None,
                 test_csv_path=None, 
                 test_embedding_feather_path=None,
                 test_label_column=None,
                 pred=None, 
                 probability=None):
        """
        Args:
            mode: 'test', 'all', 'labeled', 'unlabeled'
            train_csv_path: path to training CSV with true labels
            train_embedding_feather_path: path to training embeddings .feather
            train_noisy_label_feather_path: path to noisy labels .feather
            train_label_column: column name for true labels in CSV
            test_csv_path: path to test CSV
            test_embedding_feather_path: path to test embeddings .feather
            test_label_column: column name for labels in test CSV
            pred: boolean array indicating clean samples (for labeled/unlabeled split)
            probability: clean probability for each sample
        """
        self.mode = mode
        self.pred = pred
        self.probability = probability
        
        if mode == 'test':
            # Test mode: load test embeddings and labels
            self.base = BaseEmbeddingDataset(
                test_csv_path, 
                test_embedding_feather_path, 
                test_label_column
            )
        else:
            # Training modes: load train embeddings, true labels, and noisy labels
            self.df = pd.read_csv(train_csv_path)
            self.embeddings = pd.read_feather(train_embedding_feather_path).values
            self.noise_label = pd.read_feather(train_noisy_label_feather_path)['label'].values
            self.label_column = train_label_column
            
            # Verify dimensions
            assert len(self.df) == len(self.embeddings), \
                f"CSV rows ({len(self.df)}) != embedding rows ({len(self.embeddings)})"
            assert len(self.df) == len(self.noise_label), \
                f"CSV rows ({len(self.df)}) != noisy label rows ({len(self.noise_label)})"
            
            # Set up indices based on mode
            if mode == 'all':
                self.indices = np.arange(len(self.df))
            elif mode == 'labeled':
                self.indices = np.where(self.pred)[0]
                self.probability = [self.probability[i] for i in self.indices]
            elif mode == 'unlabeled':
                self.indices = np.where(~self.pred)[0]
            else:
                raise ValueError(f'Unknown mode: {mode}')
    
    def __len__(self):
        if self.mode == 'test':
            return len(self.base)
        return len(self.indices)
    
    def __getitem__(self, idx):
        if self.mode == 'test':
            return self.base[idx]
        
        real_idx = self.indices[idx]
        embedding = torch.tensor(self.embeddings[real_idx], dtype=torch.float32)
        
        # For MLP, we don't need two augmented views, but keep the interface
        # compatible with DivideMix by returning the same embedding twice
        emb1 = embedding
        emb2 = embedding.clone()
        
        if self.mode == 'labeled':
            label = int(self.df.iloc[real_idx][self.label_column])
            noisy_label = int(self.noise_label[real_idx])
            prob = self.probability[idx]
            return emb1, emb2, noisy_label, prob
        elif self.mode == 'unlabeled':
            return emb1, emb2
        elif self.mode == 'all':
            label = int(self.df.iloc[real_idx][self.label_column])
            noisy_label = int(self.noise_label[real_idx])
            return emb1, noisy_label, real_idx


class dataloader_tuned_mlp:
    """DataLoader factory for MLP with embeddings"""
    def __init__(self, batch_size, num_workers,
                 train_csv_path, 
                 train_embedding_feather_path,
                 train_noisy_label_feather_path,
                 train_label_column,
                 test_csv_path, 
                 test_embedding_feather_path,
                 test_label_column):
        """
        Args:
            batch_size: batch size for training
            num_workers: number of workers for data loading
            train_csv_path: path to training CSV with true labels
            train_embedding_feather_path: path to training embeddings .feather
            train_noisy_label_feather_path: path to noisy labels .feather
            train_label_column: column name for true labels in training CSV
            test_csv_path: path to test CSV
            test_embedding_feather_path: path to test embeddings .feather
            test_label_column: column name for labels in test CSV
        """
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train_csv_path = train_csv_path
        self.train_embedding_feather_path = train_embedding_feather_path
        self.train_noisy_label_feather_path = train_noisy_label_feather_path
        self.train_label_column = train_label_column
        self.test_csv_path = test_csv_path
        self.test_embedding_feather_path = test_embedding_feather_path
        self.test_label_column = test_label_column
    
    def run(self, mode, pred=None, prob=None):
        """
        Create dataloader for specified mode
        Args:
            mode: 'warmup', 'train', 'test', 'eval_train'
            pred: boolean array for clean/noisy split
            prob: clean probability array
        Returns:
            DataLoader or tuple of DataLoaders (for 'train' mode)
        """
        if mode == 'warmup':
            dataset = dataset_tuned_mlp(
                mode='all',
                train_csv_path=self.train_csv_path,
                train_embedding_feather_path=self.train_embedding_feather_path,
                train_noisy_label_feather_path=self.train_noisy_label_feather_path,
                train_label_column=self.train_label_column
            )
            loader = DataLoader(
                dataset, 
                batch_size=self.batch_size*2, 
                shuffle=True, 
                num_workers=self.num_workers
            )
            return loader
        
        elif mode == 'train':
            labeled_dataset = dataset_tuned_mlp(
                mode='labeled',
                train_csv_path=self.train_csv_path,
                train_embedding_feather_path=self.train_embedding_feather_path,
                train_noisy_label_feather_path=self.train_noisy_label_feather_path,
                train_label_column=self.train_label_column,
                pred=pred,
                probability=prob
            )
            labeled_loader = DataLoader(
                labeled_dataset, 
                batch_size=self.batch_size, 
                shuffle=True, 
                num_workers=self.num_workers
            )
            
            unlabeled_dataset = dataset_tuned_mlp(
                mode='unlabeled',
                train_csv_path=self.train_csv_path,
                train_embedding_feather_path=self.train_embedding_feather_path,
                train_noisy_label_feather_path=self.train_noisy_label_feather_path,
                train_label_column=self.train_label_column,
                pred=pred
            )
            unlabeled_loader = DataLoader(
                unlabeled_dataset, 
                batch_size=self.batch_size, 
                shuffle=True, 
                num_workers=self.num_workers
            )
            return labeled_loader, unlabeled_loader
        
        elif mode == 'test':
            test_dataset = dataset_tuned_mlp(
                mode='test',
                test_csv_path=self.test_csv_path,
                test_embedding_feather_path=self.test_embedding_feather_path,
                test_label_column=self.test_label_column
            )
            test_loader = DataLoader(
                test_dataset, 
                batch_size=self.batch_size, 
                shuffle=False, 
                num_workers=self.num_workers
            )
            return test_loader
        
        elif mode == 'eval_train':
            eval_dataset = dataset_tuned_mlp(
                mode='all',
                train_csv_path=self.train_csv_path,
                train_embedding_feather_path=self.train_embedding_feather_path,
                train_noisy_label_feather_path=self.train_noisy_label_feather_path,
                train_label_column=self.train_label_column
            )
            eval_loader = DataLoader(
                eval_dataset, 
                batch_size=self.batch_size, 
                shuffle=False, 
                num_workers=self.num_workers
            )
            return eval_loader
        
        else:
            raise ValueError(f'Unknown mode: {mode}')
