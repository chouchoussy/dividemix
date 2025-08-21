import os
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

def get_transform(image_size, is_train=True):
    padding = int(image_size * 0.125)
    if is_train:
        return transforms.Compose([
            transforms.RandomCrop(image_size, padding=padding),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])
    else:
        return transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])

class BaseImageDataset(Dataset):
    def __init__(self, csv_path, image_dir, data_column, label_column, transform):
        self.df = pd.read_csv(csv_path)
        self.image_dir = image_dir
        self.data_column = data_column
        self.label_column = label_column
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.image_dir, str(row[self.data_column]))
        img = Image.open(img_path).convert('RGB')
        img = self.transform(img)
        label = int(row[self.label_column])
        return img, label, idx

class dataset_tuned(Dataset):
    def __init__(self, mode, train_csv_path=None, train_feather_path=None, train_data_column=None, train_label_column=None, train_image_dir=None,
                 test_csv_path=None, test_data_column=None, test_label_column=None, test_image_dir=None,
                 pred=None, probability=None, transform=None):
        self.mode = mode
        self.transform = transform
        self.pred = pred
        self.probability = probability
        if mode == 'test':
            self.base = BaseImageDataset(test_csv_path, test_image_dir, test_data_column, test_label_column, transform)
        else:
            self.df = pd.read_csv(train_csv_path)
            self.noise_label = pd.read_feather(train_feather_path)['label'].values
            self.image_dir = train_image_dir
            self.data_column = train_data_column
            self.label_column = train_label_column
            if mode == 'all':
                self.indices = np.arange(len(self.df))
            elif mode == 'labeled':
                self.indices = np.where(self.pred)[0]
                self.probability = [self.probability[i] for i in self.indices]
            elif mode == 'unlabeled':
                self.indices = np.where(~self.pred)[0]
            else:
                raise ValueError('Unknown mode')

    def __len__(self):
        if self.mode == 'test':
            return len(self.base)
        return len(self.indices)

    def __getitem__(self, idx):
        if self.mode == 'test':
            return self.base[idx]
        real_idx = self.indices[idx]
        row = self.df.iloc[real_idx]
        img_path = os.path.join(self.image_dir, str(row[self.data_column]))
        img = Image.open(img_path).convert('RGB')
        img1 = self.transform(img)
        img2 = self.transform(img)
        if self.mode == 'labeled':
            label = int(row[self.label_column])
            noisy_label = int(self.noise_label[real_idx])
            prob = self.probability[idx]
            return img1, img2, noisy_label, prob
        elif self.mode == 'unlabeled':
            return img1, img2
        elif self.mode == 'all':
            label = int(row[self.label_column])
            noisy_label = int(self.noise_label[real_idx])
            return img1, noisy_label, real_idx

class dataloader_tuned:
    def __init__(self, batch_size, num_workers, image_size,
                 train_csv_path, train_feather_path, train_data_column, train_label_column, train_image_dir,
                 test_csv_path, test_data_column, test_label_column, test_image_dir):
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.image_size = image_size
        self.train_csv_path = train_csv_path
        self.train_feather_path = train_feather_path
        self.train_data_column = train_data_column
        self.train_label_column = train_label_column
        self.train_image_dir = train_image_dir
        self.test_csv_path = test_csv_path
        self.test_data_column = test_data_column
        self.test_label_column = test_label_column
        self.test_image_dir = test_image_dir
        self.transform_train = get_transform(image_size, is_train=True)
        self.transform_test = get_transform(image_size, is_train=False)

    def run(self, mode, pred=None, prob=None):
        if mode == 'warmup':
            dataset = dataset_tuned(
                mode='all',
                train_csv_path=self.train_csv_path,
                train_feather_path=self.train_feather_path,
                train_data_column=self.train_data_column,
                train_label_column=self.train_label_column,
                train_image_dir=self.train_image_dir,
                transform=self.transform_train
            )
            loader = DataLoader(dataset, batch_size=self.batch_size*2, shuffle=True, num_workers=self.num_workers)
            return loader
        elif mode == 'train':
            labeled_dataset = dataset_tuned(
                mode='labeled',
                train_csv_path=self.train_csv_path,
                train_feather_path=self.train_feather_path,
                train_data_column=self.train_data_column,
                train_label_column=self.train_label_column,
                train_image_dir=self.train_image_dir,
                pred=pred,
                probability=prob,
                transform=self.transform_train
            )
            labeled_loader = DataLoader(labeled_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)
            unlabeled_dataset = dataset_tuned(
                mode='unlabeled',
                train_csv_path=self.train_csv_path,
                train_feather_path=self.train_feather_path,
                train_data_column=self.train_data_column,
                train_label_column=self.train_label_column,
                train_image_dir=self.train_image_dir,
                pred=pred,
                transform=self.transform_train
            )
            unlabeled_loader = DataLoader(unlabeled_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)
            return labeled_loader, unlabeled_loader
        elif mode == 'test':
            test_dataset = dataset_tuned(
                mode='test',
                test_csv_path=self.test_csv_path,
                test_data_column=self.test_data_column,
                test_label_column=self.test_label_column,
                test_image_dir=self.test_image_dir,
                transform=self.transform_test
            )
            test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)
            return test_loader
        elif mode == 'eval_train':
            eval_dataset = dataset_tuned(
                mode='all',
                train_csv_path=self.train_csv_path,
                train_feather_path=self.train_feather_path,
                train_data_column=self.train_data_column,
                train_label_column=self.train_label_column,
                train_image_dir=self.train_image_dir,
                transform=self.transform_test
            )
            eval_loader = DataLoader(eval_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)
            return eval_loader
        else:
            raise ValueError('Unknown mode')
