import csv
import shutil
import sys
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset, ConcatDataset
from torchvision import transforms
from PIL import Image
import os
from pytorch_lightning import LightningDataModule


class RoverImageDataset(Dataset):
    """
    Custom Dataset for loading rover images and their corresponding labels.

    Args:
        image_path (str): Directory with all the images.
        label_dir (str): Directory with labels.
        transform (callable, optional): Optional transform to be applied on an image sample.
    """

    def __init__(self, image_path, label_dir, seq_len=5, stride=None, channel='L', transform=None, skip=False):
        self.image_path = os.path.join(image_path, 'image_path.csv')
        self.labels_file = os.path.join(label_dir, 'labels.csv')
        self.seq_len = seq_len
        self.stride = stride if stride is not None else seq_len
        self.channel = channel
        self.transform = transform
        self.image_file = self.load_image_paths()
        self.labels = self.load_labels()
        self.skip = skip

        # Debugging statement
        print(f"Loaded {len(self.image_file)} images and {len(self.labels)} labels with dim {len(self.labels[0])}.")

        # Initialize the CSV file with headers if it doesn't exist
        with open("logs/dataset_access.csv", "w", newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["Index", "Start Index", "Image Files", "Labels"])

    def load_image_paths(self):
        """
        Load image paths from the CSV file.

        Returns:
            list: List of image paths.
        """
        df = pd.read_csv(self.image_path)
        path_list = df['image_path'].tolist()
        return path_list

    def load_labels(self):
        """
        Load labels from the CSV file.

        Returns:
            list: List of labels.
        """
        df = pd.read_csv(self.labels_file)
        labels = df.values.astype(float).tolist()
        return labels

    def __len__(self):
        """
        Returns the total number of samples in the dataset.

        Returns:
            int: Number of samples.
        """
        return (len(self.image_file) - self.seq_len) // self.stride + 1  # Ensure the length accounts for sequence length to avoid out of range errors

    def __getitem__(self, idx):
        """
        Get an item from the dataset at the specified index.

        Args:
            idx (int): Index of the sample to be fetched.

        Returns:
            tuple: (image, label) where image is the transformed image and label is the corresponding label.
        """
        start_idx = idx * self.stride
        images = []
        labels = []

        # Ensure the sequence does not go out of bounds
        if (start_idx + self.seq_len > len(self.image_file)) and self.skip:
            raise IndexError("Index out of range for the dataset")

        for i in range(start_idx, start_idx + self.seq_len):
            if i < len(self.image_file):
                img_name = self.image_file[i]
                image = Image.open(img_name).convert(self.channel)

                if self.transform is not None:
                    image = self.transform(image)

                images.append(image)
                labels.append(torch.tensor(self.labels[i], dtype=torch.float32))
            else:
                # Pad with zeros or any other padding value
                images.append(torch.zeros_like(images[0]))
                labels.append(torch.tensor(0, dtype=torch.float32))

        images = torch.stack(images)  # Stack images to create a sequence
        labels = torch.stack(labels)  # Use the label of the last image in the sequence

        # Log the accessed index and details to a CSV file
        with open("logs/dataset_access.csv", "a", newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow([idx, start_idx, self.image_file[start_idx:start_idx + self.seq_len], self.labels[start_idx:start_idx + self.seq_len]])

        return images, labels


class RoverDataModule(LightningDataModule):
    """
    PyTorch Lightning DataModule for handling data loading.

    Args:
        data_dir (str): Directory with all the images.
        label_dir (str): Directory with labels.
        size (callable, optional): Optional size for transform.
        batch_size (int, optional): Batch size for data loading.
    """

    def __init__(self, data_dir: str, label_dir: str, seq_len: int = 5, stride: int = None, size=(1, 256, 256), channel: str = 'L', augment=False, batch_size: int = 32, num_workers: int = 0, split_by_traverse: bool = True, val_traverse: int = None, skip=False):
        super().__init__()
        self.test = None
        self.val = None
        self.train = None
        self.data_dir = data_dir
        self.label_dir = label_dir
        self.seq_len = seq_len
        self.stride = stride if stride is not None else seq_len
        self.size = size
        self.channel = channel
        self.augment = augment
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train_datasets = []
        self.val_datasets = []
        self.test_datasets = []
        self.split_by_traverse = split_by_traverse
        self.data_list = os.listdir(data_dir)
        self.label_list = os.listdir(label_dir)
        self.max_data_range = len(self.data_list)
        if val_traverse is None or val_traverse > self.max_data_range or val_traverse <= 0:
            self.val_traverse = self.max_data_range
        else:
            self.val_traverse = val_traverse
        self.skip=skip

        self.transform = transforms.Compose([
            transforms.Resize((self.size[1], self.size[2])),
            transforms.ToTensor()
        ])

        self.augmentation = transforms.Compose([
            transforms.Resize((self.size[1], self.size[2])),
            transforms.RandomApply(self.random_augmentation(), p=0.5),  # Apply random augmentation with 50% probability
            transforms.ToTensor()
        ])

    def random_augmentation(self):
        # Define a list of augmentation transforms, each with its own probability
        augmentation_transforms = [
            transforms.RandomApply([transforms.RandomRotation(degrees=15)], p=0.5),  # Small rotations to simulate camera angle changes
            transforms.RandomApply([transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1)], p=0.5),  # Lighting changes
            transforms.RandomApply([transforms.RandomResizedCrop((self.size[1], self.size[2]), scale=(0.8, 1.0))], p=0.5),  # Crop and resize to simulate distance changes
            transforms.RandomApply([transforms.GaussianBlur(kernel_size=5, sigma=(0.1, 2.0))], p=0.5),  # Blur to simulate focus issues
        ]

        return augmentation_transforms

    def create_datasets(self, stage):
        """
        Create datasets for training, validation, and testing based on the stage.

        Args:
            stage (str): Current stage - 'fit' or 'test'.

        Returns:
            tuple: Datasets for training, validation, and testing.
            float: Probability of augmentation
        """
        if stage == 'fit':
            train_transform = self.augmentation if self.augment else self.transform
            for i in range(0, self.max_data_range):
                if self.split_by_traverse:
                    try:
                        if i % self.val_traverse != 0:
                            train_dataset = RoverImageDataset(os.path.join(self.data_dir, str(self.data_list[i])), os.path.join(self.label_dir, str(self.label_list[i])), seq_len=self.seq_len, stride=self.stride, channel=self.channel, transform=train_transform, skip=self.skip)
                            self.train_datasets.append(train_dataset)
                            print(f"Dataset traverse {i} for training.")
                        else:
                            val_dataset = RoverImageDataset(os.path.join(self.data_dir, str(self.data_list[i])), os.path.join(self.label_dir, str(self.label_list[i])), seq_len=self.seq_len, stride=self.stride, channel=self.channel, transform=self.transform, skip=selfskip)
                            self.val_datasets.append(val_dataset)
                            print(f"Dataset traverse {i} for validation.")
                    except Exception as e:
                        print(f"Error loading dataset train {i}, val {i}: {e}")
                else:
                    try:
                        train_dataset = RoverImageDataset(os.path.join(self.data_dir, str(self.data_list[i]), 'train'), os.path.join(self.label_dir, str(self.label_list[i]), 'train'), seq_len=self.seq_len, stride=self.stride, channel=self.channel, transform=train_transform, skip=self.skip)
                        self.train_datasets.append(train_dataset)
                        val_dataset = RoverImageDataset(os.path.join(self.data_dir, str(self.data_list[i]), 'val'), os.path.join(self.label_dir, str(self.label_list[i]), 'val'), seq_len=self.seq_len, stride=self.stride, channel=self.channel, transform=self.transform, skip=self.skip)
                        self.val_datasets.append(val_dataset)
                        print(f"Dataset train {i}, val {i} loaded successfully.")
                    except Exception as e:
                        print(f"Error loading dataset train {i}, val {i}: {e}")

        else:
            print('\nOne or more dataset where not initialized correctly.')
            exit()

    def setup(self, stage: str):
        """
        Set up the datasets based on the stage.

        Args:
            stage (str): Current stage - 'fit' or 'test'.
        """
        self.create_datasets(stage)

        if stage == "fit":
            self.train = ConcatDataset(self.train_datasets)
            self.val = ConcatDataset(self.val_datasets)
            print(f"Loaded {len(self.train)} and {len(self.val)} samples for training and validation.")
            print("Training and validation datasets concatenated successfully.")
        elif stage == "test":
            self.test = ConcatDataset(self.test_datasets)
            print("Test datasets concatenated successfully.")

    def train_dataloader(self):
        """
        Create the training DataLoader.

        Returns:
            DataLoader: DataLoader for the training set.
        """
        return DataLoader(self.train, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers, persistent_workers=True)

    def val_dataloader(self):
        """
        Create the validation DataLoader.

        Returns:
            DataLoader: DataLoader for the validation set.
        """
        return DataLoader(self.val, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, persistent_workers=True)

    def test_dataloader(self):
        """
        Create the test DataLoader.

        Returns:
            DataLoader: DataLoader for the test set.
        """
        return DataLoader(self.test, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, persistent_workers=True)
