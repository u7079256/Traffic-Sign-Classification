#!/usr/bin/env python
# coding: utf-8

"""
Traffic Sign Dataset Handling
-------------------------------------------------
Classes and utilities for loading, processing, and preparing traffic sign datasets
"""

import pickle
import numpy as np
import random
import os
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from PIL import Image
import json
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


def set_seed(seed=42):
    """
    Set random seed to ensure experiment reproducibility
    
    Args:
        seed (int): Random seed value
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    if hasattr(torch, 'backends') and hasattr(torch.backends, 'cudnn'):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    print(f"Random seed set: {seed}")


class TrafficSignDataset(Dataset):
    """
    Traffic Sign Dataset Class
    """
    def __init__(self, features, labels, transform=None, class_mapping=None):
        """
        Initialize dataset
        
        Args:
            features (numpy.ndarray): Image data
            labels (numpy.ndarray): Labels
            transform (callable, optional): Data transformer
            class_mapping (dict, optional): Class label mapping
        """
        self.features = features
        self.labels = labels
        self.transform = transform
        

        self.class_names = {
            0: 'Stop',
            1: 'Turn right',
            2: 'Turn left',
            3: 'Ahead only',
            4: 'Roundabout mandatory'
        }
        
        # Update class mapping (if provided)
        if class_mapping:
            self.class_names.update(class_mapping)
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        image = self.features[idx]
        label = self.labels[idx]
        
        # If image is not RGB format, convert to RGB (3 channels)
        if len(image.shape) == 2:  # Grayscale image
            image = np.stack([image] * 3, axis=2)
        
        # Convert to PIL image for applying transformations
        image = Image.fromarray(image.astype(np.uint8))
        
        # Apply preprocessing transformations
        if self.transform:
            image = self.transform(image)
        
        return image, label
    
    def get_class_name(self, label):
        """
        Get class name
        
        Args:
            label (int): Class label
            
        Returns:
            str: Class name
        """
        return self.class_names.get(label, f"Unknown ({label})")


class TrafficSignProcessor:
    """
    Traffic Sign Data Processor: Responsible for data loading, filtering, and processing
    """
    def __init__(self, config=None):
        """
        Initialize data processor
        
        Args:
            config (dict, optional): Configuration parameters
        """
        # Default configuration
        self.config = {
            'balance_data': True,
            'max_samples_per_class': 300,
            # TODO: Verify the class IDs below are correct.
            # You need to analyze the dataset to identify the correct class IDs for the 5 traffic sign types
            'valid_classes': [14, 33, 34, 35, 40],  # Original class IDs in the dataset
            'class_mapping': ,  # Mapping to new class indices
            # TODO: Verify the image size
            'img_size': ,  # Input image size
            'mean': [0.485, 0.456, 0.406],  # ImageNet mean
            'std': [0.229, 0.224, 0.225]    # ImageNet std
        }
        
        # Update configuration (if provided)
        if config:
            self.config.update(config)
        
        # Class names
        self.class_names = {
            0: 'Stop',
            1: 'Turn right',
            2: 'Turn left',
            3: 'Ahead only',
            4: 'Roundabout mandatory'
        }
        
        # Store loaded data
        self.X_train = None
        self.y_train = None
        self.X_valid = None
        self.y_valid = None
        self.X_test = None
        self.y_test = None
        
    def load_data(self, training_file, validation_file=None, testing_file=None, balance_data=None, max_samples_per_class=None):
        """
        Load and preprocess traffic sign dataset
        
        Args:
            training_file (str): Path to the training data pickle file
            validation_file (str, optional): Path to the validation data pickle file
            testing_file (str, optional): Path to the testing data pickle file
            balance_data (bool, optional): Whether to balance training data
            max_samples_per_class (int, optional): Maximum samples per class when balancing
            
        Returns:
            tuple: Preprocessed training, validation and testing data and labels
        """
        if balance_data is None:
            balance_data = self.config['balance_data']
        if max_samples_per_class is None:
            max_samples_per_class = self.config['max_samples_per_class']
        
        print(f"Loading data files...")
        
        # Load training data
        try:
            with open(training_file, mode='rb') as f:
                train = pickle.load(f)
            X_train_raw, y_train_raw = train['features'], train['labels']
            
            # If validation and test files are not provided, automatically split the training set
            if validation_file is None or testing_file is None:
                # First split out the test set
                X_train_temp, X_test_raw, y_train_temp, y_test_raw = train_test_split(
                    X_train_raw, y_train_raw, test_size=0.2, random_state=42, stratify=y_train_raw)
                
                # Split out validation set from remaining data
                X_train_raw, X_valid_raw, y_train_raw, y_valid_raw = train_test_split(
                    X_train_temp, y_train_temp, test_size=0.25, random_state=42, stratify=y_train_temp)
                
                print(f"Data automatically split into training set ({len(X_train_raw)} samples), validation set ({len(X_valid_raw)} samples) and test set ({len(X_test_raw)} samples)")
            
            # If validation and test files are provided, load them
            else:
                if validation_file:
                    with open(validation_file, mode='rb') as f:
                        valid = pickle.load(f)
                    X_valid_raw, y_valid_raw = valid['features'], valid['labels']
                
                if testing_file:
                    with open(testing_file, mode='rb') as f:
                        test = pickle.load(f)
                    X_test_raw, y_test_raw = test['features'], test['labels']
        
        except Exception as e:
            print(f"Error loading data: {e}")
            return None, None, None, None, None, None
            
        print(f"Raw data loaded. Training: {X_train_raw.shape}, Validation: {X_valid_raw.shape if 'X_valid_raw' in locals() else 'None'}, Testing: {X_test_raw.shape if 'X_test_raw' in locals() else 'None'}")
        
        # Filter and map data to required classes
        valid_classes = self.config['valid_classes']
        class_mapping = self.config['class_mapping']
        
        X_train, y_train = self._filter_and_map_data(X_train_raw, y_train_raw, balance=balance_data, max_count=max_samples_per_class)
        
        X_valid, y_valid = None, None
        if 'X_valid_raw' in locals() and X_valid_raw is not None:
            X_valid, y_valid = self._filter_and_map_data(X_valid_raw, y_valid_raw)
            
        X_test, y_test = None, None
        if 'X_test_raw' in locals() and X_test_raw is not None:
            X_test, y_test = self._filter_and_map_data(X_test_raw, y_test_raw)
        
        # Store the processed data
        self.X_train, self.y_train = X_train, y_train
        self.X_valid, self.y_valid = X_valid, y_valid
        self.X_test, self.y_test = X_test, y_test
        
        # Print statistics
        print(f"Processed data. Training: {X_train.shape}, Validation: {X_valid.shape if X_valid is not None else 'None'}, Testing: {X_test.shape if X_test is not None else 'None'}")
        print(f"Class distribution (training set):")
        unique, counts = np.unique(y_train, return_counts=True)
        for i, (cls, cnt) in enumerate(zip(unique, counts)):
            print(f"  Class {cls} ({self.class_names[cls]}): {cnt} samples")
        
        return X_train, y_train, X_valid, y_valid, X_test, y_test
        
    def _filter_and_map_data(self, features, labels, balance=False, max_count=None):
        """
        Filter and map data to required classes
        
        Args:
            features (numpy.ndarray): Image features
            labels (numpy.ndarray): Labels
            balance (bool): Whether to balance classes
            max_count (int): Maximum samples per class
            
        Returns:
            tuple: Filtered features and labels
        """
        valid_classes = self.config['valid_classes']
        class_mapping = self.config['class_mapping']
        
        filtered_features = []
        filtered_labels = []
        
        # Create class indices for balancing
        class_indices = {}
        for cls in valid_classes:
            class_indices[cls] = np.where(labels == cls)[0]
            print(f"Class {cls}: {len(class_indices[cls])} samples")
        
        if balance:
            # Balance classes by taking up to max_count samples from each class
            for cls in valid_classes:
                indices = class_indices[cls]
                if max_count and len(indices) > max_count:
                    indices = np.random.choice(indices, max_count, replace=False)
                
                for idx in indices:
                    filtered_features.append(features[idx])
                    filtered_labels.append(class_mapping[labels[idx]])
        else:
            # Not balancing, just filter
            for idx in range(len(labels)):
                if labels[idx] in valid_classes:
                    filtered_features.append(features[idx])
                    filtered_labels.append(class_mapping[labels[idx]])
        
        return np.array(filtered_features), np.array(filtered_labels)
    
    def create_datasets(self, augment_train=True):
        """
        Create PyTorch dataset objects
        
        Args:
            augment_train (bool): Whether to apply data augmentation to training data
            
        Returns:
            tuple: Training, validation and test dataset objects
        """
        # Check if data is loaded
        if self.X_train is None or self.y_train is None:
            print("Error: Data not yet loaded. Please call load_data() method first.")
            return None, None, None
        
        print("Creating datasets...")
        img_size = self.config['img_size']
        mean = self.config['mean']
        std = self.config['std']
        
        # TODO: Implement data augmentation methods
        # Apply data augmentation to training data to improve model generalization
        # Note: Validation and test sets should not use data augmentation
        if augment_train:
            # Apply data augmentation to training data
            train_transform = transforms.Compose([
                # Basic transformations
                transforms.Resize((img_size, img_size)),
                
                # TODO: Add appropriate data augmentation methods, such as:
                # - Random rotation (recommended range: small angles, e.g., within 15 degrees)
                # - Random translation
                # - Random adjustment of brightness, contrast, etc.
                # - Other augmentation methods you consider appropriate
                
                # Convert to tensor and normalize
                transforms.ToTensor(),
                transforms.Normalize(mean, std)
            ])
            print("Training data augmentation enabled")
        else:
            # No augmentation
            train_transform = transforms.Compose([
                transforms.Resize((img_size, img_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean, std)
            ])
            print("Training data augmentation disabled")
        
        # For validation and test data, only basic processing
        eval_transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
        
        # Create datasets
        train_dataset = TrafficSignDataset(
            self.X_train, self.y_train, 
            transform=train_transform,
            class_mapping=self.class_names
        )
        
        valid_dataset = TrafficSignDataset(
            self.X_valid, self.y_valid, 
            transform=eval_transform,
            class_mapping=self.class_names
        ) if self.X_valid is not None else None
        
        test_dataset = TrafficSignDataset(
            self.X_test, self.y_test, 
            transform=eval_transform,
            class_mapping=self.class_names
        ) if self.X_test is not None else None
        
        print(f"Datasets created: {len(train_dataset)} training samples")
        if valid_dataset:
            print(f"{len(valid_dataset)} validation samples")
        if test_dataset:
            print(f"{len(test_dataset)} test samples")
        
        return train_dataset, valid_dataset, test_dataset
    
    def create_data_loaders(self, train_dataset, valid_dataset=None, test_dataset=None, batch_size=32, num_workers=4):
        """
        Create PyTorch data loaders
        
        Args:
            train_dataset, valid_dataset, test_dataset: Dataset objects
            batch_size (int): Batch size
            num_workers (int): Number of worker processes for data loading
            
        Returns:
            tuple: Training, validation and test data loaders
        """
        print(f"Creating data loaders (batch_size={batch_size}, workers={num_workers})...")
        
        train_loader = DataLoader(
            train_dataset, 
            batch_size=batch_size, 
            shuffle=True, 
            num_workers=num_workers,
            pin_memory=True if torch.cuda.is_available() else False
        )
        
        valid_loader = None
        if valid_dataset:
            valid_loader = DataLoader(
                valid_dataset, 
                batch_size=batch_size, 
                shuffle=False, 
                num_workers=num_workers,
                pin_memory=True if torch.cuda.is_available() else False
            )
        
        test_loader = None
        if test_dataset:
            test_loader = DataLoader(
                test_dataset, 
                batch_size=batch_size, 
                shuffle=False, 
                num_workers=num_workers,
                pin_memory=True if torch.cuda.is_available() else False
            )
        
        return train_loader, valid_loader, test_loader
    
    def visualize_samples(self, num_samples=5):
        """
        Basic data visualization example
        
        Args:
            num_samples: Number of samples to display for each class
        """
        if self.X_train is None or self.y_train is None:
            print("Error: Data not yet loaded. Please call load_data() method first.")
            return
        
        # Display num_samples samples for each class
        plt.figure(figsize=(15, 10))
        for class_idx in range(5):  # 5 classes
            # Find all samples of this class
            indices = np.where(self.y_train == class_idx)[0]
            
            # Select samples to display
            display_indices = indices[:min(num_samples, len(indices))]
            
            for i, idx in enumerate(display_indices):
                plt.subplot(5, num_samples, class_idx * num_samples + i + 1)
                img = self.X_train[idx]
                plt.imshow(img)
                plt.title(f"Class: {self.class_names[class_idx]}")
                plt.axis('off')
        
        plt.tight_layout()
        plt.show()
        