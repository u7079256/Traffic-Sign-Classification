#!/usr/bin/env python
# coding: utf-8

"""
Traffic Sign Classification Training Pipeline
-------------------------------------------------
Main script for training the traffic sign classification model with ResNet18
"""

import torch
import torch.nn as nn
import torch.optim as optim
import os
import time
import argparse
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

# Import custom modules
from network import ResNet18
from dataset import TrafficSignProcessor, set_seed


# Device selection
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# Training function
def train(model, train_loader, optimizer, criterion, epoch, epochs):
    """
    Train model for one epoch
    
    Args:
        model (nn.Module): Model to train
        train_loader (DataLoader): Training data loader
        optimizer (Optimizer): Optimizer
        criterion: Loss criterion
        epoch (int): Current epoch
        epochs (int): Total number of epochs
        
    Returns:
        tuple: Average loss and accuracy for the epoch
    """
    model.train()
    train_loss = 0
    correct = 0
    total = 0
    
    # Using tqdm progress bar
    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]")
    
    for batch_idx, (inputs, targets) in enumerate(pbar):
        # Move data to device
        inputs, targets = inputs.to(device), targets.to(device)
        
        # Zero gradients
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        
        # Backward pass and optimization
        loss.backward()
        optimizer.step()
        
        # Statistics
        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        
        # Update progress bar
        pbar.set_postfix({
            'loss': train_loss/(batch_idx+1), 
            'acc': 100.*correct/total
        })
    
    # TODO: Calculate training metrics
    # Calculate average loss and accuracy for the epoch
    
    # Code for calculating metrics has been removed
    # Students should implement their own calculation for avg_loss and acc
    
    return 0.0, 0.0  # Placeholder return values


# Validation function
def validate(model, val_loader, criterion):
    """
    Validate model on validation set
    
    Args:
        model (nn.Module): Model to validate
        val_loader (DataLoader): Validation data loader
        criterion: Loss criterion
        
    Returns:
        tuple: Average loss and accuracy for validation
    """
    model.eval()
    val_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        pbar = tqdm(val_loader, desc="Validating")
        
        for batch_idx, (inputs, targets) in enumerate(pbar):
            # Move data to device
            inputs, targets = inputs.to(device), targets.to(device)
            
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            # Statistics
            val_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
            # Update progress bar
            pbar.set_postfix({
                'loss': val_loss/(batch_idx+1), 
                'acc': 100.*correct/total
            })
    
    # TODO: Calculate validation metrics
    # Calculate average loss and accuracy for validation
    
    # Code for calculating metrics has been removed
    # Students should implement their own calculation for avg_loss and acc
    
    return 0.0, 0.0  # Placeholder return values


# Test model and calculate metrics
def test_model(model, test_loader, criterion, class_names):
    """
    Test model on test set and calculate metrics
    
    Args:
        model (nn.Module): Model to test
        test_loader (DataLoader): Test data loader
        criterion: Loss criterion
        class_names (dict): Mapping from class indices to class names
        
    Returns:
        dict: Test results with accuracy, loss, predictions, targets, and confusion matrix
    """
    model.eval()
    test_loss = 0
    all_targets = []
    all_predictions = []
    
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(tqdm(test_loader, desc="Testing")):
            # Move data to device
            inputs, targets = inputs.to(device), targets.to(device)
            
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            # Statistics
            test_loss += loss.item()
            _, predicted = outputs.max(1)
            
            # Save for metrics calculation
            all_targets.extend(targets.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())
    
    # Convert to numpy arrays
    all_targets = np.array(all_targets)
    all_predictions = np.array(all_predictions)
    
    # TODO: Calculate test metrics
    # Calculate accuracy, confusion matrix, and other evaluation metrics
    
    # Code for calculating metrics has been removed
    # Students should implement their own calculation for accuracy, conf_matrix, etc.
    
    # Return placeholder test results
    return {
        'accuracy': 0.0,
        'loss': 0.0,
        'predictions': all_predictions,
        'targets': all_targets,
        'confusion_matrix': np.zeros((5, 5))  # Placeholder empty confusion matrix
    }


# Save checkpoint
def save_checkpoint(model, optimizer, scheduler, epoch, acc, history, filepath):
    """
    Save model checkpoint
    
    Args:
        model (nn.Module): Model to save
        optimizer (Optimizer): Optimizer state
        scheduler: Learning rate scheduler state
        epoch (int): Current epoch
        acc (float): Validation accuracy
        history (tuple): Training history
        filepath (str): Path to save the checkpoint
    """
    state = {
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict() if scheduler else None,
        'acc': acc,
        'epoch': epoch,
        'train_history': history
    }
    
    torch.save(state, filepath)
    print(f"Checkpoint saved to {filepath}")
    
    # Save an additional best_model.pth for easier loading
    best_filepath = os.path.dirname(filepath) + '/best_model.pth'
    torch.save(state, best_filepath)
    print(f"Best model saved to {best_filepath}")


# Basic visualization function
def plot_confusion_matrix(conf_matrix, class_names):
    """
    Basic confusion matrix visualization
    
    Args:
        conf_matrix: Confusion matrix
        class_names: Class names
    """
    plt.figure(figsize=(10, 8))
    plt.imshow(conf_matrix, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    
    tick_marks = np.arange(len(class_names.values()))
    plt.xticks(tick_marks, class_names.values(), rotation=45)
    plt.yticks(tick_marks, class_names.values())
    
    # Add text labels
    thresh = conf_matrix.max() / 2.0
    for i in range(conf_matrix.shape[0]):
        for j in range(conf_matrix.shape[1]):
            plt.text(j, i, f"{conf_matrix[i, j]}",
                     horizontalalignment="center",
                     color="white" if conf_matrix[i, j] > thresh else "black")
    
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()


# Plot training history
def plot_training_history(train_loss, train_acc, val_loss, val_acc):
    """
    Basic training history visualization
    
    Args:
        train_loss, train_acc: Training loss and accuracy lists
        val_loss, val_acc: Validation loss and accuracy lists
    """
    plt.figure(figsize=(12, 5))
    
    # Plot loss curves
    plt.subplot(1, 2, 1)
    plt.plot(train_loss, label='Train Loss')
    plt.plot(val_loss, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    
    # Plot accuracy curves
    plt.subplot(1, 2, 2)
    plt.plot(train_acc, label='Train Accuracy')
    plt.plot(val_acc, label='Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (%)')
    plt.title('Training and Validation Accuracy')
    plt.legend()
    
    plt.tight_layout()
    plt.show()


# Main function
def main():
    """
    Main function for training the traffic sign classifier
    """
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Traffic Sign ResNet18 Classification')
    parser.add_argument('--lr', default=0.01, type=float, help='Learning rate')
    parser.add_argument('--batch_size', default=64, type=int, help='Batch size')
    parser.add_argument('--epochs', default=100, type=int, help='Number of training epochs')
    parser.add_argument('--resume', '-r', action='store_true', help='Resume from checkpoint')
    parser.add_argument('--optimizer', default='rmsprop', type=str, help='Optimizer (sgd, adam, rmsprop)')
    parser.add_argument('--scheduler', default='cosine', type=str, help='Learning rate scheduler (cosine, step, none)')
    parser.add_argument('--data_path', default='.', type=str, help='Data path')
    parser.add_argument('--workers', default=1, type=int, help='Number of data loading threads')
    parser.add_argument('--weight_decay', default=1e-4, type=float, help='Weight decay factor')
    parser.add_argument('--momentum', default=0.9, type=float, help='Momentum factor (SGD only)')
    parser.add_argument('--step_size', default=30, type=int, help='Step LR step size')
    parser.add_argument('--gamma', default=0.1, type=float, help='Step LR gamma')
    parser.add_argument('--early_stopping', default=15, type=int, help='Early stopping patience')
    args = parser.parse_args()
    
    # Set random seed for reproducibility
    set_seed(42)
    
    # Create checkpoint directory
    os.makedirs('./checkpoint', exist_ok=True)
    
    # Load data
    processor = TrafficSignProcessor()
    
    # Check for data files
    train_file = os.path.join(args.data_path, 'train.p')
    valid_file = os.path.join(args.data_path, 'valid.p')
    test_file = os.path.join(args.data_path, 'test.p')
    
    # Load and process data
    processor.load_data(train_file, valid_file, test_file)
    
    # Create datasets
    train_dataset, valid_dataset, test_dataset = processor.create_datasets(augment_train=True)
    
    # Create data loaders
    train_loader, valid_loader, test_loader = processor.create_data_loaders(
        train_dataset, valid_dataset, test_dataset, 
        batch_size=args.batch_size, num_workers=args.workers
    )
    
    # Initialize model
    model = ResNet18(num_classes=5)  # 5 classes for traffic signs
    model = model.to(device)
    
    # Initialize training state
    start_epoch = 0
    best_acc = 0
    train_loss_list = []
    train_acc_list = []
    val_loss_list = []
    val_acc_list = []
    checkpoint = None
    
    # Resume from checkpoint if specified
    if args.resume:
        print("Resuming from checkpoint...")
        checkpoint_path = './checkpoint/best_model.pth'
        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path)
            
            if 'model' in checkpoint:
                model.load_state_dict(checkpoint['model'])
                best_acc = checkpoint['acc']
                start_epoch = checkpoint['epoch']
                
                # Restore training history
                if 'train_history' in checkpoint:
                    history = checkpoint['train_history']
                    train_loss_list, train_acc_list, val_loss_list, val_acc_list = history
                
                print(f"Resumed from checkpoint Epoch: {start_epoch}, Accuracy: {best_acc:.2f}%")
    
    # Loss function
    criterion = nn.CrossEntropyLoss()
    
    # TODO: Choose appropriate optimizer and learning rate scheduler
    # Set up optimizer
    
    # Code for setting optimizer has been removed
    # Students should implement their own optimizer setup
    optimizer = None
    
    # Learning rate scheduler
    
    # Code for setting scheduler has been removed
    # Students should implement their own scheduler setup
    scheduler = None
    
    # Early stopping variables
    no_improve_epochs = 0
    
    # Training loop
    for epoch in range(start_epoch, args.epochs):
        # Train one epoch
        train_loss, train_acc = train(model, train_loader, optimizer, criterion, epoch, args.epochs)
        train_loss_list.append(train_loss)
        train_acc_list.append(train_acc)
        
        # Validate
        val_loss, val_acc = validate(model, valid_loader, criterion)
        val_loss_list.append(val_loss)
        val_acc_list.append(val_acc)
        
        # Print current learning rate
        current_lr = optimizer.param_groups[0]['lr'] if optimizer else 0.0
        print(f"Epoch {epoch+1}: LR: {current_lr:.6f}, Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        
        # Learning rate scheduling
        if scheduler is not None:
            scheduler.step()
        
        # Check if this is the best model
        is_best = val_acc > best_acc
        best_acc = max(val_acc, best_acc)
        
        # Save history
        history = (train_loss_list, train_acc_list, val_loss_list, val_acc_list)
        
        # Build checkpoint filename based on optimizer and learning rate
        opt_name = args.optimizer.lower()
        lr_str = f"_lr_{args.lr}" if args.lr != 0.01 else ""
        scheduler_str = "" if args.scheduler.lower() != "none" else "_no_scheduler"
        filepath = f'./checkpoint/ckpt_{opt_name}{lr_str}{scheduler_str}.pth'
        
        # Save checkpoint
        if is_best:
            save_checkpoint(model, optimizer, scheduler, epoch, val_acc, history, filepath)
            no_improve_epochs = 0
        else:
            no_improve_epochs += 1
            
        # Early stopping
        if no_improve_epochs >= args.early_stopping:
            print(f"Early stopping after {no_improve_epochs} epochs without improvement")
            break
    
    # Test the best model
    print("\nTesting the best model...")
    
    # Load the best model
    best_model_path = './checkpoint/best_model.pth'
    
    if os.path.exists(best_model_path):
        checkpoint = torch.load(best_model_path)
        model.load_state_dict(checkpoint['model'])
    else:
        print("Warning: Best model not found, using current model for testing")
    
    # Test
    test_results = test_model(model, test_loader, criterion, processor.class_names)
    
    # Visualize results
    plot_training_history(train_loss_list, train_acc_list, val_loss_list, val_acc_list)
    plot_confusion_matrix(test_results['confusion_matrix'], processor.class_names)
    
    # TODO: Implement more visualization methods
    # Student task: Extend visualization functionality to better understand model performance
    # - Visualize misclassified samples
    # - Add more detailed statistics (F1 score, precision, recall, etc.)
    
    print(f"\nTraining complete! Best validation accuracy: {best_acc:.2f}%, Test accuracy: {test_results['accuracy']:.2f}%")


if __name__ == '__main__':
    main()