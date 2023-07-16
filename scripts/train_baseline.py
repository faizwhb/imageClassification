
## add parent directory to path for importing modules
import os, sys, inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)
from tools.datasets.CSVDataset import Dataset_from_CSV
from tools.models import model_getter
import logging

import argparse
from torch.utils.data import DataLoader
from torchvision import transforms
import torch.nn.functional as F

from torch.utils.tensorboard import SummaryWriter
import torch
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description='Simple Classification Baseline on PyTorch Training')
    parser.add_argument("--model_name", default='mobilenet', type=str, help="Model name")
    parser.add_argument("--dataset_root", default='dataset', type=str, help="Dataset Root Folder")
    parser.add_argument("--train_csv_path", default='dataset/csvs/train.csv', type=str, help="Train CSV Path")
    parser.add_argument("--test_csv_path", default='dataset/csvs/test.csv', type=str, help="Test CSV Path")
    args = parser.parse_args()
    return args

def evaluate(test_loader, model, device, epoch):
    # switch to evaluation mode
    model = model.to(device)
    model.eval()
    correct = 0
    total = 0
    targets = []
    predictions = []
    with torch.no_grad():
        for i, (input, target, index) in enumerate(test_loader):
            input = input.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)

            # accumulate labels to compute evaluation metrics
            targets += np.squeeze(target.tolist()).tolist()
            
            # compute output
            outputs = model(input)
            predicted = F.softmax(outputs, dim=-1)
            predicted = torch.argmax(predicted, dim=-1)
            

            # accumulate labels to compute evaluation metrics
            predictions += np.squeeze(predicted.tolist()).tolist()
            total += target.size(0)
            correct += (predicted == target).sum().item()
    accuracy = 100*correct/total
    logging.info(f'Validation Accuracy for Epoch {epoch} is {accuracy}%')

    return accuracy

def validation_single_epoch(model, loss, data_fetcher, device, epoch):
    loss_per_epoch = 0
    model = model.to(device)
    model.eval()
    with torch.no_grad():
        for i, (images, labels, _) in enumerate(data_fetcher):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            outputs = model(images)
            loss_per_batch = loss(outputs, labels)
            loss_per_epoch += loss_per_batch.item()
    epoch_loss = loss_per_epoch/len(data_fetcher)
    logging.info(f'Validation for Epoch{epoch} is {epoch_loss}')

    return epoch_loss

def train_single_epoch(model, loss, optimizer, data_fetcher, device, epoch):
    loss_per_epoch = 0
    model = model.to(device)
    model.train()
    for i, (images, labels, _) in enumerate(data_fetcher):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        outputs = model(images)
        loss_per_batch = loss(outputs, labels)
        loss_per_epoch += loss_per_batch.item()

        optimizer.zero_grad()
        loss_per_batch.backward()
        optimizer.step()
    epoch_loss = loss_per_epoch/len(data_fetcher)
    logging.info(f'Training Loss for Epoch{epoch} is {epoch_loss}')

    return epoch_loss

def main(args):
    model_name = args.model_name
    dataset_root=args.dataset_root
    training_csv_path=args.train_csv_path
    test_csv_path=args.test_csv_path
    
    
    experiment_name = os.path.join("logs", "baseline")
    if not os.path.isdir(experiment_name):
        os.makedirs(experiment_name)
        
    
    writer = SummaryWriter(experiment_name)
    
    logging.basicConfig(
        format="%(asctime)s %(message)s",
        level=logging.INFO,
        handlers=[
            logging.FileHandler(
                "{0}/{1}/{1}.log".format(
                    "logs",
                    "baseline"
                )
            ),
            logging.StreamHandler()
        ]
    )
    
    # define Transformations for Training Dataset
    # Currently only add basic augmentation like rotation, color jitter and flips, but for the given dataset I want to explore a richer class of augmentations
    #  1. RandAugment an Augmentation Policy for Image Classification Finetuned on ImageNet
    #  2. BASNet or Segment Anything Model to do Segmentation and Swap the Backgrounds 
    
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10), 
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
        transforms.GaussianBlur(3, sigma=(0.1, 2.0)),
        transforms.ToTensor(), 
        transforms.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])
        ])
    
    # define Transformations for Test Dataset
    test_transform = transforms.Compose([transforms.Resize((224, 224)),
                                         transforms.ToTensor(),
                                         transforms.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])
                                         ])
    
    
    # define training and test datasets
    train_dataset = Dataset_from_CSV(dataset_root, training_csv_path, transform=train_transform)
    test_dataset = Dataset_from_CSV(dataset_root, test_csv_path, transform=test_transform)
    
    
    # define data loaders for training and testing
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=4, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False, num_workers=4, drop_last=True)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    num_classes = train_dataset.nb_classes()
    print(num_classes, model_name)
     
    model = model_getter.get_model_for_finetuning(model_name, num_classes)
    
    
    optimizer = torch.optim.Adam([
        {
        "params": model.pretrained_model.parameters(),
        "lr": 1e-05
        },
        {"params": model.fc.parameters(),
        "lr": 1e-03
        }
    ], eps=1e-08)
    
    loss = torch.nn.CrossEntropyLoss()
    
    logging.info('Dataset selected:' + dataset_root)
    logging.info(f'Number of classes in training:{train_dataset.nb_classes()}')
    logging.info(f'Number of Images in training:{len(train_dataset)}')
    logging.info(f'Number of classes in validation:{test_dataset.nb_classes()}')
    logging.info(f'Number of Images in validation:{len(test_dataset)}')
    
    best_accuracy = 0
    best_epoch = 0
    
    for current_epoch in range(100):
        train_loss = train_single_epoch(model, loss, optimizer, train_loader, device, current_epoch)
        
        validation_loss = validation_single_epoch(model, loss, test_loader, device, current_epoch)
        
        val_accuracy = evaluate(test_loader, model, device, current_epoch)
        
        # We can add gradient norm to check the stability of the training process
        # Other Metrics for Classification like Confusion Matrix, Precision, Recall for better debgging and insights into the training process
        
        writer.add_scalar('Loss/train', train_loss, current_epoch)
        writer.add_scalar('Loss/val', validation_loss, current_epoch)
        writer.add_scalar('Accuracy/val', val_accuracy, current_epoch)
        
        if best_accuracy < val_accuracy:
            best_accuracy = val_accuracy
            best_epoch = current_epoch
            logging.info('Best Validation Loss: ' + str(validation_loss))
            logging.info('Best Accuracy: ' + str(best_accuracy))
            logging.info('Best Epoch: ' + str(best_epoch))
            logging.info(f'Best Model saved at {experiment_name}')
            torch.save(model.state_dict(), os.path.join(experiment_name, f'{model_name}_best.pth'))
    
    logging.info('---------------------------------------------------------------------------------------------')
    logging.info('---------------------------------------------------------------------------------------------')
    logging.info('Training Completed...')
    logging.info('Best Validation Loss: ' + str(validation_loss))
    logging.info('Best Accuracy: ' + str(best_accuracy))
    logging.info('Best Epoch: ' + str(best_epoch))
 

if __name__ == '__main__':
    main(parse_args())