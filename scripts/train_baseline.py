
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

from torch.utils.tensorboard import SummaryWriter
import torch


def parse_args():
    parser = argparse.ArgumentParser(description='Simple Classification Baseline on PyTorch Training')
    parser.add_argument("--model_name", default='mobilenet', type=str, help="Model name")
    parser.add_argument("--dataset_root", default='dataset', type=str, help="Dataset Root Folder")
    parser.add_argument("--train_csv_path", default='dataset/csvs/train.csv', type=str, help="Train CSV Path")
    parser.add_argument("--test_csv_path", default='dataset/csvs/test.csv', type=str, help="Test CSV Path")
    args = parser.parse_args()
    return args

def train_single_epoch(model, loss, optimizer, data_fetcher, device, epoch, writer):
    loss_per_epoch = 0
    model = model.to(device)
    model.train()
    for i, (images, labels, indexes) in enumerate(data_fetcher):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        outputs = model(images)
        loss_per_batch = loss(outputs, labels)
        loss_per_epoch += loss_per_batch.item()

        optimizer.zero_grad()
        loss_per_batch.backward()
        optimizer.step()
        logging.info('Training Loss ' + ' Epoch ' + str(epoch) + ' for minibatch: ' + str(i) + '/' + str(len(data_fetcher)) +
                     ' is ' + str(loss_per_batch.item())) \
            if i % 10 == 0 else None

    return loss_per_epoch/len(data_fetcher)

def main(args):
    model_name = args.model_name
    dataset_root=args.dataset_root
    training_csv_path=args.train_csv_path
    test_csv_path=args.test_csv_path
    
    
    experiment_name = os.path.join("logs", "baseline")
    if not os.path.isdir(experiment_name):
        os.makedirs(experiment_name)
        
        
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
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop((224, 224)),
        transforms.RandomHorizontalFlip(),
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
        "lr": 1e-04
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
    
    for current_epoch in range(10):
        train_loss = train_single_epoch(model, loss, optimizer, train_loader, device, current_epoch, None)
    

if __name__ == '__main__':
    main(parse_args())