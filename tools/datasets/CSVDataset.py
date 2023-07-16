from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import transforms
import torchvision
import os
import pandas as pd
from PIL import Image

class Dataset_from_CSV(Dataset):
    def __init__(self, root, csv_file, transform=None):
        self.transform = transform
        self.read_data_from_csv(root, csv_file)

    def read_data_from_csv(self, root, file_path):
        self.im_paths = []
        self.ys = []
        self.I = []
        df = pd.read_csv(file_path)
        for id, value in enumerate(df['img_file']):
            image_name = value
            class_id = df['label_id'][id]
            image_path = os.path.join(root, image_name)

            self.im_paths.append(image_path)
            self.ys.append(class_id)
            self.I.append(id)

    def __len__(self):
        return len(self.I)

    def __getitem__(self, index):
        try:
            im = Image.open(self.im_paths[index])
            ## if grayscale, convert to RGB using PIL
            im = im.convert('RGB')
            # if len(list(im.split())) == 1: im = im.convert('RGB')
            if self.transform is not None:
                im = self.transform(im)
                return im, self.ys[index], index
            else:
                return self.im_paths[index], self.ys[index], index
        except Exception as e:
            print(self.im_paths[index])
            print(e)

    def get_label(self, index):
        return self.ys[index]

    def nb_classes(self):
        return len(set(self.ys))
    
    
# def testDataset(rootFolder, csvName):
    
#     train_transform = transforms.Compose([
#         transforms.RandomResizedCrop((224, 224)),
#         transforms.RandomHorizontalFlip(),
#         transforms.ToTensor(),
#         transforms.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])
#     ])
#     dataset = Dataset_from_CSV(rootFolder, csvName, transform=train_transform)
#     train_loader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=4, drop_last=True)
    
#     for i, (images, labels, _) in enumerate(train_loader):
#         print(images.shape)
# testDataset('dataset', 'dataset/csvs/train.csv')  