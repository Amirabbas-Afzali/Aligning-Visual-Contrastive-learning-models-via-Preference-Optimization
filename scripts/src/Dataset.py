import sys
import json 
import os
import pickle
import torch
import numpy as np
from torch.utils.data import DataLoader, Subset, Dataset
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, PILToTensor
from PIL import Image
import random
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader, Dataset

# Import the dataset classes
from datasets.caltech101 import Caltech101
from datasets.disentangling_ta import DisentanglingDataset
from datasets.dtd import DTD
from datasets.eurosat import EuroSAT
from datasets.fgvcaircraft import FGVCAircraft
from datasets.flowers102 import Flowers102
from datasets.food101 import Food101
from datasets.ImageNetV2 import ImageNetValDataset
from datasets.oxford_pets import OxfordIIITPet
from datasets.paint_ta import PAINTDataset
from datasets.rta100 import RTA100
from datasets.stanford_cars import StanfordCars
from datasets.sun397 import SUN397
from datasets.ImageNet100 import ImageNet100

base_path = 'your_base_path'

def dataset(args, preprocess):
    if args.dataset == 'imagenet':
        data = ImageNetValDataset(location=f'{base_path}/datasets', transform=preprocess)
    elif args.dataset == 'imagenet100':
        data = ImageNet100(root=f'{base_path}/datasets', split='train', preprocess=preprocess,)        
    elif args.dataset == 'caltech':
        data = Caltech101(root=f'{base_path}/datasets', transform=preprocess, download=True, make_typographic_dataset=True)
    elif args.dataset == 'pets':
        data = OxfordIIITPet(root=f'{base_path}/datasets', split='test', transform=preprocess, download=True, make_typographic_dataset=True)
    elif args.dataset == 'cars': 
        data = StanfordCars(root=f'{base_path}/datasets/', split='train', transform=preprocess, download=False, make_typographic_dataset=True)
    elif args.dataset == 'flowers':
        data = Flowers102(root=f'{base_path}/datasets', split='train', transform=preprocess, download=True, make_typographic_dataset=True)
    elif args.dataset == 'food':
        data = Food101(root=f'{base_path}/datasets', split='train', transform=preprocess, download=True)
    elif args.dataset == 'aircraft':
        data = FGVCAircraft(root=f'{base_path}/datasets', split='train', transform=preprocess, download=True)
    elif args.dataset == 'dtd':
        data = DTD(root=f'{base_path}/datasets', split='train', transform=preprocess, download=True)
    elif args.dataset == 'eurosat':
        data = EuroSAT(root=f'{base_path}/datasets', split='train', transform=preprocess, download=True)
    elif args.dataset == 'sun':
        data = SUN397(root=f'{base_path}/datasets', split='train', transform=preprocess, download=True)
    elif args.dataset == 'paint':
        data = PAINTDataset(root=f'{base_path}/datasets/paint', transform=preprocess)
    elif args.dataset == 'disentangling':
        data = DisentanglingDataset(root=f'{base_path}/datasets/disentangling', transform=preprocess)
    elif args.dataset == 'rta-100':
        data = RTA100(root='datasets', transform=preprocess)
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}") 
    return data

# Define the custom Dataset
class CustomDataset(Dataset):
    def __init__(self, data):
        """
        Args:
            data (list of tuples): List where each element is a tuple (torch.tensor, y1, y2, y3, type)
        """
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        """
        Args:
            idx (int): Index of the element to fetch

        Returns:
            tuple: (tensor, y1, y2, y3, type)
        """
        tensor, y1, y2, y3, type_ = self.data[idx]
        return tensor, y1, y2, y3, type_


class DatasetHandler:
    def __init__(self, obj, preprocessor, data_split_ratio, batch_size, pretokenize):
        self.obj = obj
        self.preprocessor = preprocessor
        self.data_split_ratio = data_split_ratio
        self.batch_size = batch_size
        self.pretokenize = pretokenize

        self.data = None
        self.NUM_CLASSES = 0
        self.train_subset = None
        self.val_subset = None
        self.test_subset = None

    def load_dataset(self):
        self.data = dataset(self.obj, self.preprocessor) 


    def load_typographic_image_classes(self, path):
        self.load_dataset()

        with open(path, "rb") as f:
            self.data._typographic_image_classes = pickle.load(f)
        # self.NUM_CLASSES = np.unique(self.data._typographic_image_classes).shape[0]  
        self.NUM_CLASSES = len(self.data.classes)  
        # print(self.data._typographic_image_classes)


    def load_typographic_image_classes_json(self, path):        
        with open(path, 'r') as file:
            self.data._typographic_image_classes = json.load(file)

        self.NUM_CLASSES = len(self.data.classes)  


    def prepare_prompts(self):
        prompts = [self.data.templates[0].format(f'{c}') for c in self.data.classes]
        if self.pretokenize:
            from transformers import CLIPProcessor
            clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
            prompts = clip_processor(text=prompts, padding=True, return_tensors='pt')
            print('tokenized dataset')
        return prompts 

    def split_data(self):
        train_size = int(self.data_split_ratio * len(self.data))
        indices = torch.randperm(len(self.data)).tolist()
        train_indices = indices[:train_size]
        test_indices = indices[train_size:]
        val_indices = torch.tensor(test_indices)[torch.randperm(len(test_indices))][:train_size]

        temp = 512*10
        train_size = 1024*2 
        self.train_subset = Subset(self.data, train_indices) # [:temp]
        self.test_subset = Subset(self.data, test_indices[:train_size])
        self.val_subset = Subset(self.data, val_indices[:temp]) 
        
    def split_data_for_zeroshut_datasets(self):
        train_size = int(self.data_split_ratio * len(self.data))
        indices = torch.randperm(len(self.data)).tolist()
        train_indices = indices[:train_size]
        test_indices = indices[train_size:]
        val_indices = torch.tensor(test_indices)[torch.randperm(len(test_indices))][:train_size]

        test_size = 512*10  # TODO 
        self.train_subset = None
        self.val_subset = None 
        self.test_subset = Subset(self.data, test_indices[:test_size])

    def get_dataloader(self, subset):
        n_cpu = os.cpu_count()
        return DataLoader(subset, batch_size=self.batch_size, shuffle=True, num_workers=n_cpu)

    def get_train_dataloader(self):
        return self.get_dataloader(self.train_subset)

    def get_test_dataloader(self):
        return self.get_dataloader(self.test_subset)

    def get_val_dataloader(self):
        return self.get_dataloader(self.val_subset)

    def create_data(self, save_path):
        # Create data as per the provided script
        self.load_dataset()
        with open(save_path, "wb") as f:
            pickle.dump(self.data._typographic_image_classes, f)
        print(f"Typographic image classes saved to {save_path}")


    def get_random_index(self, n, x):
        while True:
            rand_int = random.randint(0, n-1)
            if rand_int not in x:
                return rand_int



class ExpandedDataset(Dataset):
    def __init__(self, dataloader, param):
        self.true_images = []
        self.typo_images = []
        self.true_labels = []
        self.typo_labels = []

        # Expand the dataset by iterating over the dataloader
        for _, (true_imgs, typo_imgs_list, true_lbls, typo_lbls_list) in enumerate(dataloader):
            for i in range(len(true_imgs)):  # iterate over batch
                for j in range(param):  # expand to 10 times by selecting one typo sample at a time
                    self.true_images.append(true_imgs[i])
                    self.typo_images.append(typo_imgs_list[j][i])
                    self.true_labels.append(true_lbls[i])
                    self.typo_labels.append(typo_lbls_list[j][i])

    def __len__(self):
        return len(self.true_images)

    def __getitem__(self, idx):
        return (self.true_images[idx], [self.typo_images[idx],],
                self.true_labels[idx], [self.typo_labels[idx],])

def create_expanded_dataloader(org_dataloader, param):
    batch_size = org_dataloader.batch_size
    # Create an expanded dataset
    expanded_dataset = ExpandedDataset(org_dataloader, param)

    # Return a new DataLoader with the expanded dataset
    expanded_dataloader = DataLoader(expanded_dataset, batch_size=batch_size, shuffle=True)
    
    return expanded_dataloader




def main():
    # Define the preprocessor
    size = 168
    preprocessor = Compose([
        Resize(size=size, interpolation=Image.BICUBIC),
        CenterCrop(size=(size, size)),
        PILToTensor(),
    ])  

    # Define the object
    num_typographic = 1
    obj = type('obj', (object,), {'dataset': 'eurosat', 'num_typographic': num_typographic})

    # Initialize the DatasetHandler
    handler = DatasetHandler(obj, preprocessor, data_split_ratio=0.01, batch_size=64, pretokenize=True)

    handler.load_dataset() 
    prompts = handler.prepare_prompts() 
    handler.split_data_for_zeroshut_datasets()


    # Get DataLoaders
    test_dataloader = handler.get_test_dataloader()

    # Print to verify the setup
    print(f"Number of classes: {handler.NUM_CLASSES}")
    print(f"Test dataset size: {len(handler.test_subset)}")

  



if __name__ == "__main__":
    main()
