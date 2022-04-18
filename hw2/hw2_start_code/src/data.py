from torchvision import datasets, transforms
from torch.utils.data import DataLoader, WeightedRandomSampler
import os


## Note that: here we provide a basic solution for loading data and transforming data.
## You can directly change it if you find something wrong or not good enough.

## the mean and standard variance of imagenet dataset
## mean_vals = [0.485, 0.456, 0.406]
## std_vals = [0.229, 0.224, 0.225]

def load_data(data_dir="../data/", train_dir="1-Large-Scale", test_dir="test", input_size=224, batch_size=36):
    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize(input_size),
            transforms.RandomResizedCrop(input_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'test': transforms.Compose([
            transforms.Resize(input_size),
            transforms.CenterCrop(input_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }
    ## The default dir is for the first task of large-scale deep learning
    ## For other tasks, you may need to modify the data dir or even rewrite some part of 'data.py'
    image_dataset_train = datasets.ImageFolder(os.path.join(data_dir, train_dir, 'train'), data_transforms['train'])
    image_dataset_valid = datasets.ImageFolder(os.path.join(data_dir, test_dir), data_transforms['test'])

    train_loader = DataLoader(image_dataset_train, batch_size=batch_size, shuffle=True, num_workers=4)
    valid_loader = DataLoader(image_dataset_valid, batch_size=batch_size, shuffle=False, num_workers=4)

    return train_loader, valid_loader


# banlance dataset
def count_class_num(dataset):
    cls_num = [0 for _ in range(10)]
    for x, y in dataset:
        cls_num[y] += 1
    weights = [1 / cls_num[y] for _, y in dataset]

    t = [0 for _ in range(10)]
    for k in range(len(weights)):
        t[dataset[k][1]] += weights[k]
    return weights


# load data for task c
def load_data_c(data_dir="../data/", train_dir="1-Large-Scale", test_dir="test", input_size=224, batch_size=36):
    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize(input_size),
            transforms.RandomResizedCrop(input_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'test': transforms.Compose([
            transforms.Resize(input_size),
            transforms.CenterCrop(input_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }
    ## The default dir is for the first task of large-scale deep learning
    ## For other tasks, you may need to modify the data dir or even rewrite some part of 'data.py'
    image_dataset_train = datasets.ImageFolder(os.path.join(data_dir, train_dir, 'train'), data_transforms['train'])
    image_dataset_valid = datasets.ImageFolder(os.path.join(data_dir, test_dir), data_transforms['test'])

    wrsampler = WeightedRandomSampler(count_class_num(image_dataset_train), len(count_class_num(image_dataset_train)))
    train_loader = DataLoader(image_dataset_train, batch_size=batch_size, sampler=wrsampler,
                              num_workers=4)
    valid_loader = DataLoader(image_dataset_valid, batch_size=batch_size, shuffle=False, num_workers=4)

    return train_loader, valid_loader
