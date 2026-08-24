from torch.utils.data import Dataset, DataLoader, random_split
from PIL import Image

import os

class_to_idx = {
    "airplane": 0,
    "automobile": 1,
    "bird": 2,
    "cat": 3,
    "deer": 4,
    "dog": 5,
    "frog": 6,
    "horse": 7,
    "ship": 8,
    "truck": 9
}

# Custom Dataset Definition
class CIFAR10Dataset(Dataset):
    def __init__(self, image_dir, labels, transform=None):
        self.image_dir = image_dir
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        image_id = self.labels.iloc[index]["id"]
        image_label = self.labels.iloc[index]["label"]

        image_path = os.path.join(self.image_dir, f"{image_id}.png")

        image = Image.open(image_path)
        if self.transform:
            image = self.transform(image)
        label = class_to_idx[image_label]

        return image, label


# Function to split dataset into train, validation and test datasets
def split_dataset(dataset, train_size, val_size, test_size):

    train_dataset, val_dataset, test_dataset = random_split(
        dataset,
        [train_size, val_size, test_size]
    )

    return train_dataset, val_dataset, test_dataset


# Function to create dataloaders for the dataset
def create_dataloader(dataset, batch_size, shuffle):

    return DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle
    )

