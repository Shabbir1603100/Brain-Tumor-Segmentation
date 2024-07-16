import os
import numpy as np
import nibabel as nib
from glob import glob
from sklearn.model_selection import train_test_split
from scipy.ndimage import zoom, rotate
from skimage import exposure
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

# Augmentation function
def augment(image, mask):
    if np.random.rand() > 0.5:
        angle = np.random.uniform(-10, 10)
        image = rotate(image, angle, axes=(1, 2), reshape=False, mode='nearest')
        mask = rotate(mask, angle, axes=(1, 2), reshape=False, mode='nearest')

    if np.random.rand() > 0.5:
        image = np.flip(image, axis=1).copy()
        mask = np.flip(mask, axis=1).copy()

    if np.random.rand() > 0.5:
        image = np.flip(image, axis=2).copy()
        mask = np.flip(mask, axis=2).copy()

    return image, mask

# Normalization function
def normalize(image):
    return (image - np.min(image)) / (np.max(image) - np.min(image))

# Resizing function
def resize(image, target_shape=(128, 128, 128)):
    factors = [target_shape[i] / image.shape[i] for i in range(3)]
    return zoom(image, factors, order=1)

# Define a custom Dataset class for BraTS21
class BraTSDataset(Dataset):
    def __init__(self, image_paths, mask_paths, transform=None, augment=False):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.transform = transform
        self.augment = augment

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = nib.load(self.image_paths[idx]).get_fdata()
        mask = nib.load(self.mask_paths[idx]).get_fdata()

        # Normalize and resize
        image = normalize(image)
        image = resize(image)
        mask = resize(mask, target_shape=image.shape)

        if self.augment:
            image, mask = augment(image, mask)

        image = np.expand_dims(image, axis=0)  # Add channel dimension
        mask = np.expand_dims(mask, axis=0)  # Add channel dimension

        sample = {'image': torch.tensor(image, dtype=torch.float32), 'mask': torch.tensor(mask, dtype=torch.long)}
        
        if self.transform:
            sample = self.transform(sample)
        
        return sample

# Load image and mask paths
image_paths = sorted(glob('/path/to/BraTS21/images/*.nii.gz'))
mask_paths = sorted(glob('/path/to/BraTS21/masks/*.nii.gz'))

# Split into training and validation sets
train_images, val_images, train_masks, val_masks = train_test_split(image_paths, mask_paths, test_size=0.2, random_state=42)

# Create DataLoaders with augmentation for the training set
train_dataset = BraTSDataset(train_images, train_masks, augment=True)
val_dataset = BraTSDataset(val_images, val_masks, augment=False)
train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False)

import torch.optim as optim

def train_model(model, dataloader, criterion, optimizer, num_epochs=20):
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        for i, data in enumerate(dataloader):
            inputs, labels = data['image'].to(device), data['mask'].to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            if i % 10 == 9:    # Print every 10 mini-batches
                print(f'Epoch {epoch+1}, Batch {i+1}, Loss: {running_loss / 10:.4f}')
                running_loss = 0.0

    print('Finished Training')

# Define the loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Train the model
train_model(model, train_loader, criterion, optimizer, num_epochs=20)


