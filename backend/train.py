import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from model import FruitClassifier, save_classes

def train_model():
    print("Initializing training...")
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Hyperparameters
    batch_size = 64
    num_epochs = 1
    learning_rate = 0.001

    # Data transformation
    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    # Data loaders
    data_dir = 'dataset'
    
    if not os.path.exists(data_dir) or not os.path.exists(os.path.join(data_dir, 'train')):
        print("Dataset not found! Please make sure 'dataset/train' exists.", flush=True)
        return

    image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                          data_transforms[x])
                  for x in ['train', 'val']}
    
    # Use subset to speed up training dramatically (200 random samples per phase)
    
    dataloaders = {x: DataLoader(image_datasets[x], batch_size=batch_size,
                                 shuffle=True, num_workers=0)
                  for x in ['train', 'val']}
    
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
    
    # Get original classes
    original_train_dataset = datasets.ImageFolder(os.path.join(data_dir, 'train'))
    class_names = original_train_dataset.classes
    num_classes = len(class_names)
    
    print(f"Training on {num_classes} classes: {class_names[:5]}...", flush=True)
    save_classes(class_names, "backend/classes.json")

    # Initialize model
    model = FruitClassifier(num_classes=num_classes).to(device)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.model.classifier.parameters(), lr=learning_rate)

    best_acc = 0.0

    # Training loop
    # Modified to run just 1 epoch for quick demo purposes
    num_epochs = 1
    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs}', flush=True)
        print('-' * 10, flush=True)

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            for i, (inputs, labels) in enumerate(dataloaders[phase]):
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                torch.save(model.state_dict(), 'backend/fruit_model.pth')

        print()

    print(f'Training complete. Best val Acc: {best_acc:4f}')
    print('Model saved to backend/fruit_model.pth')

if __name__ == '__main__':
    train_model()
