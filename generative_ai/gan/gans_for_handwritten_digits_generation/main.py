import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.datasets import CocoDetection
from torchvision.transforms import transforms
from unet import UNet  # assuming you have implemented the U-Net architecture in a separate file

# define transforms for the dataset
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor()
])

# load the COCO dataset
coco_train = CocoDetection(root='path/to/coco/train', annFile='path/to/coco/train/annotations.json', transform=transform)
coco_val = CocoDetection(root='path/to/coco/val', annFile='path/to/coco/val/annotations.json', transform=transform)

# define the U-Net model
model = UNet()

# define the loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# define the data loaders
train_loader = DataLoader(coco_train, batch_size=16, shuffle=True)
val_loader = DataLoader(coco_val, batch_size=16, shuffle=False)

# train the model
num_epochs = 10

for epoch in range(num_epochs):
    # train for one epoch
    model.train()
    train_loss = 0.0
    for i, (images, targets) in enumerate(train_loader):
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    train_loss /= len(train_loader)

    # evaluate on validation set
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for images, targets in val_loader:
            outputs = model(images)
            loss = criterion(outputs, targets)
            val_loss += loss.item()
        val_loss /= len(val_loader)

    # print the losses
    print(f'Epoch {epoch+1}/{num_epochs}, train_loss: {train_loss:.4f}, val_loss: {val_loss:.4f}')

# save the model
torch.save(model.state_dict(), 'path/to/save/model.pth')
