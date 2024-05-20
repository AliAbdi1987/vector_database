import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torch.nn.functional as F
from torchsummary import summary
import numpy as np
import matplotlib.pyplot as plt
import os


# Load MNIST dataset
transform = transforms.Compose([
    transforms.ToTensor(),    # Convert images to PyTorch tensors
    transforms.Normalize((0.1307,), (0.3081,)) # Normalize the images
])


# Load MNIST dataset
# Data augmentation

transform2 = transforms.Compose([ # Define a sequence of data transformations to apply to the images for data augmentation to prevent overfitting
    transforms.RandomRotation(5),  # Rotate the image by 10 degrees
    transforms.RandomAdjustSharpness(2),  # Adjust the sharpness of the image
    transforms.RandomAffine(0, translate=(0.1, 0.1)),  # Translate the image horizontally and vertically
    transforms.RandomAffine(0, shear=10),  # Shear the image by 10 degrees
    transforms.ToTensor(),    # Convert images to PyTorch tensors
    transforms.Normalize((0.1307,), (0.3081,)) # Normalize the images
])


train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform2)
test_dataset = datasets.MNIST('./data', train=False, download=True, transform=transform)

# Create data loaders
train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=64, shuffle=True)
test_loader = torch.utils.data.DataLoader(
    test_dataset, batch_size=64, shuffle=False)

# Data visualization
class_names = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
sample_per_class = 5

plt.figure(figsize=(10, 10))
plt.suptitle('Samples')  # Add title to the figure

for cls, name in enumerate(class_names): 
    idxs = np.flatnonzero(np.array(train_dataset.targets) == cls)
    idxs = np.random.choice(idxs, sample_per_class, replace=False)
    for i, idx in enumerate(idxs):
        plt.subplot(sample_per_class, len(class_names), i * len(class_names) + cls + 1)
        plt.imshow(train_dataset.data[idx], cmap='gray')
        plt.axis('off')
        if i == 0:
            plt.title(name)
plt.show()

# Convolutional Neural Network
class CNN(nn.Module):
    """
    Convolutional Neural Network with 3 convolutional layers and 3 fully connected layers.
    Dropout is applied after each pooling layer and fully connected layer.
    filters: 64, 64, 128, 128, 256, 256
    fully connected layers: 1024, 512, 10
    flatenning: 256 * 3 * 3
    """
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.conv6 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(256 * 3 * 3, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 10)
        self.dropout1 = nn.Dropout(0.25) # Dropout with 25% probability 
        self.dropout2 = nn.Dropout(0.5) # Dropout with 50% probability
    
    def forward(self, x):
        x = F.leaky_relu(self.conv1(x)) # Apply Leaky ReLU activation function
        x = F.leaky_relu(self.conv2(x))
        x = self.pool(x)
        x = self.dropout1(x)
        x = F.leaky_relu(self.conv3(x))
        x = F.leaky_relu(self.conv4(x))
        x = self.pool(x)
        x = self.dropout1(x)
        x = F.leaky_relu(self.conv5(x))
        x = F.leaky_relu(self.conv6(x))
        x = self.pool(x)
        x = self.dropout1(x) # Apply dropout
        x = x.view(-1, 256 * 3 * 3)
        x = F.leaky_relu(self.fc1(x))
        x = self.dropout2(x) # Apply dropout
        x = F.leaky_relu(self.fc2(x))
        x = self.dropout2(x)
        x = self.fc3(x)
        return x
    
# Instantiate the network, loss function, and optimizer
model = CNN()
criterion = nn.CrossEntropyLoss() # Cross-entropy loss
optimizer = optim.Adam(model.parameters(), lr=0.0001) # Adam optimizer with learning rate of 0.0001

# Initialize lists to track the loss and accuracy
train_losses = []
test_losses = []
train_accuracies = []
test_accuracies = []

# Lists to hold misclassified example data
misclassified_images = []
misclassified_preds = []
misclassified_true = []

# Training loop
num_epochs = 35
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Training on device {device}")
model.to(device)
for epoch in range(num_epochs): # Loop over the dataset multiple times
    model.train()
    train_loss = 0
    correct_train = 0
    total_train = 0
    for batch_idx, (data, target) in enumerate(train_loader): # Get the inputs; data is a list of [inputs, labels]
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = nn.functional.cross_entropy(output, target)
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * data.size(0)
        _, predicted = torch.max(output, 1)
        correct_train += (predicted == target).sum().item()
        total_train += target.size(0)
    
    train_losses.append(train_loss / total_train)
    train_accuracies.append(100 * correct_train / total_train)

    # Evaluate the model
    model.eval()
    test_loss = 0
    correct_test = 0
    total_test = 0
    # correct = 0
    with torch.no_grad(): # Turn off gradients for validation, saves memory and computations
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = nn.functional.cross_entropy(output, target, reduction='sum')
            test_loss += loss.item()
            _, predicted = torch.max(output.data, 1)   
            correct_test += (predicted == target).sum().item()
            total_test += target.size(0)

            # Collect misclassified examples
            mask = ~(predicted == target)
            misclassified_images.extend(data[mask].cpu())
            misclassified_preds.extend(predicted[mask].cpu())
            misclassified_true.extend(target[mask].cpu())
    
    test_losses.append(test_loss / total_test)
    test_accuracies.append(100 * correct_test / total_test)
    
    print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_losses[-1]:.4f}, Train Accuracy: {train_accuracies[-1]:.2f}%, Test Loss: {test_losses[-1]:.4f}, Test Accuracy: {test_accuracies[-1]:.2f}%")
    
# Plotting the training and test losses
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(range(2, num_epochs + 1), train_losses[1:], label='Train Loss')
plt.plot(range(2, num_epochs + 1), test_losses[1:], label='Test Loss')
plt.title('Epoch vs Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

# Plotting the training and test accuracies
plt.subplot(1, 2, 2)
plt.plot(range(2, num_epochs + 1), train_accuracies[1:], label='Train Accuracy')
plt.plot(range(2, num_epochs + 1), test_accuracies[1:], label='Test Accuracy')
plt.title('Epoch vs Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')
plt.legend()

plt.show()

def plot_misclassified(misclassified_images, misclassified_preds, misclassified_true, num_images=10):
    # Ensure we have enough images to plot
    num_images = min(num_images, len(misclassified_images))
    if num_images == 0:
        print("No misclassified images to display.")
        return

    # Set up subplot dimensions
    fig, axes = plt.subplots(nrows=2, ncols=num_images, figsize=(15, 6))
    if num_images == 1:  # Fix for the edge case where we have only one subplot
        axes = np.array([[axes]])

    for i in range(num_images):
        img = misclassified_images[i].squeeze()  # Remove channel dimension
        pred = misclassified_preds[i].item()
        true = misclassified_true[i].item()

        # Plot image
        ax = axes[0][i]  # Top row for images
        ax.imshow(img, cmap='gray', interpolation='none')
        ax.set_title(f'Pred: {pred}, True: {true}')
        ax.axis('off')

        # Optional: Add more plots below each image if needed
        # For example, plotting the difference or error heatmap
        # ax2 = axes[1][i]
        # ax2.imshow(...)

    plt.tight_layout()
    plt.show()


# Call the function to plot misclassified examples
plot_misclassified(misclassified_images, misclassified_preds, misclassified_true)

model.to(device)
summary(model, (1, 28, 28))

# Save the model
torch.save(model.state_dict(), 'mnist_cnn.pth')
