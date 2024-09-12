import torch
import torch
from torch.utils.data import DataLoader, TensorDataset, Dataset, random_split
from torchvision.models import resnet50, ResNet50_Weights
import os
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR

# Adversarial images dataset
class AdversarialDataset(Dataset):
    def __init__(self, adv_images, labels):
        self.adv_images = adv_images
        self.labels = labels

    def __len__(self):
        return len(self.adv_images)

    def __getitem__(self, idx):
        image = self.adv_images[idx]
        label = self.labels[idx]
        return image, label

def load_adversarial_images(results_path):
    data = torch.load(results_path)
    adv_images = data['adv_images']
    labels = data['labels']
    
    # Filter out invalid labels
    num_classes = 1000  # for ImageNet dataset
    valid_indices = (labels >= 0) & (labels < num_classes)
    adv_images = adv_images[valid_indices]
    labels = labels[valid_indices]
    
    return adv_images, labels

def check_labels(file_path):
    adv_images, labels = load_adversarial_images(file_path)

    # Check if all labels are within the valid range
    num_classes = 1000  # for ImageNet
    invalid_labels = labels[(labels < 0) | (labels >= num_classes)]

    if len(invalid_labels) > 0:
        print(f"Found {len(invalid_labels)} invalid labels:", invalid_labels)
        
    # filter out invalid entries
    valid_indices = (labels >= 0) & (labels < num_classes)
    adv_images = adv_images[valid_indices]
    labels = labels[valid_indices]


# Load the adversarial images and labels
saved_data = torch.load('results_dataset/adv_images_eps_0.1.pt')
adv_images = saved_data['adv_images']
labels = saved_data['labels']

results_dir = 'results_dataset'
adversarial_files = [f for f in os.listdir(results_dir) if f.endswith('.pt')]

# Create the dataset

all_adv_images = []
all_labels = []

all_adv_images.append(adv_images)
all_labels.append(labels)
    
for file_name in adversarial_files:
    print(f'Loading {file_name}...')
    file_path = os.path.join(results_dir, file_name)
    
    try:
        adv_images, labels = load_adversarial_images(file_path)
    except Exception as e:
        print(f"Error loading {file_name}: {e}. Skipping this file.")
        continue
    
    # Store all adversarial images and labels
    all_adv_images.append(adv_images)
    all_labels.append(labels)


all_adv_images = torch.cat(all_adv_images, dim=0)
all_labels = torch.cat(all_labels, dim=0)

combined_dataset = AdversarialDataset(all_adv_images, all_labels)

# Split dataset into training (80%) and testing (20%)
train_size = int(0.8 * len(combined_dataset))
test_size = len(combined_dataset) - train_size
train_dataset, test_dataset = random_split(combined_dataset, [train_size, test_size])

# Create dataloaders
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# adv_dataset = AdversarialDataset(adv_images, labels)
# # adv_dataset = TensorDataset(adv_images, labels)

# # Create the dataloader for fine-tuning
# adv_loader = DataLoader(adv_dataset, batch_size=32, shuffle=True) 0


# Load the pre-trained ResNet model
weights = ResNet50_Weights.DEFAULT
model = resnet50(weights=weights)

# Freeze all layers except the final fully connected layer
for param in model.parameters():
    param.requires_grad = False

# Replace the final fully connected layer for fine-tuning
num_ftrs = model.fc.in_features
# model.fc = nn.Linear(num_ftrs, len(set(all_labels.tolist())))  
model.fc = nn.Linear(num_ftrs, 1000)

# Move model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Loss function
criterion = nn.CrossEntropyLoss()
# Optimizer
optimizer = optim.Adam(model.fc.parameters(), lr=0.001)
# optimizer = optim.Adam(model.parameters(), lr=lr)
scheduler = StepLR(optimizer, step_size=7, gamma=0.1)


# set model to training mode    
model.train() 

# Training loop
num_epochs = 10 
for epoch in range(num_epochs):
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        # Validate labels
        if torch.any((labels < 0) | (labels >= 1000)):  # ImageNet # classes = 1000
            print(f"Invalid labels found in batch {i}. Skipping this batch.")
            continue
    
        # Zero the parameter gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward pass and optimization
        loss.backward()
        optimizer.step()

        # Accumulate loss and accuracy
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        # Free memory after each batch
        torch.cuda.empty_cache()

    scheduler.step()
    
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss / len(train_loader)}, Accuracy: {100 * correct / total:.2f}%")

print("Finished fine-tuning the model")

# optional: save the fine-tuned model
torch.save(model.state_dict(), 'results/fine_tuned_resnet.pth')

########################## TESTING #####################
def test_model(model, test_loader, device='cuda'):
    model.eval()  # Set model to evaluation mode
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            predicted = outputs.argmax(dim=1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f'Accuracy on the test set: {accuracy}%')
    return accuracy

# Load the fine-tuned model
model.load_state_dict(torch.load('results/fine_tuned_resnet.pth'))
model.to(device)

# Test the model on the test dataset
test_model(model, test_loader)

# Load the test dataset 
# hidden_test_images, hidden_test_labels = load_adversarial_images('results_dataset/adv_images_eps_0.3.pt')
# hidden_test_dataset = TensorDataset(hidden_test_images, hidden_test_labels)
# hidden_test_loader = DataLoader(hidden_test_dataset, batch_size=32)

# results_dir = 'results_dataset'
# trained_on = 'adv_images_eps_0.1.pt'  # The set the model was trained on
# adversarial_files = [f for f in os.listdir(results_dir) if f.endswith('.pt') and f != trained_on]

# # Test the model on each adversarial image set
# for file_name in adversarial_files:
#     print(f'\nTesting on {file_name}...')
#     file_path = os.path.join(results_dir, file_name)
    
#     # Load the hidden adversarial images and labels
#     hidden_test_images, hidden_test_labels = load_adversarial_images(file_path)
#     hidden_test_dataset = TensorDataset(hidden_test_images, hidden_test_labels)
#     hidden_test_loader = DataLoader(hidden_test_dataset, batch_size=32)
    
#     # Test the model on the loaded dataset
#     test_model(model, hidden_test_loader)
