import torch
import torchvision
import torchvision.transforms.v2 as transforms
import torch.nn as nn
import torch.optim as optim
from cnn import Conv_Net
from ffn import FF_Net
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import numpy as np

# Based on skeleton from fashionmnist.py fileciteturn1file0

# Part 1: Data preprocessing
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])
batch_size = 64

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Part 2: Load dataset
trainset = torchvision.datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)

testset = torchvision.datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False)

# Part 3: Model instantiation
feedforward_net = FF_Net().to(device)
conv_net = Conv_Net().to(device)

# Part 4: Loss and optimizers
criterion = nn.CrossEntropyLoss()
optimizer_ffn = optim.SGD(feedforward_net.parameters(), lr=0.01, momentum=0.9)
optimizer_cnn = optim.Adam(conv_net.parameters(), lr=0.001)

# Part 5: Train feedforward network
num_epochs_ffn = 10
train_losses_ffn = []
for epoch in range(num_epochs_ffn):
    running_loss = 0.0
    for inputs, labels in trainloader:
        inputs, labels = inputs.to(device), labels.to(device)
        # Flatten inputs for FFN
        inputs = inputs.view(inputs.size(0), -1)

        optimizer_ffn.zero_grad()
        outputs = feedforward_net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer_ffn.step()
        running_loss += loss.item()
    epoch_loss = running_loss / len(trainloader)
    train_losses_ffn.append(epoch_loss)
    print(f"FFN Epoch {epoch+1}/{num_epochs_ffn}, Loss: {epoch_loss:.4f}")
print("Finished FFN Training")
torch.save(feedforward_net.state_dict(), 'ffn.pth')

# Part 5: Train convolutional network
num_epochs_cnn = 10
train_losses_cnn = []
for epoch in range(num_epochs_cnn):
    running_loss = 0.0
    for inputs, labels in trainloader:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer_cnn.zero_grad()
        outputs = conv_net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer_cnn.step()
        running_loss += loss.item()
    epoch_loss = running_loss / len(trainloader)
    train_losses_cnn.append(epoch_loss)
    print(f"CNN Epoch {epoch+1}/{num_epochs_cnn}, Loss: {epoch_loss:.4f}")
print("Finished CNN Training")
torch.save(conv_net.state_dict(), 'cnn.pth')

# Part 6: Evaluation

def evaluate(model, loader, is_ffn=False):
    model.eval()
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    misclassified = []
    correct_examples = []
    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            if is_ffn:
                inputs_flat = inputs.view(inputs.size(0), -1)
                outputs = model(inputs_flat)
            else:
                outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            # store first misclassified and correct example
            for img, pred, true in zip(inputs.cpu(), predicted.cpu(), labels.cpu()):
                if pred != true and len(misclassified) < 1:
                    misclassified.append((img, pred.item(), true.item()))
                if pred == true and len(correct_examples) < 1:
                    correct_examples.append((img, pred.item(), true.item()))
    accuracy = correct / total
    return accuracy, all_labels, all_preds, misclassified, correct_examples

acc_ffn, y_true_ffn, y_pred_ffn, mis_ffn, corr_ffn = evaluate(feedforward_net, testloader, is_ffn=True)
acc_cnn, y_true_cnn, y_pred_cnn, mis_cnn, corr_cnn = evaluate(conv_net, testloader)
print(f"Accuracy FFN: {acc_ffn:.4f}")
print(f"Accuracy CNN: {acc_cnn:.4f}")

# Part 7: Plotting
# Training loss curves
plt.figure()
plt.plot(range(1, num_epochs_ffn+1), train_losses_ffn, label='FFN Loss')
plt.plot(range(1, num_epochs_cnn+1), train_losses_cnn, label='CNN Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss Comparison')
plt.legend()
plt.savefig('training_loss.png')

# Function to show images
classes = trainset.classes if hasattr(trainset, 'classes') else [str(i) for i in range(10)]
def imshow(img):
    img = img * 0.5 + 0.5  # unnormalize
    npimg = img.numpy()
    plt.imshow(np.squeeze(npimg), cmap='gray')
    plt.axis('off')

# Example figures
plt.figure(figsize=(6,3))
plt.subplot(1,2,1)
imshow(mis_ffn[0][0])
plt.title(f"FFN Mis: P={classes[mis_ffn[0][1]]}, T={classes[mis_ffn[0][2]]}")
plt.subplot(1,2,2)
imshow(corr_ffn[0][0])
plt.title(f"FFN Corr: P={classes[corr_ffn[0][1]]}, T={classes[corr_ffn[0][2]]}")
plt.savefig('ffn_examples.png')

plt.figure(figsize=(6,3))
plt.subplot(1,2,1)
imshow(mis_cnn[0][0])
plt.title(f"CNN Mis: P={classes[mis_cnn[0][1]]}, T={classes[mis_cnn[0][2]]}")
plt.subplot(1,2,2)
imshow(corr_cnn[0][0])
plt.title(f"CNN Corr: P={classes[corr_cnn[0][1]]}, T={classes[corr_cnn[0][2]]}")
plt.savefig('cnn_examples.png')

# Parameter counts
ffn_params = sum(p.numel() for p in feedforward_net.parameters())
cnn_params = sum(p.numel() for p in conv_net.parameters())
print(f"FFN Parameters: {ffn_params}")
print(f"CNN Parameters: {cnn_params}")

# Confusion matrices
cm_ffn = confusion_matrix(y_true_ffn, y_pred_ffn)
cm_cnn = confusion_matrix(y_true_cnn, y_pred_cnn)

plt.figure(figsize=(8,6))
sns.heatmap(cm_ffn, annot=True, fmt='d')
plt.title('FFN Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.savefig('ffn_confusion.png')

plt.figure(figsize=(8,6))
sns.heatmap(cm_cnn, annot=True, fmt='d')
plt.title('CNN Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.savefig('cnn_confusion.png')


