import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from ffn import FF_Net
from cnn import Conv_Net
import matplotlib.pyplot as plt

# 1) Prepare one batch of test data
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])
testset = datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)
loader = DataLoader(testset, batch_size=8, shuffle=True)
images, labels = next(iter(loader))

# 2) Load models
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
ffn = FF_Net().to(device)
ffn.load_state_dict(torch.load('ffn.pth', map_location=device))
ffn.eval()

cnn = Conv_Net().to(device)
cnn.load_state_dict(torch.load('cnn.pth', map_location=device))
cnn.eval()

# 3) Inference
imgs = images.to(device)
with torch.no_grad():
    ffn_logits = ffn(imgs.view(imgs.size(0), -1))
    cnn_logits = cnn(imgs)
ffn_preds = ffn_logits.argmax(dim=1).cpu()
cnn_preds = cnn_logits.argmax(dim=1).cpu()

print("True labels:  ", labels.tolist())
print("FFN predicts: ", ffn_preds.tolist())
print("CNN predicts: ", cnn_preds.tolist())

# 4) (Optional) Plot the batch with predictions
classes = testset.classes
fig, axes = plt.subplots(2, 4, figsize=(8,4))
for idx, ax in enumerate(axes.flatten()):
    img = images[idx] * 0.5 + 0.5  # unnormalize
    ax.imshow(img.squeeze(), cmap='gray')
    ax.set_title(f"T:{classes[labels[idx]]}\nF:{classes[ffn_preds[idx]]}\nC:{classes[cnn_preds[idx]]}")
    ax.axis('off')
plt.tight_layout()
plt.show()
