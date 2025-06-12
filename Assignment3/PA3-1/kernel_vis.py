import cv2
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import math
from cnn import Conv_Net

# Load pretrained CNN model
dev = torch.device('cpu')
conv_net = Conv_Net()
conv_net.load_state_dict(torch.load('cnn.pth', map_location=dev))
conv_net.eval()

# Helper for robust normalization
def normalize(arr):
    min_val = arr.min()
    max_val = arr.max()
    denom = max_val - min_val
    if denom == 0:
        return np.zeros_like(arr)
    return (arr - min_val) / denom

# 1) Extract first conv layer kernels
# Shape: [out_channels, in_channels, kH, kW]
kernels = conv_net.conv1.weight.data.clone().cpu()
n_kernels = kernels.shape[0]
n_cols = 8
n_rows = math.ceil(n_kernels / n_cols)

# 2) Plot kernels in a grid
fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols, n_rows))
for idx, ax in enumerate(axes.flatten()):
    if idx < n_kernels:
        kernel = kernels[idx, 0, :, :].numpy()
        norm_kernel = normalize(kernel)
        ax.imshow(norm_kernel, cmap='gray')
    ax.axis('off')
plt.tight_layout()
plt.savefig('kernel_grid.png')
plt.close()

# 3) Load and preprocess sample image
img = cv2.imread('sample_image.png', cv2.IMREAD_GRAYSCALE)
img = cv2.resize(img, (28, 28))
img = img.astype(np.float32) / 255.0
img_tensor = torch.from_numpy(img).unsqueeze(0).unsqueeze(0)  # [1,1,28,28]

# 4) Apply kernels via conv2d
output = F.conv2d(img_tensor, kernels, bias=None, stride=1, padding=1)
output = output.squeeze(0).unsqueeze(1)  # [n_kernels,1,H,W]

# 5) Plot transformed images
fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols, n_rows))
for idx, ax in enumerate(axes.flatten()):
    if idx < n_kernels:
        feat = output[idx, 0, :, :].detach().numpy()
        norm_feat = normalize(feat)
        ax.imshow(norm_feat, cmap='gray')
    ax.axis('off')
plt.tight_layout()
plt.savefig('image_transform_grid.png')
plt.close()

# 6) Feature-map progression through conv+relu and pooling layers
layers = []

# conv1 + relu
x = conv_net.conv1(img_tensor)
x1 = F.relu(x)
layers.append(('conv1_relu', x1))
# pool1
x1p = conv_net.pool(x1)
layers.append(('pool1', x1p))
# conv2 + relu
x2 = conv_net.conv2(x1p)
x2a = F.relu(x2)
layers.append(('conv2_relu', x2a))
# pool2
x2p = conv_net.pool(x2a)
layers.append(('pool2', x2p))
# conv3 + relu
x3 = conv_net.conv3(x2p)
x3a = F.relu(x3)
layers.append(('conv3_relu', x3a))
# pool3
x3p = conv_net.pool(x3a)
layers.append(('pool3', x3p))

# Plot all stages (including original)
n_stages = len(layers) + 1
fig, axes = plt.subplots(1, n_stages, figsize=(n_stages * 2, 2))
# original
orig = img_tensor.squeeze().numpy()
axes[0].imshow(orig, cmap='gray')
axes[0].set_title('original')
axes[0].axis('off')

for i, (name, feat) in enumerate(layers, start=1):
    channel = feat[0, 0, :, :].detach().numpy()
    norm_channel = normalize(channel)
    axes[i].imshow(norm_channel, cmap='gray')
    axes[i].set_title(name)
    axes[i].axis('off')

plt.tight_layout()
plt.savefig('feature_progression.png')
plt.close()
