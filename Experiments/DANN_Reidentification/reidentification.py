import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
torch.manual_seed(0)

def pairwise_cosine_similarity(features):
    normed = F.normalize(features, p=2, dim=-1)
    return torch.matmul(normed, normed.t())

# Feature extractor (Siamese CNN)
class FeatureExtractor(nn.Module):
    def __init__(self):
        super(FeatureExtractor, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, padding=3)  # (7x7 filter)  # output size: 32x32
        self.conv2 = nn.Conv2d(64, 128, kernel_size=5, padding=2)  # (5x5 filter) # output size: 16x16
        self.pool = nn.MaxPool2d(2, 2)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)  # (3x3 filter) # output size: 8x8
        self.conv4 = nn.Conv2d(256, 512, kernel_size=3, padding=1)


        self.relu = nn.Tanh()
        self.fc = nn.Linear(512 * 32 * 12, 500)  # Assuming input image size is 32x32
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.relu(self.conv2(x))
        x = self.pool(self.relu(self.conv3(x)))
        x = self.dropout(x)
        x = self.relu(self.conv4(x))
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        # print(x.size())
        return x  # 500-dimensional descriptor
    


# Gradient reversal layer
class GradReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)
    
    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.alpha, None

# Domain classifier
class DomainClassifier(nn.Module):
    def __init__(self):
        super(DomainClassifier, self).__init__()
        self.fc1 = nn.Linear(500, 500)
        self.fc2 = nn.Linear(500, 1)

    def forward(self, x, alpha=10):
        x = GradReverse.apply(x, alpha)
        # x = F.relu(self.fc1(x))
        x = F.tanh(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))
        return x

# Descriptor predictor (used for verification loss)
class DescriptorPredictor(nn.Module):
    def __init__(self):
        super(DescriptorPredictor, self).__init__()
        self.fc = nn.Linear(500, 1)

    def forward(self, x):
        return self.fc(x)
    
# load images in cam_a and cam_b
def load_images1():
    import os
    from PIL import Image
    import numpy as np

    path_cam_a = 'Reidentification_Datasets/PRIM/cam_a'
    path_cam_b = 'Reidentification_Datasets/PRIM/cam_b'
    images_a = []
    images_b = []

    for filename in os.listdir(path_cam_a):
        if filename.endswith('.png') and filename.startswith('img'):
            img_path = os.path.join(path_cam_a, filename)
            img = Image.open(img_path).convert('RGB')
            img = img.resize((48, 128))
            images_a.append(np.array(img))
    
    for filename in os.listdir(path_cam_b):
        if filename.endswith('.png') and filename.startswith('img'):
            img_path = os.path.join(path_cam_b, filename)
            img = Image.open(img_path).convert('RGB')
            img = img.resize((48, 128))
            images_b.append(np.array(img))
    
    images_a = np.array(images_a)
    images_b = np.array(images_b)

    return images_a, images_b

def load_images2():
    import os
    from PIL import Image
    import numpy as np

    path_cam_a = 'Reidentification_Datasets/VIPeR/cam_a'
    path_cam_b = 'Reidentification_Datasets/VIPeR/cam_b'
    images_a = []
    images_b = []

    for filename in os.listdir(path_cam_a):
        if filename.endswith('.bmp'):
            img_path = os.path.join(path_cam_a, filename)
            img = Image.open(img_path).convert('RGB')
            # img = img.resize((32, 32))
            images_a.append(np.array(img))
    
    for filename in os.listdir(path_cam_b):
        if filename.endswith('.bmp'):
            img_path = os.path.join(path_cam_b, filename)
            img = Image.open(img_path).convert('RGB')
            # img = img.resize((32, 32))
            images_b.append(np.array(img))
    
    images_a = np.array(images_a)
    images_b = np.array(images_b)

    return images_a, images_b

source_cam1, source_cam2 = load_images1()

target_cam1, target_cam2 = load_images2()

source_cam1_dmy = torch.tensor(source_cam1, dtype=torch.float32)
source_cam2_dmy = torch.tensor(source_cam2, dtype=torch.float32)
target_cam1_dmy = torch.tensor(target_cam1, dtype=torch.float32)
target_cam2_dmy = torch.tensor(target_cam2, dtype=torch.float32)


def ideal_dim(x):
    """
    Ideal dimension for the input images
    """
    import numpy as np
    x2 = np.array(x)
    a = x2.shape[0]
    b = x2.shape[1]
    c = x2.shape[2]
    d = x2.shape[3]
    src_img = np.zeros((a, d, b, c))
    for img in range(a):
        for i in range(d):
            for j in range(b):
                for k in range(c):
                    src_img[img][i][j][k] = x2[img][j][k][i]
    
    # convert to tensor
    src_img = torch.tensor(src_img, dtype=torch.float32)
    return src_img

source_cam1 = ideal_dim(source_cam1_dmy)
source_cam2 = ideal_dim(source_cam2_dmy)    
target_cam1 = ideal_dim(target_cam1_dmy)
target_cam2 = ideal_dim(target_cam2_dmy)

print(source_cam1.shape)
print(source_cam2.shape)
print(target_cam1.shape)
print(target_cam2.shape)

source_img = torch.cat((source_cam1, source_cam2), dim=0)
target_img = torch.cat((target_cam1, target_cam2), dim=0)
source_label = torch.zeros(source_img.shape[0]) 
for i in range(source_cam1.shape[0]):
    source_label[i] = i
    source_label[i+source_cam1.shape[0]] = i
target_label = torch.ones(target_img.shape[0])
for i in range(target_cam1.shape[0]):
    target_label[i] = i
    target_label[i+target_cam1.shape[0]] = i

source_loader = list(zip(source_img, source_label))
target_loader = list(zip(target_img, target_label))

num_epochs = 20

import torch.optim.sgd
from torch.utils.data import DataLoader, TensorDataset

# Create DataLoader objects with batch size 128
source_dataset = TensorDataset(source_img,source_label)  # Assuming source_cam1 and source_cam2 are tensors
target_dataset = TensorDataset(target_img,target_label)  # Assuming target_cam1 and target_cam2 are tensors

source_loader = DataLoader(source_dataset, batch_size=128, shuffle=True)
target_loader = DataLoader(target_dataset, batch_size=128, shuffle=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# feature_extractor = FeatureExtractor().to(device)
# domain_classifier = DomainClassifier().to(device)
# descriptor_predictor = DescriptorPredictor().to(device)

# load the model
feature_extractor = FeatureExtractor()
# feature_extractor.load_state_dict(torch.load('feature_extractor.pth'))
feature_extractor = feature_extractor.to(device)
domain_classifier = DomainClassifier()
# domain_classifier.load_state_dict(torch.load('domain_classifier.pth'))
domain_classifier = domain_classifier.to(device)
descriptor_predictor = DescriptorPredictor()
# descriptor_predictor.load_state_dict(torch.load('descriptor_predictor.pth'))
descriptor_predictor = descriptor_predictor.to(device)


optimizer = torch.optim.Adam(
    list(feature_extractor.parameters()) +
    list(domain_classifier.parameters()) +
    list(descriptor_predictor.parameters()), lr=0.001
)

from sklearn.metrics import log_loss
def binomial_deviance_loss(similarity_matrix, labels, alpha=2, beta=0.5, c=2):
    """
    Binomial Deviance Loss as in Yi et al. (2014)
    similarity_matrix: cosine similarities (N x N)
    labels: binary label matrix (N x N)
    """
    positive = labels.float()
    negative = 1 - labels.float()
    M = positive - negative

    # # Ensure similarity_matrix and labels have the same shape
    # # if similarity_matrix.size() != labels.size():
    # #     raise ValueError(f"Shape mismatch: similarity_matrix {similarity_matrix.size()} and labels {labels.size()}")

    # # pos_loss = torch.log(1 + torch.exp(-alpha * (similarity_matrix - beta)* positive)) 
    # # neg_loss = torch.log(1 + torch.exp(c * alpha * (similarity_matrix - beta)* negative)) 
    # # loss = (pos_loss + neg_loss).mean()
    # # return loss
    loss = torch.log(1 + torch.exp(-alpha * (similarity_matrix - beta) * M))
    loss = loss.mean()*(loss.size(0))
    return loss
    

for epoch in tqdm(range(num_epochs)):
    print("Epoch: ", epoch)
    dom_loss = 0
    verif_loss = 0
    count = 0
    for (src_imgs, src_labels), (tgt_imgs, _) in zip(source_loader, target_loader):
        src_imgs, tgt_imgs = src_imgs.to(device), tgt_imgs.to(device)
        src_labels = src_labels.to(device)
        # print(src_imgs.shape, tgt_imgs.shape, src_labels.shape)

        # Forward
        src_feats = feature_extractor(src_imgs)
        tgt_feats = feature_extractor(tgt_imgs)

        # Verification loss (source only)
        similarity = pairwise_cosine_similarity(src_feats)
        label_matrix = (src_labels.unsqueeze(1) == src_labels.unsqueeze(0)).float().to(device)

        # print(similarity.shape, label_matrix.shape)


        loss_verif = binomial_deviance_loss(similarity, label_matrix)
        

        # Domain adversarial loss
        src_domain_preds = domain_classifier(src_feats, alpha=2.0)
        tgt_domain_preds = domain_classifier(tgt_feats, alpha=2.0)
        domain_labels = torch.cat([
            torch.ones(src_domain_preds.size(0)),
            torch.zeros(tgt_domain_preds.size(0))
        ]).to(device)
        domain_preds = torch.cat([src_domain_preds, tgt_domain_preds], dim=0)
        loss_domain = F.binary_cross_entropy(domain_preds, domain_labels.unsqueeze(1))

        # Descriptor prediction loss
        # src_descriptor_preds = descriptor_predictor(src_feats)
        # loss_descriptor = F.mse_loss(src_descriptor_preds, src_labels.float().unsqueeze(1))

        # Total loss
        # print(loss_verif, loss_domain)
        loss_total = loss_verif + loss_domain 
        # print(epoch, loss_total.item())


        optimizer.zero_grad()
        loss_total.backward()
        optimizer.step()

        dom_loss += loss_domain.item()
        verif_loss += loss_verif.item()
        count += 1
    dom_loss /= count
    verif_loss /= count
    print(f"Domain Loss: {dom_loss}, Verification Loss: {verif_loss}")
    if epoch % 1000 == 0:
        # save the model every 1000 epochs
        torch.save(feature_extractor.state_dict(), 'feature_extractor.pth')
        torch.save(domain_classifier.state_dict(), 'domain_classifier.pth')
        torch.save(descriptor_predictor.state_dict(), 'descriptor_predictor.pth')



torch.save(feature_extractor.state_dict(), 'feature_extractor.pth')
torch.save(domain_classifier.state_dict(), 'domain_classifier.pth')
torch.save(descriptor_predictor.state_dict(), 'descriptor_predictor.pth')
# # Save the model
# torch.save(feature_extractor.state_dict(), 'pfeature_extractor.pth')

# torch.save(domain_classifier.state_dict(), 'pdomain_classifier.pth')

# torch.save(descriptor_predictor.state_dict(), 'pdescriptor_predictor.pth')

# # Load the model

# feature_extractor.load_state_dict(torch.load('feature_extractor.pth'))

# domain_classifier.load_state_dict(torch.load('domain_classifier.pth'))

# descriptor_predictor.load_state_dict(torch.load('descriptor_predictor.pth'))

# Set the model to evaluation mode
print("Model saved")
# feature_extractor.eval()
# domain_classifier.eval()
# descriptor_predictor.eval()

import numpy as np
import matplotlib.pyplot as plt


# max_idx = np.zeros(target_img.shape[0]//2)
max_idx = np.zeros(source_img.shape[0]//2)
# send max_idx to device
max_idx = torch.tensor(max_idx, dtype=torch.float32).to(device)
# send target_img to device
source_img = source_img.to(device)

features = feature_extractor(source_img)
similarity = pairwise_cosine_similarity(features)

rank_array = []
for i in range(source_img.shape[0]//2):
    target = similarity[i][i+source_img.shape[0]//2]
    rnk = 1
    for j in range(source_img.shape[0]):
        if similarity[i][j] > target:
            rnk += 1
    rank_array.append(rnk)
print(rank_array)

rnk = np.array(rank_array)
print(rnk.mean())


# plot cdf of rnk
plt.hist(rnk, bins=50, cumulative=True, color='blue', alpha=0.5)
plt.xlabel('Rank')
plt.ylabel('Cumulative Frequency')
plt.title('CDF of Rank')
plt.grid()
plt.show()



max_idx = np.zeros(target_img.shape[0]//2)
# send max_idx to device
max_idx = torch.tensor(max_idx, dtype=torch.float32).to(device)
# send target_img to device
target_img = target_img.to(device)

features = feature_extractor(target_img)
similarity = pairwise_cosine_similarity(features)

rank_array = []
for i in range(target_img.shape[0]//2):
    target = similarity[i][i+target_img.shape[0]//2]
    rnk = 1
    for j in range(target_img.shape[0]):
        if similarity[i][j] > target:
            rnk += 1
    rank_array.append(rnk)
print(rank_array)

rnk = np.array(rank_array)
print(rnk.mean())


# plot cdf of rnk
plt.hist(rnk, bins=50, cumulative=True, color='blue', alpha=0.5)
plt.xlabel('Rank')
plt.ylabel('Cumulative Frequency')
plt.title('CDF of Rank')
plt.grid()
plt.show()