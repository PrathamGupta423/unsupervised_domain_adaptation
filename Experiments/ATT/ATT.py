import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.preprocessing import OneHotEncoder
from scipy.io import loadmat
from skimage.transform import resize
import sklearn
import pprint

import matplotlib.pyplot as plt

from torch.utils.data import Dataset , DataLoader

import torch
import torch.nn as nn
import torch.nn.functional as F

import os

from tqdm import tqdm

import seaborn as sns
import matplotlib.pyplot as plt

import sys
import logging
from functools import wraps
from datetime import datetime

import pickle


# Set random seed for reproducibility
torch.manual_seed(23626)

current_dir = os.path.dirname(os.path.abspath(__file__))
log_file_path = os.path.join(current_dir, 'print_logs.log')
# Logs

logging.basicConfig(

    filename=log_file_path,
    level=logging.INFO,
    format='%(asctime)s - %(message)s',
)

def log_print(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        message = ' '.join(map(str, args))
        logging.info(message)
        return func(*args, **kwargs)
    return wrapper

# Decorate the built-in print function
print = log_print(print)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# Dataset Paths 
path_svhn_train = '../datasets/svhn/train_32x32.mat' 
path_svhn_test = '../datasets/svhn/test_32x32.mat'

# Utility Functions
def dense_to_one_hot_svhn(labels_dense, num_classes=10):
    labels_dense = np.where(labels_dense == 10, 0, labels_dense)
    encoder = OneHotEncoder(sparse_output=False, categories='auto')
    return encoder.fit_transform(labels_dense.reshape(-1, 1))

def dense_to_one_hot(labels_dense, num_classes=10):
    if int(sklearn.__version__.split(".")[1]) >= 2:
        encoder = OneHotEncoder(sparse_output=False, dtype=np.float32)
    else:
        encoder = OneHotEncoder(sparse_output=False, dtype=np.int32)
    return encoder.fit_transform(labels_dense.reshape(-1, 1))

def return_svhn(path_train, path_test):
    svhn_train = loadmat(path_train)
    svhn_test = loadmat(path_test)
    svhn_train_im = svhn_train['X'].transpose(3, 0, 1, 2)
    svhn_test_im = svhn_test['X'].transpose(3, 0, 1, 2)
    svhn_label = dense_to_one_hot_svhn(svhn_train['y'])
    svhn_label_test = dense_to_one_hot_svhn(svhn_test['y'])
    return svhn_train_im, svhn_test_im, svhn_label, svhn_label_test

def visualize_samples(data, labels, num_samples=5):
    plt.figure(figsize=(10, 2))
    random_indices = np.random.choice(len(data), num_samples, replace=False)
    data = data[random_indices]
    labels = labels[random_indices]
    for i in range(num_samples):
        plt.subplot(1, num_samples, i + 1)
        plt.imshow(data[i])
        plt.title(np.argmax(labels[i]))
        plt.axis('off')
    plt.show()

def return_mnist():
    mnist = fetch_openml('mnist_784', version=1, as_frame=False)
    X = mnist['data'].reshape(-1, 28, 28).astype(np.float32) / 255.
    y = mnist['target'].astype(int)
    
    # Split
    X_train, X_test = X[:60000], X[60000:]
    y_train, y_test = y[:60000], y[60000:]
    
    # Resize to (32, 32, 3)
    X_train = np.stack([resize(x, (32, 32)) for x in X_train])
    X_test = np.stack([resize(x, (32, 32)) for x in X_test])

    # Add channel dimension and duplicate to make 3 channels
    X_train = np.repeat(X_train[..., np.newaxis], 3, axis=-1)
    X_test = np.repeat(X_test[..., np.newaxis], 3, axis=-1)

    y_train = dense_to_one_hot(y_train)
    y_test = dense_to_one_hot(y_test)

    return X_train, X_test, y_train, y_test


# Loading the SVHN dataset
data_t_im, data_t_im_test, data_t_label, data_t_label_test = return_svhn(path_svhn_train, path_svhn_test)

data_s_im, data_s_im_test, data_s_label, data_s_label_test = return_mnist()


# Making the dataset and dataloader

class SVHNDataset(Dataset):
    def __init__(self, data, labels, transform=None):
        # Convert to float tensor and permute dimensions
        self.data = torch.tensor(data, dtype=torch.float32).permute(0, 3, 1, 2) / 255.0
        # Convert one-hot to label index
        self.labels = torch.tensor(np.argmax(labels, axis=1), dtype=torch.long)
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = self.data[idx]
        y = self.labels[idx]
        if self.transform:
            x = self.transform(x)
        return x, y

class MNISTDataset(Dataset):
    def __init__(self, data, labels, transform=None):
        # Convert to float tensor and permute dimensions from (N, H, W, C) → (N, C, H, W)
        self.data = torch.tensor(data, dtype=torch.float32).permute(0, 3, 1, 2)
        # print(self.data[0])
        # print to txt
        # print(self.data[0].numpy().tolist())
        # with open('data.txt', 'w') as f:
        #     f.write(str(self.data[0].numpy().tolist()))

        # Convert one-hot labels to integer class labels
        self.labels = torch.tensor(np.argmax(labels, axis=1), dtype=torch.long)
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = self.data[idx]
        y = self.labels[idx]
        if self.transform:
            x = self.transform(x)
        return x, y

svhn_train = SVHNDataset(data_t_im, data_t_label)
svhn_test = SVHNDataset(data_t_im_test, data_t_label_test)

mnist_train = MNISTDataset(data_s_im, data_s_label)
mnist_test = MNISTDataset(data_s_im_test, data_s_label_test)

batch_size = 128
svhn_train_loader = DataLoader(svhn_train, batch_size=batch_size, shuffle=True)
svhn_test_loader = DataLoader(svhn_test, batch_size=batch_size, shuffle=False)
mnist_train_loader = DataLoader(mnist_train, batch_size=batch_size, shuffle=True)
mnist_test_loader = DataLoader(mnist_test, batch_size=batch_size, shuffle=False)

print(f"SVHN train: {len(svhn_train)}, test: {len(svhn_test)}")
print(f"MNIST train: {len(mnist_train)}, test: {len(mnist_test)}")


# Neural Network Architecture

class FeatureExtractor(nn.Module):
    def __init__(self, use_bn=False, use_dropout=False, dropout_rate=0.5):
        super(FeatureExtractor, self).__init__()
        # self.conv_block = nn.Sequential(
        #     nn.Conv2d(3, 64, kernel_size=5, padding=2),
        #     # print(f"shape: {self.conv_block[0].weight.shape}"),
        #     nn.ReLU(),
        #     nn.MaxPool2d(kernel_size=3, stride=2),
        #     # print(f"shape: {self.conv_block[1].weight.shape}"),

        #     nn.Conv2d(64, 64, kernel_size=5, padding=2),
        #     # print(f"shape: {self.conv_block[2].weight.shape}"),
        #     nn.ReLU(),
        #     nn.MaxPool2d(kernel_size=3, stride=2),
        #     # print(f"shape: {self.conv_block[3].weight.shape}"),

        #     nn.Conv2d(64, 128, kernel_size=5, padding=2),
        #     # print(f"shape: {self.conv_block[4].weight.shape}"),
        #     nn.ReLU()
        # )

        layers = [
            nn.Conv2d(3, 64, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),

            nn.Conv2d(64, 64, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),

            nn.Conv2d(64, 128, kernel_size=5, padding=2),
            nn.ReLU()
        ]
        if use_bn:
            layers.append(nn.BatchNorm2d(128))

        layers.append(nn.ReLU())

        self.conv_block = nn.Sequential(*layers)

        # self.fc = nn.Sequential(
        #     nn.Flatten(),
        #     nn.Linear(128 * 7 * 7, 3072),  # assuming 32x32 input size
        #     nn.ReLU()
        # )
        fc_layers = [
            nn.Flatten(),
            nn.Linear(128 * 7 * 7, 3072),  # assuming 32x32 input size
            nn.ReLU()
        ]
        if use_dropout:
            fc_layers.append(nn.Dropout(dropout_rate))
        self.fc = nn.Sequential(*fc_layers)


    def forward(self, x):
        x = self.conv_block(x)
        x = self.fc(x)
        return x

class Classifier(nn.Module):
    def __init__(self, input_dim=3072, hidden_dim=2048, num_classes=10 , use_bn=False , use_dropout=False, dropout_rate=0.5):
        super(Classifier, self).__init__()
        # self.classifier = nn.Sequential(
        #     nn.Linear(input_dim, hidden_dim),
        #     nn.ReLU(),
        #     nn.Linear(hidden_dim, num_classes)
        # )
        layers = [
            nn.Linear(input_dim, hidden_dim),
        ]
        if use_bn:
            layers.append(nn.BatchNorm1d(hidden_dim))
        layers.append(nn.ReLU())
        if use_dropout:
            layers.append(nn.Dropout(dropout_rate))
        layers.append(nn.Linear(hidden_dim, num_classes))
        self.classifier = nn.Sequential(*layers)


    def forward(self, x):
        return self.classifier(x)
    

# Training Utility Functions

def labeling(F, F1, F2, data_loader, Nt, threshold=0.95):
    F.eval(), F1.eval(), F2.eval()
    pseudo_x, pseudo_y = [], []

    with torch.no_grad():
        for x, _ in data_loader:
            x = x.to(device)
            features = F(x)
            out1 = F1(features)
            out2 = F2(features)

            prob1 = torch.softmax(out1, dim=1)
            prob2 = torch.softmax(out2, dim=1)
            pred1 = prob1.argmax(dim=1)
            pred2 = prob2.argmax(dim=1)

            agree = pred1 == pred2
            confident = (prob1.max(dim=1)[0] > threshold) | (prob2.max(dim=1)[0] > threshold)

            mask = agree & confident

            if mask.any():
                pseudo_x.append(x[mask].cpu())
                pseudo_y.append(pred1[mask].cpu())

    if len(pseudo_x) == 0:
        return torch.empty(0), torch.empty(0)

    pseudo_x = torch.cat(pseudo_x, dim=0)
    pseudo_y = torch.cat(pseudo_y, dim=0)


    if Nt < len(pseudo_x):
        indices = torch.randperm(len(pseudo_x))[:Nt]
        pseudo_x = pseudo_x[indices]
        pseudo_y = pseudo_y[indices]

    # pseudo_x = pseudo_x.to(device)
    # pseudo_y = pseudo_y.to(device)
    return pseudo_x, pseudo_y

def cross_view_penalty(F1, F2):
    # Extract last Linear layer weights from F1 and F2
    W1 = F1.classifier[0].weight  
    W2 = F2.classifier[0].weight
    # Compute Frobenius norm of W1^T W2
    return torch.norm(torch.mm(W1, W2.T), p='fro')


# Training Classes 

class DomainAdaptationModel():
    def __init__(self, Source_train_ds, Source_train_loader, Target_train_ds, Target_train_loader, Source_test_loader, Target_test_loader, name, use_bn = False, use_dropout=False, dropout_rate = 0.5, num_classes=10 , num_iters=10,K=10 , cross_view_penalty_weight=0.01):
        self.Source_train_ds = Source_train_ds
        self.Target_train_ds = Target_train_ds
        self.Source_train_loader = Source_train_loader
        self.Target_train_loader = Target_train_loader
        self.Source_test_loader = Source_test_loader
        self.Target_test_loader = Target_test_loader

        self.num_classes = num_classes

        # Initialize models
        self.F = FeatureExtractor(use_bn=use_bn, use_dropout=use_dropout, dropout_rate=dropout_rate).to(device)
        self.F1 = Classifier(use_bn=use_bn, use_dropout=use_dropout, dropout_rate=dropout_rate).to(device)
        self.F2 = Classifier(use_bn=use_bn, use_dropout=use_dropout, dropout_rate=dropout_rate).to(device)
        self.Ft = Classifier(use_bn=False, use_dropout=use_dropout, dropout_rate=dropout_rate).to(device)

        # Optimizers
        self.optimizer_F = optim.SGD(self.F.parameters(), lr=0.01, momentum=0.9)
        self.optimizer_F1 = optim.SGD(self.F1.parameters(), lr=0.01, momentum=0.9)
        self.optimizer_F2 = optim.SGD(self.F2.parameters(), lr=0.01, momentum=0.9)
        self.optimizer_Ft = optim.SGD(self.Ft.parameters(), lr=0.01, momentum=0.9)
        # Loss function
        self.criterion = nn.CrossEntropyLoss()
        # Threshold for pseudo-labeling
        self.threshold = 0.95
        # Cross-view penalty weight
        self.cross_view_penalty_weight = cross_view_penalty_weight

        # Storage Location
        current_dir = os.path.dirname(os.path.abspath(__file__))
        self.name = name
        self.path = os.path.join(current_dir, self.name)
        if not os.path.exists(self.path):
            os.makedirs(self.path)

        self.num_iters = num_iters
        self.K = K

    def train(self, Nt=1000):
        # Intial Training
        num_iters = self.num_iters
        self.metrics = []
        if os.path.exists(os.path.join(self.path, 'F.pth')):
            print("Loading existing models...")
            self.F.load_state_dict(torch.load(os.path.join(self.path, 'F.pth')))
            self.F1.load_state_dict(torch.load(os.path.join(self.path, 'F1.pth')))
            self.F2.load_state_dict(torch.load(os.path.join(self.path, 'F2.pth')))
            self.Ft.load_state_dict(torch.load(os.path.join(self.path, 'Ft.pth')))
        else:
            print("Training models from scratch...")
            
            for epoch in tqdm(range(num_iters)):
                for xs, ys in self.Source_train_loader:
                    xs, ys = xs.to(device), ys.to(device)
                    feats = self.F(xs)

                    out1 = self.F1(feats)
                    out2 = self.F2(feats)
                    loss1 =self.criterion(out1, ys)
                    loss2 =self.criterion(out2, ys)

                    # Cross-view constraint ||W1^T W2||
                    W1 = self.F1.classifier[0].weight  # adjust if different layer name
                    W2 = self.F2.classifier[0].weight
                    cross_loss = self.cross_view_penalty_weight * torch.norm(torch.mm(W1.T, W2))

                    loss_total = loss1 + loss2 + cross_loss

                    self.optimizer_F1.zero_grad()
                    self.optimizer_F2.zero_grad()
                    self.optimizer_F.zero_grad()

                    loss_total.backward()
                    self.optimizer_F1.step()
                    self.optimizer_F2.step()
                    self.optimizer_F.step()
                    # Print loss for monitoring
                # if epoch % 1 == 0:
                print(f"Epoch [{epoch+1}/{num_iters}], Loss1: {loss1.item():.4f}, Loss2: {loss2.item():.4f}, Cross Loss: {cross_loss.item():.4f}")
                Total_loss = loss1 + loss2 + cross_loss
                # metrics = self.test(tag = 'Initial', u = epoch)
                # self.metrics.append(metrics)
                print(f"Total Loss: {Total_loss.item():.4f}")
                # print(f"Accuracy: {metrics['Accuracy']:.4f}")
                torch.save(self.F.state_dict(), os.path.join(self.path, 'F.pth'))
                torch.save(self.F1.state_dict(), os.path.join(self.path, 'F1.pth'))
                torch.save(self.F2.state_dict(), os.path.join(self.path, 'F2.pth'))
                torch.save(self.Ft.state_dict(), os.path.join(self.path, 'Ft.pth'))
                print("Models saved.")
                torch.save(self.F.state_dict(), os.path.join(self.path, f'F_clean_{num_iters}.pth'))
                torch.save(self.F1.state_dict(), os.path.join(self.path, f'F1_clean_{num_iters}.pth'))
                torch.save(self.F2.state_dict(), os.path.join(self.path, f'F2_clean_{num_iters}.pth'))
                torch.save(self.Ft.state_dict(), os.path.join(self.path, f'Ft_clean_{num_iters}.pth'))
                print("Models saved.")


        # Pseudo-labeling
        self.N_t = 1000
        Xt_l_x , Xt_l_y = labeling(self.F, self.F1, self.F2, self.Target_train_loader, Nt=self.N_t, threshold=self.threshold)

        Xt_l_loader = DataLoader(list(zip(Xt_l_x, Xt_l_y)), batch_size=batch_size, shuffle=True)

        # L_combined = list(self.Source_train_ds) + list(zip(Xt_l_x, Xt_l_y))
        # for x,y in L_combined:
        #     x = x.to(device)
        #     y = y.to(device)

        L_loader = DataLoader(
            list(self.Source_train_ds) + list(zip(Xt_l_x, Xt_l_y)),
            batch_size=batch_size,
            shuffle=True
        )

        K = self.K
        for k in tqdm(range(K)):
            for epoch in range(num_iters):
                for (xL, yL) , (xtl, ytl)  in zip(L_loader,Xt_l_loader):
                    xL, yL = xL.to(device), yL.to(device)
                    xtl, ytl = xtl.to(device), ytl.to(device)
                
                    featsL = self.F(xL)
                    out1 = self.F1(featsL)
                    out2 = self.F2(featsL)

                    loss1 = self.criterion(out1, yL)
                    loss2 = self.criterion(out2, yL)
                    W1 = self.F1.classifier[0].weight  # adjust if different layer name
                    W2 = self.F2.classifier[0].weight
                    cross_loss =  torch.norm(torch.mm(W1.T, W2)) * self.cross_view_penalty_weight

                    total_loss = loss1 + loss2 + cross_loss

                    self.optimizer_F1.zero_grad()
                    self.optimizer_F2.zero_grad()
                    self.optimizer_F.zero_grad()
                    total_loss.backward()
                    self.optimizer_F1.step()
                    self.optimizer_F2.step()
                    self.optimizer_F.step()

                    featsT = self.F(xtl)
                    outt = self.Ft(featsT)
                    lossFt = self.criterion(outt, ytl)

                    self.optimizer_Ft.zero_grad()
                    self.optimizer_F.zero_grad()
                    lossFt.backward()
                    self.optimizer_Ft.step()
                    self.optimizer_F.step()

            Nt = int((k + 1) * len(self.Target_train_ds) / K)
            Xt_l_x, Xt_l_y = labeling(self.F, self.F1, self.F2, self.Target_train_loader, Nt)
            Xt_l_loader = DataLoader(list(zip(Xt_l_x, Xt_l_y)), batch_size=batch_size, shuffle=True)
            L_combined = list(self.Source_train_ds) + list(zip(Xt_l_x, Xt_l_y))
            L_loader = DataLoader(L_combined, batch_size=batch_size, shuffle=True)

            print(f"Iteration [{k+1}/{K}], Nt: {Nt}, Loss1: {loss1.item():.4f}, Loss2: {loss2.item():.4f}, Cross Loss: {cross_loss.item():.4f}, Ft Loss: {lossFt.item():.4f}")
            Total_loss = loss1 + loss2 + cross_loss + lossFt
            # Total_loss = lossFt
            print(f"Iteration [{k+1}/{K}], Nt: {Nt}, Ft Loss: {lossFt.item():.4f}")

            metrics = self.test(tag = 'Iter', u = k)
            self.metrics.append(metrics)
            print(f"Total Loss: {Total_loss.item():.4f}")
            print(f"Accuracy: {metrics['Accuracy']:.4f}")
            # Save the model
            torch.save(self.F.state_dict(), os.path.join(self.path, 'F.pth'))
            torch.save(self.F1.state_dict(), os.path.join(self.path, 'F1.pth'))
            torch.save(self.F2.state_dict(), os.path.join(self.path, 'F2.pth'))
            torch.save(self.Ft.state_dict(), os.path.join(self.path, 'Ft.pth'))
            print("Models saved.")

        # Save the model
        torch.save(self.F.state_dict(), os.path.join(self.path, 'F.pth'))
        torch.save(self.F1.state_dict(), os.path.join(self.path, 'F1.pth'))
        torch.save(self.F2.state_dict(), os.path.join(self.path, 'F2.pth'))
        torch.save(self.Ft.state_dict(), os.path.join(self.path, 'Ft.pth'))
        print("Models saved.")
        # Save the metrics
        with open(os.path.join(self.path, 'metrics.txt'), 'w') as f:
            for metric in self.metrics:
                f.write(f"{metric}\n")
        print("Metrics saved.")



    def test(self,tag=None , u = None):
        self.confusion_matrix = self.find_confusion_matrix(self.Target_test_loader)
        L = tag
        O = u
        if tag is None:
            L = ''
        if u is None:
            u = ''
        self.plot_confusion_matrix(self.confusion_matrix, classes=range(self.num_classes),
                                   title=f'Confusion Matrix for {self.name} {L} {O}')
        stats = self.stats()
        # save the stats
        with open(os.path.join(self.path, f'stats_{L}_{O}.txt'), 'w') as f:
            for key, value in stats.items():
                f.write(f"{key}: {value}\n")
        # Print stats
        return stats



    def find_confusion_matrix(self, data_loader):
        self.Ft.eval()
        self.F.eval()
        num_classes = 10
        confusion_matrix = torch.zeros(num_classes, num_classes)
        
        for x, y in data_loader:
            x = x.to(device)
            with torch.no_grad():
                features = self.F(x)
                out = self.Ft(features)
                _, predicted = torch.max(out, 1)
                for t, p in zip(y.view(-1), predicted.view(-1)):
                    confusion_matrix[t.long(), p.long()] += 1
        return confusion_matrix
    
    @staticmethod
    def accuracy(confusion_matrix):
        correct = confusion_matrix.diag().sum()
        total = confusion_matrix.sum()
        return correct / total

    @staticmethod
    def precision(confusion_matrix):
        tp = confusion_matrix.diag()
        fp = confusion_matrix.sum(0) - tp
        precision = tp / (tp + fp)
        return precision.nan_to_num()
    
    @staticmethod
    def recall(confusion_matrix):
        tp = confusion_matrix.diag()
        fn = confusion_matrix.sum(1) - tp
        recall = tp / (tp + fn)
        return recall.nan_to_num()
    
    @staticmethod
    def f1_score(confusion_matrix):
        p = DomainAdaptationModel.precision(confusion_matrix)
        r = DomainAdaptationModel.recall(confusion_matrix)
        f1 = 2 * (p * r) / (p + r)
        return f1.nan_to_num()
    
    @staticmethod
    def specificity(confusion_matrix):
        tn = confusion_matrix.sum() - (confusion_matrix.sum(0) + confusion_matrix.sum(1) - confusion_matrix.diag())
        fp = confusion_matrix.sum(0) - confusion_matrix.diag()
        specificity = tn / (tn + fp)
        return specificity.nan_to_num()
    

    def plot_confusion_matrix(self, confusion_matrix, classes, title='confusion matrix'):
        plt.figure(figsize=(10, 8))
        sns.heatmap(confusion_matrix.numpy(), annot=True, fmt='g', cmap='Blues', xticklabels=classes, yticklabels=classes)
        plt.title(title)
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.savefig(os.path.join(self.path, f"{title}.png"))
        # plt.show()
        plt.close()
        

    def stats(self):
        acc = self.accuracy(self.confusion_matrix)
        precision = self.precision(self.confusion_matrix)
        recall = self.recall(self.confusion_matrix)
        f1 = self.f1_score(self.confusion_matrix)
        specificity = self.specificity(self.confusion_matrix)

        stats = {
            'Accuracy': acc,
            'Precision': precision,
            'Recall': recall,
            'F1 Score': f1,
            'Specificity': specificity
        }

        return stats
            
# Example usage
if __name__ == "__main__":

    print("Starting Domain Adaptation...")
    # Initialize the model


    # Initialize the model
    print("Initializing model for SVHN to MNIST with no batch norm, Dropout and cross-view penalty...")
    model = DomainAdaptationModel(
        Source_train_ds=svhn_train,
        Source_train_loader=svhn_train_loader,
        Target_train_ds=mnist_train,
        Target_train_loader=mnist_train_loader,
        Source_test_loader=svhn_test_loader,
        Target_test_loader=mnist_test_loader,
        name='SVHN_MNIST_nbn_ydo',
        use_bn=False,
        use_dropout=True,
        num_classes=10,
        num_iters=20,
        K=100,
        cross_view_penalty_weight=0.01
    )

    # Train the model
    model.train(Nt=1000)
    print("Training completed.")
    # Save the model
    with open(os.path.join(model.path, 'model_SVHN_MNIST_nbn_ydo.pkl'), 'wb') as f:
        pickle.dump(model, f)

    print("Model saved.")

    print("Initializing model for MNIST with no batch norm, Dropout and cross-view penalty...")
    model_mnist = DomainAdaptationModel(
        Source_train_ds=mnist_train,
        Source_train_loader=mnist_train_loader,
        Target_train_ds=svhn_train,
        Target_train_loader=svhn_train_loader,
        Source_test_loader=mnist_test_loader,
        Target_test_loader=svhn_test_loader,
        name='MNIST_SVHN_nbn_ydo',
        use_bn=False,
        use_dropout=True,
        num_classes=10,
        num_iters=20,
        K=100,
        cross_view_penalty_weight=0.01
    )
    # Train the model
    model_mnist.train(Nt=1000)
    print("Training completed.")

    # Save the model
    with open(os.path.join(model_mnist.path, 'model_MNIST_SVHN_nbn_ydo.pkl'), 'wb') as f:
        pickle.dump(model_mnist, f)
    print("Model saved.")

    print("Initializing model for SVHN to MNIST with batch norm, Dropout and cross-view penalty...")
    model = DomainAdaptationModel(
        Source_train_ds=svhn_train,
        Source_train_loader=svhn_train_loader,
        Target_train_ds=mnist_train,
        Target_train_loader=mnist_train_loader,
        Source_test_loader=svhn_test_loader,
        Target_test_loader=mnist_test_loader,
        name='SVHN_MNIST_bn_ydo',
        use_bn=True,
        use_dropout=True,
        num_classes=10,
        num_iters=20,
        K=40,
        cross_view_penalty_weight=0.01
    )
    # Train the model
    model.train(Nt=1000)
    print("Training completed.")
    # Save the model
    with open(os.path.join(model.path, 'model_SVHN_MNIST_bn_ydo.pkl'), 'wb') as f:
        pickle.dump(model, f)

    print("Model saved.")
    print("Initializing model for MNIST to SVHN with batch norm, Dropout and cross-view penalty...")
    model_mnist = DomainAdaptationModel(
        Source_train_ds=mnist_train,
        Source_train_loader=mnist_train_loader,
        Target_train_ds=svhn_train,
        Target_train_loader=svhn_train_loader,
        Source_test_loader=mnist_test_loader,
        Target_test_loader=svhn_test_loader,
        name='MNIST_SVHN_bn_ydo',
        use_bn=True,
        use_dropout=True,
        num_classes=10,
        num_iters=20,
        K=40,
        cross_view_penalty_weight=0.01
    )
    # Train the model
    model_mnist.train(Nt=1000)
    print("Training completed.")
    # Save the model
    with open(os.path.join(model_mnist.path, 'model_MNIST_SVHN_bn_ydo.pkl'), 'wb') as f:
        pickle.dump(model_mnist, f)
    print("Model saved.")

    
    







