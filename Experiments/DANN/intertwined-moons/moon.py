import matplotlib.pyplot as plt
import torch.nn as nn
import torch.optim as optim
import torch 
import numpy as np
from sklearn.datasets import make_moons
from sklearn.decomposition import PCA
from torch.utils.data import DataLoader, TensorDataset
from itertools import cycle

# ------------------------------
# Data Generation
# ------------------------------

# Save the original source and target data for later visualization.
source_data_original, source_labels = make_moons(n_samples=300, noise=0.1, random_state=42)
target_data_original, _ = make_moons(n_samples=300, noise=0.1, random_state=42)
angle = np.deg2rad(35)
rotation_matrix = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
target_data_original = target_data_original.dot(rotation_matrix.T)


# ------------------------------
# DANN and NN config
# ------------------------------

# Feature extractor: One hidden layer network.
class FeatureExtractor(nn.Module):
    def __init__(self):
        super(FeatureExtractor, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(2, 15),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.layers(x)

# Label classifier: Changed to use LogSoftmax instead of Softmax to match NLLLoss.
class LabelClassifier(nn.Module):
    def __init__(self):
        super(LabelClassifier, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(15, 2),  # 2 classes
            nn.LogSoftmax(dim=1)  # Using log softmax so that NLLLoss receives log-probabilities
        )

    def forward(self, x):
        return self.layers(x)
    
# Gradient reversal layer remains the same.
class GradientReversalFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.lambda_ = lambda_
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.lambda_ * grad_output, None

# Domain classifier: forward requires an alpha parameter.
class DANNDomainClassifier(nn.Module):
    def __init__(self):
        super(DANNDomainClassifier, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(15, 1),
            nn.Sigmoid()
        )

    def forward(self, x, alpha):
        # Pass the alpha value to the gradient reversal layer.
        reversed_features = GradientReversalFunction.apply(x, alpha)
        return self.layers(reversed_features)
    
# non DANN domain classifier    
class DomainClassifier(nn.Module):
    def __init__(self):
        super(DomainClassifier, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(15, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.layers(x)  # No gradient reversal
    
# ------------------------------
# Helper Functions
# ------------------------------

def compute_label_accuracy(feature_extractor, label_predictor, data, labels):
    feature_extractor.eval()
    label_predictor.eval()
    with torch.no_grad():
        features = feature_extractor(torch.tensor(data, dtype=torch.float32))
        preds = label_predictor(features)
        predicted_labels = torch.argmax(preds, dim=1)
        accuracy = (predicted_labels == torch.tensor(labels)).float().mean().item()
    return accuracy


def compute_domain_accuracy_dann(feature_extractor, domain_classifier, source_data, target_data, alpha=1.0):
    feature_extractor.eval()
    domain_classifier.eval()
    with torch.no_grad():
        source_features = feature_extractor(torch.tensor(source_data, dtype=torch.float32))
        target_features = feature_extractor(torch.tensor(target_data, dtype=torch.float32))
        combined_features = torch.cat((source_features, target_features), 0)
        domain_preds = domain_classifier(combined_features, alpha).squeeze()
        domain_preds_binary = (domain_preds >= 0.5).float()
        domain_labels = torch.cat((torch.zeros(len(source_data)), torch.ones(len(target_data))))
        accuracy = (domain_preds_binary == domain_labels).float().mean().item()
    return accuracy

def compute_domain_accuracy(feature_extractor, domain_classifier, source_data, target_data):
    feature_extractor.eval()
    domain_classifier.eval()
    with torch.no_grad():
        source_features = feature_extractor(torch.tensor(source_data, dtype=torch.float32))
        target_features = feature_extractor(torch.tensor(target_data, dtype=torch.float32))
        combined_features = torch.cat((source_features, target_features), 0)
        domain_preds = domain_classifier(combined_features).squeeze()
        domain_preds_binary = (domain_preds >= 0.5).float()
        domain_labels = torch.cat((torch.zeros(len(source_data)), torch.ones(len(target_data))))
        accuracy = (domain_preds_binary == domain_labels).float().mean().item()
    return accuracy

# ------------------------------
# Training Function
# ------------------------------

def train_dann(source_data, source_labels, target_data, num_epochs=10000):
    # Prepare the TensorDatasets using the original numpy arrays.
    # (We use our saved original data for training.)
    source_dataset = TensorDataset(torch.tensor(source_data, dtype=torch.float32), 
                                    torch.tensor(source_labels, dtype=torch.long))
    # For target, we use dummy labels (all zeros) since they are unlabeled.
    target_dataset = TensorDataset(torch.tensor(target_data, dtype=torch.float32), 
                                    torch.zeros(len(target_data), dtype=torch.long))

    source_loader = DataLoader(source_dataset, batch_size=64, shuffle=True)
    target_loader = DataLoader(target_dataset, batch_size=64, shuffle=True)

    # Initialize models.
    feature_extractor = FeatureExtractor()
    label_predictor = LabelClassifier()
    domain_classifier = DANNDomainClassifier()

    # Optimizers for each component.
    optimizer_feat = optim.Adam(feature_extractor.parameters(), lr=0.001)
    optimizer_label = optim.Adam(label_predictor.parameters(), lr=0.001)
    optimizer_domain = optim.Adam(domain_classifier.parameters(), lr=0.001)

    # Losses.
    classification_loss = nn.NLLLoss()  # Now matches the log_softmax output.
    domain_loss = nn.BCELoss()

    # Set a constant value for alpha (the gradient reversal coefficient) as given in the paper
    alpha = 6.0

    # Training loop.
    for epoch in range(num_epochs):
        for (source_batch, source_labels_batch), (target_batch, _) in zip(source_loader, cycle(target_loader)):
            # Forward pass through the feature extractor.
            source_features = feature_extractor(source_batch)
            target_features = feature_extractor(target_batch)

            # ------------------------------
            # Label Prediction Loss (Source Domain)
            # ------------------------------
            label_preds = label_predictor(source_features)
            loss_label = classification_loss(label_preds, source_labels_batch)

            # ------------------------------
            # Domain Prediction Loss (Source + Target)
            # ------------------------------
            # Concatenate features from both domains.
            combined_features = torch.cat((source_features, target_features), 0)
            # Domain labels: 0 for source, 1 for target.
            domain_labels = torch.cat((torch.zeros(len(source_batch)), torch.ones(len(target_batch))), 0)
            # Pass the combined features through the domain classifier with the alpha parameter.
            domain_preds = domain_classifier(combined_features, alpha)
            loss_domain = domain_loss(domain_preds.squeeze(), domain_labels)

            # ------------------------------
            # Backpropagation: Update label predictor and feature extractor on label loss.
            # ------------------------------
            optimizer_feat.zero_grad()
            optimizer_label.zero_grad()
            loss_label.backward(retain_graph=True)
            optimizer_feat.step()
            optimizer_label.step()

            # ------------------------------
            # Backpropagation: Update domain classifier and feature extractor on domain loss.
            # ------------------------------
            optimizer_feat.zero_grad()
            optimizer_domain.zero_grad()
            loss_domain.backward()
            optimizer_feat.step()
            optimizer_domain.step()

        if epoch % 100 == 0:
            print(f"Epoch [{epoch}/10000] - Label Loss: {loss_label.item():.4f}, Domain Loss: {loss_domain.item():.4f}")
    
    domain_accuracy = compute_domain_accuracy_dann(feature_extractor, domain_classifier, source_data, target_data, alpha)
    label_accuracy = compute_label_accuracy(feature_extractor, label_predictor, source_data, source_labels)

    print(f"Final Domain Accuracy: {domain_accuracy:.4f}")
    print(f"Final Label Accuracy: {label_accuracy:.4f}")

    return feature_extractor, label_predictor, domain_classifier
# ------------------------------

def visualise_dann(feature_extractor, label_predictor, domain_classifier, source_data, source_labels, target_data, alpha=1.0):

    # ------------------------------
    # Decision Boundary Visualization( label classification)
    # ------------------------------
    plt.figure(figsize=(8, 6))
    # Create a grid of points.
    xx, yy = np.meshgrid(np.linspace(-2, 3, 1000), np.linspace(-1.5, 2, 1000))
    grid = torch.tensor(np.c_[xx.ravel(), yy.ravel()], dtype=torch.float32)
    # Get predictions from the label predictor.
    preds = label_predictor(feature_extractor(grid)).detach().numpy()
    Z = np.argmax(preds, axis=1).reshape(xx.shape)
    plt.contourf(xx, yy, Z, alpha=0.5)
    # Use the original source data for scatter plot.
    plt.scatter(source_data[:, 0], source_data[:, 1], c=source_labels, edgecolor='k')
    plt.scatter(target_data[:, 0], target_data[:, 1], 
            c='lightgray', edgecolor='black', marker='o', label='Target Data', alpha=0.7)


    plt.title('Decision Boundary (DANN)')
    plt.savefig('dann_decision_boundary.png')
    plt.show()

    # ------------------------------
    # PCA Embeddings Visualization
    # ------------------------------
    # Get features from the feature extractor for both source and target.
    all_data = np.vstack([source_data, target_data])
    features = feature_extractor(torch.tensor(all_data, dtype=torch.float32)).detach().numpy()
    pca = PCA(n_components=2)
    embeddings = pca.fit_transform(features)
    # Determine splitting index.
    split_idx = len(source_data)
    plt.scatter(embeddings[:split_idx, 0], embeddings[:split_idx, 1], c='blue', label='Source')
    plt.scatter(embeddings[split_idx:, 0], embeddings[split_idx:, 1], c='orange', label='Target')
    plt.legend()
    plt.title('Feature Embeddings (PCA) for DANN')
    plt.savefig('dann_pca_embeddings.png')
    plt.show()



    # Decision boundary visualization (Domain Classification)
    plt.figure(figsize=(6, 6))
    domain_preds = domain_classifier(feature_extractor(grid), alpha).detach().numpy().reshape(xx.shape)
    plt.contourf(xx, yy, domain_preds >= 0.5, alpha=0.5, cmap='coolwarm')
    plt.scatter(source_data[:, 0], source_data[:, 1], c=source_labels, edgecolor='k', label='Source')
    plt.scatter(target_data[:, 0], target_data[:, 1], c='orange', edgecolor='k', label='Target')
    plt.title('Decision Boundary (Domain Classification) for DANN')
    plt.legend()
    plt.savefig('dann_domain_decision_boundary.png')
    plt.show()


def train(source_data, source_labels, target_data, num_epochs=10000):
    # Prepare the TensorDatasets using the original numpy arrays.
    # (We use our saved original data for training.)
    source_dataset = TensorDataset(torch.tensor(source_data, dtype=torch.float32), 
                                   torch.tensor(source_labels, dtype=torch.long))
    # For target, we use dummy labels (all zeros) since they are unlabeled.
    target_dataset = TensorDataset(torch.tensor(target_data, dtype=torch.float32), 
                                   torch.zeros(len(target_data), dtype=torch.long))
    
    source_loader = DataLoader(source_dataset, batch_size=64, shuffle=True)
    target_loader = DataLoader(target_dataset, batch_size=64, shuffle=True)

    # Initialize models.
    feature_extractor = FeatureExtractor()
    label_predictor = LabelClassifier()
    domain_classifier = DomainClassifier()

    # Optimizers for each component.
    optimizer_feat = optim.Adam(feature_extractor.parameters(), lr=0.001)
    optimizer_label = optim.Adam(label_predictor.parameters(), lr=0.001)
    optimizer_domain = optim.Adam(domain_classifier.parameters(), lr=0.001)

    # Losses.
    classification_loss = nn.NLLLoss()  # Now matches the log_softmax output.
    domain_loss = nn.BCELoss()

    # Training loop.
    for epoch in range(num_epochs):
        for (source_batch, source_labels_batch), (target_batch, _) in zip(source_loader, cycle(target_loader)):
            # Forward pass through the feature extractor.
            source_features = feature_extractor(source_batch)
            target_features = feature_extractor(target_batch)

            # ------------------------------
            # Label Prediction Loss (Source Domain)
            # ------------------------------
            label_preds = label_predictor(source_features)
            loss_label = classification_loss(label_preds, source_labels_batch)

            # ------------------------------
            # Domain Prediction Loss (Source + Target)
            # ------------------------------
            # Concatenate features from both domains.
            combined_features = torch.cat((source_features, target_features), 0)
            # Domain labels: 0 for source, 1 for target.
            domain_labels = torch.cat((torch.zeros(len(source_batch)), torch.ones(len(target_batch))), 0)
            # Pass the combined features through the domain classifier with the alpha parameter.
            domain_preds = domain_classifier(combined_features)
            loss_domain = domain_loss(domain_preds.squeeze(), domain_labels)

            # ------------------------------
            # Backpropagation: Update label predictor and feature extractor on label loss.
            # ------------------------------
            optimizer_feat.zero_grad()
            optimizer_label.zero_grad()
            loss_label.backward(retain_graph=True)
            optimizer_feat.step()
            optimizer_label.step()

            # ------------------------------
            # Backpropagation: Update domain classifier and feature extractor on domain loss.
            # ------------------------------
            optimizer_feat.zero_grad()
            optimizer_domain.zero_grad()
            loss_domain.backward()
            optimizer_feat.step()
            optimizer_domain.step()

        if epoch % 100 == 0:
            print(f"Epoch [{epoch}/10000] - Label Loss: {loss_label.item():.4f}, Domain Loss: {loss_domain.item():.4f}")

    domain_accuracy = compute_domain_accuracy(feature_extractor, domain_classifier, source_data, target_data)
    label_accuracy = compute_label_accuracy(feature_extractor, label_predictor, source_data, source_labels)

    print(f"Final Domain Accuracy: {domain_accuracy:.4f}")
    print(f"Final Label Accuracy: {label_accuracy:.4f}")

    return feature_extractor, label_predictor, domain_classifier

def visualise(feature_extractor, label_predictor, domain_classifier, source_data, source_labels, target_data):
    # ------------------------------
    # Decision Boundary Visualization( label classification)
    # ------------------------------
    plt.figure(figsize=(8, 6))
    # Create a grid of points.
    xx, yy = np.meshgrid(np.linspace(-2, 3, 1000), np.linspace(-1.5, 2, 1000))
    grid = torch.tensor(np.c_[xx.ravel(), yy.ravel()], dtype=torch.float32)
    # Get predictions from the label predictor.
    preds = label_predictor(feature_extractor(grid)).detach().numpy()
    Z = np.argmax(preds, axis=1).reshape(xx.shape)
    plt.contourf(xx, yy, Z, alpha=0.5)
    # Use the original source data for scatter plot.
    plt.scatter(source_data[:, 0], source_data[:, 1], c=source_labels, edgecolor='k')
    plt.scatter(target_data[:, 0], target_data[:, 1], 
            c='lightgray', edgecolor='black', marker='o', label='Target Data', alpha=0.7)


    plt.title('Decision Boundary for Non-DANN')
    plt.savefig('non_dann_decision_boundary.png')
    plt.show()

    # ------------------------------
    # PCA Embeddings Visualization
    # ------------------------------
    # Get features from the feature extractor for both source and target.
    all_data = np.vstack([source_data, target_data])
    features = feature_extractor(torch.tensor(all_data, dtype=torch.float32)).detach().numpy()
    pca = PCA(n_components=2)
    embeddings = pca.fit_transform(features)
    # Determine splitting index.
    split_idx = len(source_data)
    plt.scatter(embeddings[:split_idx, 0], embeddings[:split_idx, 1], c='blue', label='Source')
    plt.scatter(embeddings[split_idx:, 0], embeddings[split_idx:, 1], c='orange', label='Target')
    plt.legend()
    plt.title('Feature Embeddings (PCA) for Non-DANN')
    plt.savefig('non_dann_pca_embeddings.png')
    plt.show()

    
    
    # Decision boundary visualization (Domain Classification)
    plt.figure(figsize=(6, 6))
    domain_preds = domain_classifier(feature_extractor(grid)).detach().numpy().reshape(xx.shape)
    plt.contourf(xx, yy, domain_preds >= 0.5, alpha=0.5, cmap='coolwarm')
    plt.scatter(source_data[:, 0], source_data[:, 1], c=source_labels, edgecolor='k', label='Source')
    plt.scatter(target_data[:, 0], target_data[:, 1], c='orange', edgecolor='k', label='Target')
    plt.title('Decision Boundary (Domain Classification) for Non-DANN')
    plt.legend()
    plt.savefig('non_dann_domain_decision_boundary.png')
    plt.show()


# ------------------------------
# Main function to run the training and visualization.
# ------------------------------

if __name__ == "__main__":
    NUM_EPOCHS = 10000   #set as per your requirement
    # Train the DANN model.

    feature_extractor, label_predictor, domain_classifier = train_dann(source_data_original, source_labels, target_data_original, NUM_EPOCHS)

    # Visualize the results.
    visualise_dann(feature_extractor, label_predictor, domain_classifier, source_data_original, source_labels, target_data_original, alpha = 6.0)

    # Train the non-DANN model.
    feature_extractor, label_predictor, domain_classifier = train(source_data_original, source_labels, target_data_original, NUM_EPOCHS)

    # Visualize the results.
    visualise(feature_extractor, label_predictor, domain_classifier, source_data_original, source_labels, target_data_original)
