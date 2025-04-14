import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, TensorDataset
from torch.autograd import Function
import numpy as np
import pandas as pd
import random

import os

BOOK_location = 'sentiment_analysis/processed_acl/books'
DVD_location = "sentiment_analysis/processed_acl/dvd"
ELECTRONICS_location = "sentiment_analysis/processed_acl/electronics"
KITCHEN_location = "sentiment_analysis/processed_acl/kitchen"

Source_loc = DVD_location
Target_loc = BOOK_location



class GradientReversal(Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha
        return output, None

def grad_reverse(x, alpha):
    return GradientReversal.apply(x, alpha)
    
class FeatureExtractor(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(FeatureExtractor, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)  # First hidden layer
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)  # Second hidden layer

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x
    
class LabelPredictor(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(LabelPredictor, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.softmax(x)
        return x
    
class DomainClassifier(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(DomainClassifier, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return x

class DANN(nn.Module):
    def __init__(self, feature_extractor, label_predictor, domain_classifier, alpha):
        super(DANN, self).__init__()
        self.feature_extractor = feature_extractor
        self.label_predictor = label_predictor
        self.domain_classifier = domain_classifier
        self.alpha = alpha

    def forward(self, x, reverse=True):
        features = self.feature_extractor(x)
        label_output = self.label_predictor(features)
        if reverse:
            domain_output = self.domain_classifier(grad_reverse(features, self.alpha))
        else:
            domain_output = self.domain_classifier(features)
        return label_output, domain_output


def train_dann_sentiment(dann_model, source_loader, target_loader, optimizer,
                         classification_criterion, domain_criterion, epochs, alpha,lamda):
    dann_model.train()
    for epoch in range(epochs):
        for i, ((source_data, source_labels), (target_data, _)) in enumerate(zip(source_loader, target_loader)):
            source_data, source_labels = source_data, source_labels
            target_data = target_data

            optimizer.zero_grad()

            # Forward pass
            label_output_source, domain_output_source = dann_model(source_data, reverse=True)
            _, domain_output_target = dann_model(target_data, reverse=True)

            # Calculate losses
            classification_loss = classification_criterion(label_output_source, source_labels)
            domain_loss_source = domain_criterion(domain_output_source, torch.zeros_like(domain_output_source))
            domain_loss_target = domain_criterion(domain_output_target, torch.ones_like(domain_output_target))
            domain_loss = domain_loss_source + domain_loss_target
            total_loss = classification_loss + domain_loss

            # Backward pass
            total_loss.backward()
            optimizer.step()

            if (i + 1) % 10 == 0:
                print(f'Epoch [{epoch+1}/{epochs}], Batch [{i+1}/{len(source_loader)}], Total Loss: {total_loss.item():.4f}, '
                      f'Classification Loss: {classification_loss.item():.4f}, Domain Loss: {domain_loss.item():.4f}')
    return dann_model

def extract_dictionary(reviews):
    dictionary = {}
    i = 0
    for review in reviews:
        words = review.split()
        for word in words:
            act_word,freq = word.split(':')
            if act_word not in dictionary:
                dictionary[act_word] = (i,0)
                i += 1
    
    for review in reviews:
        words = review.split()
        for word in words:
            act_word,freq = word.split(':')
            try:
                n = int(freq)
            except:
                n = 0
            dictionary[act_word] = (dictionary[act_word][0], dictionary[act_word][1] + n)
    
    # Sort the dictionary based on frequency
    sorted_dict = sorted(dictionary.items(), key=lambda item: item[1][1], reverse=True)
    # Create a new dictionary with sorted keys
    sorted_dictionary = {k: v for k, v in sorted_dict}
    # taked the first 400 words
    sorted_dictionary = dict(list(sorted_dictionary.items())[:400])
    
    final_dict = {}
    for i, (word, (index, freq)) in enumerate(sorted_dictionary.items()):
        final_dict[word] = i
    return final_dict

def OneHotEncoding(reviews, dictionary):
    one_hot_reviews = []
    for review in reviews:
        one_hot_review = np.zeros(len(dictionary))
        words = review.split()
        for word in words:
            act_word,freq = word.split(':')
            try:
                n = int(freq)
            except:
                n = 0
            if act_word in dictionary:
                index = dictionary[act_word]
                one_hot_review[index] += n
        one_hot_reviews.append(one_hot_review)
    return np.array(one_hot_reviews)
    
def data_extractor(domain_path, target_path):
    domain_folder = domain_path
    positive_reviews = []
    negative_reviews = []
    unlabeled_reviews = []
    print(os.listdir(domain_folder))
    for filename in os.listdir(domain_folder):
        if filename == 'negative.review':
            with open(os.path.join(domain_folder, filename), 'r', encoding='utf-8') as file:
                for line in file:
                    negative_reviews.append(line.strip())
        elif filename == 'positive.review':
            with open(os.path.join(domain_folder, filename), 'r', encoding='utf-8') as file:
                for line in file:
                    positive_reviews.append(line.strip())
        elif filename == 'unlabeled.review':
            with open(os.path.join(domain_folder, filename), 'r', encoding='utf-8') as file:
                for line in file:
                    unlabeled_reviews.append(line.strip())
    
    
    source_reviews = positive_reviews + negative_reviews
    dictionary = extract_dictionary(source_reviews)

    positive_reviews_encoded = OneHotEncoding(positive_reviews, dictionary)
    negative_reviews_encoded = OneHotEncoding(negative_reviews, dictionary)

    encoded_data = np.concatenate((positive_reviews_encoded, negative_reviews_encoded), axis=0)

    labelled_reviews_encoded =np.zeros(2000)
    for i in range(1000):
        labelled_reviews_encoded[i] = 1

    index = list(range(2000))
    random.shuffle(index)


    Source_Y = np.array([labelled_reviews_encoded[i] for i in index])
    Source_x = np.array([encoded_data[i] for i in index])

    books_folder2 = target_path
    positive_reviews2 = []
    negative_reviews2 = []
    unlabeled_reviews2 = []
    print(os.listdir(books_folder2))
    for filename in os.listdir(books_folder2):
        if filename == 'negative.review':
            with open(os.path.join(books_folder2, filename), 'r', encoding='utf-8') as file:
                for line in file:
                    negative_reviews2.append(line.strip())
        elif filename == 'positive.review':
            with open(os.path.join(books_folder2, filename), 'r', encoding='utf-8') as file:
                for line in file:
                    positive_reviews2.append(line.strip())
        elif filename == 'unlabeled.review':
            with open(os.path.join(books_folder2, filename), 'r', encoding='utf-8') as file:
                for line in file:
                    unlabeled_reviews2.append(line.strip())

    
    positive_reviews_encoded2 = OneHotEncoding(positive_reviews2, dictionary)
    negative_reviews_encoded2 = OneHotEncoding(negative_reviews2, dictionary)

    encoded_data2 = np.concatenate((positive_reviews_encoded2, negative_reviews_encoded2), axis=0)

    labelled_reviews_encoded2 =np.zeros(2000)
    for i in range(1000):
        labelled_reviews_encoded2[i] = 1

    index = list(range(2000))
    random.shuffle(index)


    Target_Y = np.array([labelled_reviews_encoded2[i] for i in index])
    Target_x = np.array([encoded_data2[i] for i in index])

    Target_x = torch.tensor(Target_x, dtype=torch.float32)  # Use torch.long for embedding
    Source_x = torch.tensor(Source_x, dtype=torch.float32)  # Use torch.long for embedding
    Source_Y = torch.tensor(Source_Y, dtype=torch.long)
    Target_Y = torch.tensor(Target_Y, dtype=torch.long)

    return Source_x,Source_Y,Target_x,Target_Y

def SVM(X_train, y_train, X_test):
    from sklearn import svm
    from sklearn.metrics import accuracy_score
    # Create a SVM classifier
    clf = svm.SVC(kernel='linear', C=0.1)
    # Fit the classifier to the training data
    clf.fit(X_train, y_train)
    # Make predictions on the test data
    y_pred = clf.predict(X_test)
    return y_pred
    



    
    


vocab_size = 400
embedding_dim = 128
hidden_size = 128
output_size = 64
num_classes = 2
alpha = 0.1
epochs = 100
learning_rate = 0.001
batch_size = 32


Source_x,Source_Y,Target_x,Target_Y = data_extractor(Source_loc,Target_loc)


# train loader
source_dataset = TensorDataset(Source_x, Source_Y)
source_loader = DataLoader(source_dataset, batch_size=batch_size, shuffle=True)

# target loader
target_dataset = TensorDataset(Target_x, torch.zeros_like(Target_x))  # Dummy labels for target domain
target_loader = DataLoader(target_dataset, batch_size=batch_size, shuffle=True)

# Initialize model, loss functions, and optimizer
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
feature_extractor = FeatureExtractor(vocab_size, hidden_size, output_size) 
label_predictor = LabelPredictor(output_size, hidden_size, num_classes)
domain_classifier = DomainClassifier(output_size, hidden_size)
dann_model = DANN(feature_extractor, label_predictor, domain_classifier, alpha)
classification_criterion = nn.CrossEntropyLoss()
domain_criterion = nn.BCELoss()
optimizer = optim.Adam(dann_model.parameters(), lr=learning_rate)

# Train the model
train_dann_sentiment(dann_model, source_loader, target_loader, optimizer,
                     classification_criterion, domain_criterion, epochs,1,0.5)
# Save the model
torch.save(dann_model.state_dict(), 'dann_sentiment_model.pth')
# Load the model (if needed)
dann_model.load_state_dict(torch.load('dann_sentiment_model.pth'))
# Test the model (optional) 

# test on target domain
dann_model.eval()
correct = 0
total = 0
target_dataset2 = TensorDataset(Target_x,Target_Y)  # Dummy labels for target domain
target_loader2 = DataLoader(target_dataset2, batch_size=batch_size, shuffle=False)

with torch.no_grad():
    for data, labels in target_loader2:
        outputs, _ = dann_model(data, reverse=False)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
accuracy_dann = 100 * correct / total
print(f'Accuracy of the DANN model on the target domain: {accuracy_dann:.2f}%')

# test on target domain
dann_model.eval()
correct = 0
total = 0
# target_dataset2 = TensorDataset(Target_x,Target_Y)  # Dummy labels for target domain
# target_loader2 = DataLoader(target_dataset2, batch_size=batch_size, shuffle=False)

with torch.no_grad():
    for data, labels in source_loader:
        outputs, _ = dann_model(data, reverse=False)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
accuracy = 100 * correct / total
print(f'Accuracy of the model on the target domain: {accuracy:.2f}%')

# train loader
source_dataset = TensorDataset(Source_x, Source_Y)
source_loader = DataLoader(source_dataset, batch_size=batch_size, shuffle=True)

# target loader
target_dataset = TensorDataset(Target_x, torch.zeros_like(Target_x))  # Dummy labels for target domain
target_loader = DataLoader(target_dataset, batch_size=batch_size, shuffle=True)

# Initialize model, loss functions, and optimizer
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
feature_extractor = FeatureExtractor(vocab_size, hidden_size, output_size) 
label_predictor = LabelPredictor(output_size, hidden_size, num_classes)
domain_classifier = DomainClassifier(output_size, hidden_size)
dann_model = DANN(feature_extractor, label_predictor, domain_classifier, 0)
classification_criterion = nn.CrossEntropyLoss()
domain_criterion = nn.BCELoss()
optimizer = optim.Adam(dann_model.parameters(), lr=learning_rate)

# Train the model
train_dann_sentiment(dann_model, source_loader, target_loader, optimizer,
                     classification_criterion, domain_criterion, epochs,0,0.5)
# Save the model
torch.save(dann_model.state_dict(), 'dann_sentiment_model.pth')
# Load the model (if needed)
dann_model.load_state_dict(torch.load('dann_sentiment_model.pth'))
# Test the model (optional) 

# test on target domain
dann_model.eval()
correct = 0
total = 0
target_dataset2 = TensorDataset(Target_x,Target_Y)  # Dummy labels for target domain
target_loader2 = DataLoader(target_dataset2, batch_size=batch_size, shuffle=False)

with torch.no_grad():
    for data, labels in target_loader2:
        outputs, _ = dann_model(data, reverse=False)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
accuracy_DNN = 100 * correct / total
print(f'Accuracy of the DNN model on the target domain: {accuracy_DNN:.2f}%')

Y_pred = SVM(Source_x, Source_Y, Target_x)
# check the accuracy
from sklearn.metrics import accuracy_score
accuracy_SVM = accuracy_score(Target_Y, Y_pred)
print(f'Accuracy of the model on the target domain: {accuracy_dann:.2f}%')
print(f'Accuracy of the DNN model on the target domain: {accuracy_DNN:.2f}%')
print(f'Target Accuracy on SVM: {accuracy_SVM:.4f}')

