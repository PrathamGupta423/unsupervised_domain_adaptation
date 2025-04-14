import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from sklearn.datasets import make_moons
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import logging
from functools import wraps
from datetime import datetime
from tqdm import tqdm
import time
from joblib import Parallel, delayed 
# Device


current_dir = os.path.dirname(os.path.abspath(__file__))
log_file_path = os.path.join(current_dir, 'print_logs.log')

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
# Set random seed for reproducibility
torch.manual_seed(23626)

# Neural network
class FeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.net(x)

class Classifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(16, 2)

    def forward(self, features):
        return self.fc(features)

class Experiment1():
    def __init__(self):
        feature_extractor, classifier, Xs, ys, Xt, yt = self.train(n_epochs=1200, lambda_mmd=.5)
        self.plot_decision_boundary(classifier, feature_extractor, Xs, ys, Xt, yt)

    # Generate synthetic source and target data
    @staticmethod
    def generate_data(n_samples=1000,angle_degrees=30):
        Xs, ys = make_moons(n_samples=n_samples, noise=0.1)
        Xt, yt = make_moons(n_samples=n_samples, noise=0.1)
        # Xt[:, 0] += 0.5  # shift target to simulate domain difference
        # --- Rotate target data ---
        angle = np.radians(angle_degrees)
        R = np.array([[np.cos(angle), -np.sin(angle)],
                    [np.sin(angle),  np.cos(angle)]])
        Xt = Xt @ R.T  # rotate each point

        return torch.tensor(Xs, dtype=torch.float32), torch.tensor(ys), torch.tensor(Xt, dtype=torch.float32), torch.tensor(yt)
    
    @staticmethod
    def gaussian_kernel(x, y, sigma=1.0):
        x = x.unsqueeze(1)
        y = y.unsqueeze(0)
        dist = ((x - y) ** 2).sum(2)
        return torch.exp(-dist / (2 * sigma ** 2))


    @staticmethod
    def mmd_loss(source, target, sigma=1.0):
        # This is a baised Estimator of MMD , Commonly used in literature for training neural networks
        Kxx = Experiment1.gaussian_kernel(source, source, sigma).mean()
        Kyy = Experiment1.gaussian_kernel(target, target, sigma).mean()
        Kxy = Experiment1.gaussian_kernel(source, target, sigma).mean()
        return Kxx + Kyy - 2 * Kxy
    
    
    # Training loop
    @staticmethod
    def train(n_epochs=100, lambda_mmd=0.5):
        Xs, ys, Xt, yt = Experiment1.generate_data()
        Xs, ys, Xt, yt = Xs.to(device), ys.to(device), Xt.to(device), yt.to(device)

        feature_extractor = FeatureExtractor().to(device)
        classifier = Classifier().to(device)

        optimizer = optim.Adam(list(feature_extractor.parameters()) + list(classifier.parameters()), lr=0.001)
        criterion = nn.CrossEntropyLoss()

        for epoch in range(n_epochs):
            feature_extractor.train()
            classifier.train()
            optimizer.zero_grad()

            fs = feature_extractor(Xs)
            ft = feature_extractor(Xt)
            logits = classifier(fs)

            clf_loss = criterion(logits, ys)
            mmd = Experiment1.mmd_loss(fs, ft)
            loss = clf_loss + lambda_mmd * mmd

            loss.backward()
            optimizer.step()

            if epoch % 10 == 0 or epoch == n_epochs - 1:
                pred = classifier(feature_extractor(Xt)).argmax(dim=1)
                acc = (pred == yt).float().mean().item()
                print(f"Epoch {epoch:3d} | Clf Loss: {clf_loss.item():.4f} | MMD: {mmd.item():.4f} | Target Acc: {acc:.4f}")

        return feature_extractor, classifier, Xs.cpu(), ys.cpu(), Xt.cpu(), yt.cpu()

    @staticmethod
    def plot_decision_boundary(model, feature_extractor, Xs, ys, Xt, yt):
        h = 0.01
        x_min, x_max = min(Xs[:, 0].min(), Xt[:, 0].min()) - 0.5, max(Xs[:, 0].max(), Xt[:, 0].max()) + 0.5
        y_min, y_max = min(Xs[:, 1].min(), Xt[:, 1].min()) - 0.5, max(Xs[:, 1].max(), Xt[:, 1].max()) + 0.5
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
        grid = torch.tensor(np.c_[xx.ravel(), yy.ravel()], dtype=torch.float32).to(device)

        with torch.no_grad():
            Z = model(feature_extractor(grid)).argmax(dim=1).cpu().numpy()
        Z = Z.reshape(xx.shape)

        plt.figure(figsize=(8, 6))
        plt.contourf(xx, yy, Z, alpha=0.3)
        plt.scatter(Xs[:, 0], Xs[:, 1], c=ys, cmap="coolwarm", edgecolors='k', label="Source")
        plt.scatter(Xt[:, 0], Xt[:, 1], c=yt, cmap="coolwarm", marker="x", label="Target")
        plt.legend()
        plt.title("Decision Boundary with MMD-based Domain Adaptation")
        plt.xlabel("Feature 1")
        plt.ylabel("Feature 2")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig("decision_boundary.png")
        # plt.show()
        plt.close()

class Experiment2():


    def __init__(self):
        # P: isotropic Gaussians (std = 1)
        X = Experiment2.make_blobs_grid(std=1.0, corr=0.0, n_samples=500)
        # Q: elliptical Gaussians (std = 1, corr induced by ε)
        epsilon = 6
        rho = (epsilon - 1) / (epsilon + 1)
        Y = Experiment2.make_blobs_grid(std=1.0, corr=rho, n_samples=500, seed=42)
        # Plot the datasets
        Experiment2.print_dataset(X, Y)
        Experiment2.exp()



    @staticmethod
    def make_blobs_grid(std=1.0, corr=0.0, grid_size=5, spacing=10, n_samples=500, seed=0):
        np.random.seed(seed)
        centers = []
        for i in range(grid_size):
            for j in range(grid_size):
                centers.append([i * spacing, j * spacing])
        cov = np.array([[1, corr], [corr, 1]]) * std**2
        samples = []
        for _ in range(n_samples):
            center = centers[np.random.randint(len(centers))]
            sample = np.random.multivariate_normal(center, cov)
            samples.append(sample)
        return np.array(samples)
    
    @staticmethod
    def print_dataset(X,Y):
        plt.figure(figsize=(6, 6))
        plt.scatter(X[:, 0], X[:, 1], alpha=0.5, label='P (isotropic)', s=15)
        plt.scatter(Y[:, 0], Y[:, 1], alpha=0.5, label='Q (elliptical)', s=15)
        plt.legend()
        plt.title("Blobs dataset: P vs Q (ε = 6)")
        plt.axis("equal")
        plt.grid(True)
        plt.savefig("blobs_dataset.png")
        # plt.show()
        plt.close()

    @staticmethod
    def rbf_kernel(X, Y, sigma):
        X = X.astype(np.float64)
        Y = Y.astype(np.float64)
        XX = np.sum(X**2, axis=1, keepdims=True)
        YY = np.sum(Y**2, axis=1, keepdims=True)
        XY = X @ Y.T
        dists = np.maximum(0.0, XX - 2 * XY + YY.T)
        return np.exp(-dists / (2 * sigma**2))
    
    @staticmethod
    def compute_mmd2(X, Y, sigma):
        # Unbiased MMD^2 estimator
        K_xx = Experiment2.rbf_kernel(X, X, sigma)
        K_yy = Experiment2.rbf_kernel(Y, Y, sigma)
        K_xy = Experiment2.rbf_kernel(X, Y, sigma)
        # Use .fill_diagonal for potential minor efficiency gain
        # Although creating masks is also efficient
        np.fill_diagonal(K_xx, 0)
        np.fill_diagonal(K_yy, 0)
        m = len(X)
        n = len(Y) # Use n for Y if sizes can differ, here m=n=500
        if m == 0 or n == 0:
            return 0.0
        if m == 1 or n == 1: # Cannot compute unbiased with 1 sample
            # Fallback to biased or return 0/NaN, depending on need.
            # Here, return 0 as permutation test likely won't run.
            return 0.0

        mmd2 = (K_xx.sum() / (m * (m - 1)) +
                K_yy.sum() / (n * (n - 1)) - # Use n*(n-1) if Y can have different size
                2 * K_xy.sum() / (m * n))
        return mmd2
    
    @staticmethod
    # Function to run a single trial (for parallelization)
    def run_single_trial(eps, sigma, trial_seed, n_samples=500, n_permutations=100):
        """Runs one trial: generates data, computes MMD, runs permutation test."""
        rho = (eps - 1) / (eps + 1)
        X = Experiment2.make_blobs_grid(std=1.0, corr=0.0, n_samples=n_samples, seed=trial_seed)
        Y = Experiment2.make_blobs_grid(std=1.0, corr=rho, n_samples=n_samples, seed=trial_seed + 10000) # Ensure distinct seeds

        # Compute observed MMD
        mmd2_observed = Experiment2.compute_mmd2(X, Y, sigma)

        # Permutation test
        combined = np.concatenate([X, Y])
        count = 0
        m = len(X)
        null_stats = np.zeros(n_permutations) # Pre-allocate numpy array

        # Use a local RandomState for permutations within a trial
        perm_rng = np.random.RandomState(trial_seed + 20000)

        for i in range(n_permutations):
            perm_rng.shuffle(combined) # Shuffle in-place
            X_ = combined[:m]
            Y_ = combined[m:]
            null_stats[i] = Experiment2.compute_mmd2(X_, Y_, sigma)

        threshold = np.percentile(null_stats, 95)

        return 1 if mmd2_observed > threshold else 0
    
    @staticmethod
    def exp():
        start_time = time.time()
        epsilons = [1,2,4,6,8,10] 
        sigmas = np.logspace(-1, 1.5, 50)  # 10 sigma values
        trials = 30
        n_perms = 30
        n_jobs = -1  # use your 20 CPU cores

        results = {}

        for eps in epsilons:
            print(f"Processing epsilon = {eps}")
            rejection_rates = []
            # Use tqdm for the sigma loop
            for sigma in tqdm(sigmas, desc=f"Sigma (eps={eps})"):
                # Parallel execution of trials
                # Each call to run_single_trial is independent
                # We generate unique seeds for each trial run
                trial_seeds = range(trials) # Seeds 0 to trials-1
                reject_results = Parallel(n_jobs=n_jobs)(
                    delayed(Experiment2.run_single_trial)(eps, sigma, seed, n_samples=500, n_permutations=n_perms) for seed in trial_seeds
                )
                reject_count = sum(reject_results)
                rejection_rate = reject_count / trials
                rejection_rates.append(rejection_rate)
            results[eps] = rejection_rates

        end_time = time.time()
        print(f"Total execution time: {end_time - start_time:.2f} seconds")

        plt.figure(figsize=(10, 6))
        for eps in epsilons:
            plt.plot(sigmas, results[eps], label=f"ε = {eps}")
        plt.xscale('log')
        plt.xlabel("Sigma (RBF kernel bandwidth)")
        plt.ylabel("Rejection Rate")
        plt.title("Test Power vs Kernel Bandwidth for Different ε (Parallel CPU)")
        plt.legend()
        plt.grid(True)
        plt.savefig("test_power_vs_kernel_bandwidth.png")
        # plt.show()
        plt.close()



    


    
if __name__ == "__main__":
    Experiment1()
    Experiment2()

    


