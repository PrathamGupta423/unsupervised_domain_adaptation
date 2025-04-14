!/bin/bash

# Make directory svhn
mkdir -p svhn
# Download the SVHN dataset
# 1. http://ufldl.stanford.edu/housenumbers/train_32x32.mat
wget -P svhn http://ufldl.stanford.edu/housenumbers/train_32x32.mat
# 2. http://ufldl.stanford.edu/housenumbers/test_32x32.mat
wget -P svhn http://ufldl.stanford.edu/housenumbers/test_32x32.mat