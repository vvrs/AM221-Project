# Get CIFAR10
wget http://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz
tar -xzvf cifar-10-python.tar.gz
rm cifar-10-python.tar.gz 
mkdir cifar-10-python
mv cifar-10-batches-py cifar-10-python

# Get MNIST

wget yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz
wget yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz