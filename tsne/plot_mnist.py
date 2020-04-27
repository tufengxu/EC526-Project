import numpy as np 
import matplotlib.pyplot as plt

mnist_out = np.loadtxt('tsne_mnist.dat')
mnist_data = np.loadtxt('mnist2500_X.txt')
mnist_label = np.loadtxt('mnist2500_labels.txt')

plt.scatter(mnist_out[:, 0], mnist_out[:, 1], c=mnist_label)
plt.savefig('tsne_mnist.png')
plt.close()

plt.figure()
(_, n) = mnist_data.shape
plt.scatter(
    np.mean(mnist_data[:, :(n // 2)], axis=1), 
    np.mean(mnist_data[:, (n // 2):], axis=1), 
    c=mnist_label)
plt.savefig('mnist.png')
plt.close()