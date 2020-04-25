import numpy as np 
import matplotlib.pyplot as plt

iris_out = np.loadtxt('tsne_iris.dat')
iris_data = np.loadtxt('iris.data', delimiter=',', dtype=np.string_)
colormap = {
    b'Iris-setosa': 0,
    b'Iris-versicolor': 1,
    b'Iris-virginica': 2
} 
colorlist = np.array([colormap[i.tostring()] for i in iris_data[:, 4]])

plt.scatter(iris_out[:, 0], iris_out[:, 1], c=colorlist)
plt.savefig('tsne_iris.png')