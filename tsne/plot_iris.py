import numpy as np 
import matplotlib.pyplot as plt

iris_out = np.loadtxt('tsne_iris.dat')
iris_data = np.loadtxt('iris.data', delimiter=',', dtype=np.string_)
colormap = {
    b'Iris-setosa': 'red',
    b'Iris-versicolor': 'green',
    b'Iris-virginica': 'blue'
} 
colorlist = [colormap[i.tostring()] for i in iris_data[:, 4]]

plt.scatter(iris_out[:, 0], iris_out[:, 1], c=colorlist)
plt.savefig('tsne_iris.png')