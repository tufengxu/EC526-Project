CXX=g++

all: test_mnist_plot

test_mnist_plot: test_mnist
	./test_mnist
	python3 plot_mnist.py

test_iris_plot: test_iris iris.data
	./test_iris > tsne_iris.dat
	python3 plot_iris.py

test_iris: test.cpp tsne.cpp iris.data
	$(CXX) -fopenmp -l gomp test.cpp -o test_iris

test_mnist: test.cpp tsne.cpp mnist2500_labels.txt
	$(CXX) -fopenmp -l gomp test.cpp -o test_mnist

test: sbs_test_py.py sbs_test_cpp.cpp test.cpp tsne.cpp iris.data
	$(CXX) sbs_test_cpp.cpp -o sbs_test_cpp
	echo "CPP:"
	./sbs_test_cpp
	echo "PYTHON:"
	python3 sbs_test_py.py


clean:
	rm tsne_*.dat a.out test_iris *.png
