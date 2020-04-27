CXX=g++
CXX_FLAGS=-Wall -O2
OMP_FLAGS=-fopenmp -l gomp

all: test_mnist_plot

test_mnist_plot: test_mnist
	./test_mnist
	python3 plot_mnist.py

test_iris_plot: test_iris iris.data
	./test_iris
	python3 plot_iris.py

test_iris: test.cpp tsne.cpp iris.data
	$(CXX) $(OMP_FLAGS) $(CXX_FLAGS) -D IRIS test.cpp -o test_iris

test_mnist: test.cpp tsne.cpp mnist2500_labels.txt
	$(CXX) $(OMP_FLAGS) $(CXX_FLAGS) test.cpp -o test_mnist

no_parallel: test.cpp tsne.cpp mnist2500_X.txt mnist2500_labels.txt
	$(CXX) -D NO_PARALLEL $(CXX_FLAGS) test.cpp -o test_mnist_nop
	./test_mnist_nop
	python3 plot_mnist.py

no_parallel_iris: test.cpp tsne.cpp iris.data
	$(CXX) -D NO_PARALLEL -D IRIS $(CXX_FLAGS) test.cpp -o test_iris_nop
	./test_iris_nop
	python3 plot_iris.py

scc: test_mnist
	./test_mnist

scc_no_parallel: test.cpp tsne.cpp mnist2500_X.txt mnist2500_labels.txt
	$(CXX) -D NO_PARALLEL $(CXX_FLAGS) test.cpp -o test_mnist_nop
	./test_mnist_nop


# test: sbs_test_py.py sbs_test_cpp.cpp test.cpp tsne.cpp iris.data
# 	$(CXX) sbs_test_cpp.cpp -o sbs_test_cpp
# 	echo "CPP:"
# 	./sbs_test_cpp
# 	echo "PYTHON:"
# 	python3 sbs_test_py.py

clean:
	rm tsne_*.dat a.out test_iris *.png test_iris_nop test_mnist_nop test_mnist
