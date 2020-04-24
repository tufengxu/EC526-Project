all: test_plot

test_plot: test_iris iris.data
	./test_iris > tsne_iris.dat
	python3 plot_iris.py

test_iris: test.cpp tsne.cpp iris.data
	g++ test.cpp -o test_iris

clean:
	rm tsne_*.dat a.out test_iris *.png
