all : 
	g++ -O3 -fopenmp -o openmp openmp.cpp

get:
	scp s2120210448@node1:/home/s2120210448/res_64.csv .
	scp s2120210448@node1:/home/s2120210448/res_128.csv .

clean:
	rm -rf *.out
