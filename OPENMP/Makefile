all : 
	g++ -fopenmp -g -march=native -o openmp openmp.cpp
	g++ -fopenmp -g -march=native -o openmp_update openmp_update.cpp

update:
	g++ -fopenmp -g -march=native -o openmp openmp_update.cpp

get:
	scp s2120210448@node1:/home/s2120210448/res_64.csv .
	scp s2120210448@node1:/home/s2120210448/res_128.csv .

clean:
	rm -rf *.out
