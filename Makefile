all : 
	clang++ -g -march=armv8-a base_for_linux.cpp -o base.out
	scp base.out s2120210448@node1:/home/s2120210448

clean:
	rm -rf *.out
