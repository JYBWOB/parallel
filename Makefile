all : 
	clang++ -g -march=armv8-a base_for_linux.cpp -o base_c.out
	scp base_c.out s2120210448@node1:/home/s2120210448
	g++ -g -march=native base_for_linux.cpp -o base_g.out
	scp base_g.out s2120210448@node1:/home/s2120210448

get:
	scp s2120210448@node1:/home/s2120210448/res_base.csv .

clean:
	rm -rf *.out
