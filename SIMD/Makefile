all : 
	@echo "clang++ for neon64.cpp"
	@clang++ -g -march=armv8-a neon64.cpp -o neon64_c.out
	@echo "copy neon64_c.out to node1"
	@scp neon64_c.out s2120210448@node1:/home/s2120210448 > log.txt

	@echo "g++ for neon64.cpp"
	@g++ -g -march=native neon64.cpp -o neon64_g.out
	@echo "copy neon64_g.out to node1"
	@scp neon64_g.out s2120210448@node1:/home/s2120210448 > log.txt
	
	@echo "clang++ for neon128.cpp"
	@clang++ -g -march=armv8-a neon128.cpp -o neon128_c.out
	@echo "copy neon128_c.out to node1"
	@scp neon128_c.out s2120210448@node1:/home/s2120210448 > log.txt

	@echo "g++ for neon128.cpp"
	@g++ -g -march=native neon128.cpp -o neon128_g.out
	@echo "copy neon128_g.out to node1"
	@scp neon128_g.out s2120210448@node1:/home/s2120210448 > log.txt

get:
	scp s2120210448@node1:/home/s2120210448/res_64.csv .
	scp s2120210448@node1:/home/s2120210448/res_128.csv .

clean:
	rm -rf *.out
