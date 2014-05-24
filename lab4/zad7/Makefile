all:
	mpicxx main.cpp -o miller_rabin -Wall
	
clean:
	rm miller_rabin

rebuild: clean all

run: 
	mpirun -n 2 ./miller_rabin test.txt 20000
	mpirun -n 3 ./miller_rabin test.txt 20000
	mpirun -n 4 ./miller_rabin test.txt 20000
	mpirun -n 5 ./miller_rabin test.txt 20000
	mpirun -n 6 ./miller_rabin test.txt 20000
	mpirun -n 7 ./miller_rabin test.txt 20000
	mpirun -n 8 ./miller_rabin test.txt 20000
	mpirun -n 9 ./miller_rabin test.txt 20000
	mpirun -n 10 ./miller_rabin test.txt 20000
	mpirun -n 11 ./miller_rabin test.txt 20000
	mpirun -n 12 ./miller_rabin test.txt 20000
	mpirun -n 13 ./miller_rabin test.txt 20000
	mpirun -n 14 ./miller_rabin test.txt 20000
	mpirun -n 15 ./miller_rabin test.txt 20000
	mpirun -n 16 ./miller_rabin test.txt 20000
