rm -rf ./src/*.o  ./exp  
rm -rf obj
make
/usr/local/cuda/bin/cuda-memcheck ./exp ./example/output.vertices ./example/output.edges 6 8 3 2 1
		
#./exp ./example/output.vertices ./example/output.edges 6 8 4
