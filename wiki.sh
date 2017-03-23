rm -rf ./src/*.o  ./exp  
rm -rf obj
make

#small example with output
#/usr/local/cuda/bin/cuda-memcheck ./exp ./example/output.vertices ./example/output.edges 6 8 3 2 4 >file.txt  2>&1
#without output
#/usr/local/cuda/bin/cuda-memcheck ./exp ./example/output.vertices ./example/output.edges 6 8 3 2 4 

#wiki
#/usr/local/cuda/bin/cuda-memcheck ./exp ./example/wiki-talk.vertices ./example/wiki-talk.edges 2394385 5021410  618287  1164892 4 >> ./txt/file.txt 

/usr/local/cuda/bin/cuda-memcheck ./exp ./example/dataset-4/wiki-talk.vertices ./example/dataset-4/wiki-talk.edges 2394385 5021410  618287  1164892 4  
#amazon

#/usr/local/cuda/bin/cuda-memcheck ./exp ./example/dataset-4/amazon.vertices ./example/dataset-4/amazon.edges 735322 5158012 356275 880813 4 

#/usr/local/cuda/bin/cuda-memcheck ./exp ./example/amazon.vertices ./example/amazon.edges 735322 5158012 356275 880813 4 
