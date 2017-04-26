rm -rf ./src/*.o  ./exp  
rm -rf obj
make


#wiki
#/usr/local/cuda/bin/cuda-memcheck ./exp ./example/dataset-2/wiki-2.vertices ./example/dataset-2/wiki-2.edges 2394385 5021410  1216070 2329783 2  
#/usr/local/cuda/bin/cuda-memcheck ./exp ./example/dataset-4/wiki-talk.vertices ./example/dataset-4/wiki-talk.edges 2394385 5021410  618287  1164892 4 


#amazon


/usr/local/cuda/bin/cuda-memcheck ./exp ./example/dataset-2/amazon-2.vertices ./example/dataset-2/amazon-2.edges 735322 5158012 538554 1761624 2 /home/xxling/amazon.txt_totem
/usr/local/cuda/bin/cuda-memcheck ./exp ./example/dataset-4/amazon.vertices ./example/dataset-4/amazon.edges 735322 5158012 356275 880813 4 /home/xxling/amazon.txt_totem
#/usr/local/cuda/bin/cuda-memcheck ./exp ./example/dataset-4/amazon.vertices ./example/dataset-4/amazon.edges 735322 5158012 356275 880813 4 
