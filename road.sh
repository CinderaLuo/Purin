#rm -rf ./src/*.o  ./exp  
#rm -rf obj
#make

rm -rf ./result/*.txt

datapath=/home/xxling/dataset/VGP-master/dist
origin=/home/xxling/dataset

#/usr/local/cuda/bin/cuda-memcheck ./exp ./example/dataset-4/amazon.vertices ./example/dataset-4/amazon.edges 735322 5158012 356275 880813 4 

#roadca
echo "roadca.."
#/usr/local/cuda/bin/cuda-memcheck ./exp ${datapath}/roadNet-CA-2.vertices ${datapath}/roadNet-CA-2.edges 1971281 5533214 1234123 1383305 2 ${origin}/roadNet-CA.txt_totem 

#roadca

#echo "roadca.."
/usr/local/cuda/bin/cuda-memcheck ./exp ${datapath}/roadNet-CA-4.vertices ${datapath}/roadNet-CA-4.edges 1971281 5533214 1054087 692363 4 ${origin}/roadNet-CA.txt_totem 

#roadpa
#rmat16_2
#/usr/local/cuda/bin/cuda-memcheck ./exp ${datapath}/RMAT6_2.vertices ${datapath}/RMAT6_2.edges 10000000 160000000 
