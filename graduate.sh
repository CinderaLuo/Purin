rm -rf ./src/*.o  ./exp  
rm -rf obj
make

rm -rf ./result/*.txt

datapath=/home/xxling/dataset
origin=/home/xxling/dataset
backfix=.txt_totem

#amazon
#echo "amazon.."
#/usr/local/cuda/bin/cuda-memcheck ./exp ./example/dataset-2/amazon-2.vertices ./example/dataset-2/amazon-2.edges 735322 5158012 538554 1761624 2 ${origin}/amazon${backfix}  2>&1 >  ./result/amazon.txt
#/usr/local/cuda/bin/cuda-memcheck ./exp ./example/dataset-4/amazon.vertices ./example/dataset-4/amazon.edges 735322 5158012 356275 880813 4 

#dblp
#echo "dblp.."
#/usr/local/cuda/bin/cuda-memcheck ./exp ${datapath}/dblp-2.vertices ${datapath}/dblp-2.edges  986207 6707236  587759 1676809 2 ${origin}/dblp${backfix}  2>&1 >  ./result/dblp.txt

#roadca
#echo "roadca.."
#/usr/local/cuda/bin/cuda-memcheck ./exp ${datapath}/roadNet-CA-2.vertices ${datapath}/roadNet-CA-2.edges 1971281 5533214 1234123 1383305 2 ${origin}/roadNet-CA${backfix}  2>&1 > ./result/roadca.txt

#roadpa
#echo "roadpa.."
#/usr/local/cuda/bin/cuda-memcheck ./exp ${datapath}/roadNet-PA-2.vertices ${datapath}/roadNet-PA-2.edges 1090920 3083796 683391 770949 2 ${origin}/roadNet-PA${backfix} 2>&1 > ./result/roadpa.txt

#rmat16_1
echo "rmat1.."
/usr/local/cuda/bin/cuda-memcheck ./exp ${datapath}/RMAT16_1-2.vertices ${datapath}/RMAT16_1-2.edges 1000000 16000000 477204 7985768 2 ${origin}/RMAT16_1${backfix} 2>&1 > ./result/rmat1.txt

#rmat16_2
#/usr/local/cuda/bin/cuda-memcheck ./exp ${datapath}/RMAT6_2.vertices ${datapath}/RMAT6_2.edges 10000000 160000000 
