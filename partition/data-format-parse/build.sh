g++ mod_random_edgelist_totem.c -o mod_edgelist_totem
rm -rf *.txt*
for i in amazon dblp  youtube wiki-Talk roadNet-CA
do
	echo "${i}...."
		./mod_edgelist_totem  ../nochange/${i}.txt  ${i}.txt_totem 
	echo -e "\n"
done
