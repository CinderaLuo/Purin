rm -rf graph.h.gch  timer.h.gch core.* a.out
g++ -g  main.cpp graph.cpp  check.cpp codingIndex.cpp bfs.cpp  graph.h  timer.h algorithm.h
./a.out  ../example/output.vertices ../example/output.edges 6 8 3 2 4
