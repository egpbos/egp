#include "octree.h"
#include <iostream>

#ifdef WIN32
#include <tchar.h>
int _tmain(int argc, _TCHAR* argv[])
#else
int main(int argc, char* argv[])
#endif
{
    Octree<unsigned short int> o(4096); /* Create 4096x4096x4096 octree containing doubles. */
    o(1,2,3) = 3.1416;      /* Put pi in (1,2,3). */
    std::cout << o.at(1,2,3) << "\n";
    
    o.erase(1,2,3);         /* Erase that node. */
    std::cout << o.at(1,2,3) << "\n" << double(0);
	return 0;
}

