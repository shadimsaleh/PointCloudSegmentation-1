rm -Rf build
mkdir build
cd build
cmake ..
make
cd ..
cp test.pcd build/

