#!/usr/bin/env bash
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
pushd ${DIR}
echo "Compiling code..."
if [ ! -d ./build ]; then
  mkdir build
fi

# run cmake
pushd build
cmake ../
make
echo "Generating Results"
./estimate_voltage
echo "Plotting"
../scripts/plot_kalman.py
popd
popd

