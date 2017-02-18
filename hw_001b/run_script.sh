#!/usr/bin/env bash

if [ ! -f 2016-02-16-22-36-39.bag ]; then
  echo 'Extracing rosbag file'
  tar -Jxf bag/2016-02-16-22-36-39.bag.tar.xz
fi

echo 'Starting subsriber'
./scripts/extended_kalman_filter.py &

echo 'Playing rosbag file'
rosbag play '2016-02-16-22-36-39.bag'

kill %%
