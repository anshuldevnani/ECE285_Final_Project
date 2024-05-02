#!/bin/bash


# clear old files on pynq board
rm y_hw.npy
ssh -T xilinx@192.168.2.99 << _remote_commands
cd /home/xilinx
rm hls4ml* axi_stream_driver.py package.tar.gz *.npy run_pynq.py
_remote_commands

# copy package to pynq board
scp ./$1/package.tar.gz xilinx@192.168.2.99:/home/xilinx
scp run_pynq.py xilinx@192.168.2.99:/home/xilinx

# Extract and run 
ssh -T xilinx@192.168.2.99 << _remote_commands_2
cd /home/xilinx
tar -xf package.tar.gz
sudo -i
source /etc/profile.d/pynq_venv.sh
cd /home/xilinx
python3 run_pynq.py
_remote_commands_2

# copy results back to local
scp xilinx@192.168.2.99:/home/xilinx/y_hw.npy ./

# Rasberry pi copy
ssh -T anshul@192.168.1.82 << _remote_commands_3
cd /home/anshul
rm x_test.npy *.h5 run_pi.py
_remote_commands_3

scp -T x_test.npy $1.h5 run_pi.py anshul@192.168.1.82:/home/anshul

ssh -T anshul@192.168.1.82 << _remote_commands_4
cd /home/anshul
python run_pi.py --model $1
_remote_commands_4

