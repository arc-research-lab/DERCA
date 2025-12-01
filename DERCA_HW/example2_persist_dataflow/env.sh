#setup Vitis (2020.2/2021.1/2021.2/2022.2/2023.1)
source /tools/Xilinx/Vitis/2021.1/settings64.sh

#setup XRT (only 2022.1 availiable)
source /opt/xilinx/xrt/setup.sh

unset LD_LIBRARY_PATH #(If needed)

#setup petalinux, comment if using U250
source /home/shixin/Resource/petalinux-2021.1/environment-setup-cortexa72-cortexa53-xilinx-linux