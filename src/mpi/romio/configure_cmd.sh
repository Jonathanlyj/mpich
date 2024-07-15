CC=/scratch/yll6162/mpich/mpich-install/bin/mpicc \
CFLAGS="-O2 -I/usr/local/cuda-12.5/targets/x86_64-linux/include/" \
LDFLAGS="-L/usr/local/cuda-12.5/lib64 -lcudart -lcufile " \
./configure --prefix=/scratch/yll6162/romio/romio-install/