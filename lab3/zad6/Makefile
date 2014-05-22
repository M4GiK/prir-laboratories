CXX=g++

CUDA_INSTALL_PATH=/usr/local/cuda
CFLAGS= -I. -I$(CUDA_INSTALL_PATH)/include -I/usr/include/opencv -I/usr/local/include/opencv
LDFLAGS= -L$(CUDA_INSTALL_PATH)/lib -lcudart -L/usr/local/lib -lopencv_core -lcv -lhighgui 
#CUDAFLAGS= -gencode arch=compute_20,code=sm_2

#Uncomment the line below if you dont have CUDA enabled GPU
#EMU=-deviceemu

#ifdef EMU
#CUDAFLAGS+=-deviceemu
#endif

all:
	$(CXX) $(CFLAGS) -c main.cpp -o main.o -c VideoOperations.cpp -o video.o
	nvcc $(CUDAFLAGS) -c kernel_gpu.cu -o kernel_gpu.o
	$(CXX) $(LDFLAGS) main.o kernel_gpu.o video.o -o gauss_gpu 

clean:
	rm -f *.o gauss_gpu
