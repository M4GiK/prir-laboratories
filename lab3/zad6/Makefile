CXX=g++

CUDA_INSTALL_PATH=/usr/local/cuda
OPENVC_INSTALL_PATH=/usr/local/include/opencv
CFLAGS= -I. -I$(CUDA_INSTALL_PATH)/include -I$(OPENVC_INSTALL_PATH)
LDFLAGS= -L$(OPENVC_INSTALL_PATH) -lopencv_highgui -lopencv_core
CUDAFLAGS= -arch sm_20

#Uncomment the line below if you dont have CUDA enabled GPU
#EMU=-deviceemu

#ifdef EMU
#CUDAFLAGS+=-deviceemu
#endif

all:
	$(CXX) $(CFLAGS) -c main.cpp -o main.o -c  
	$(CXX) $(CFLAGS) -c VideoOperations.cpp -o video.o
	nvcc $(CFLAGS) $(CUDAFLAGS) -c kernel_gpu.cu -o kernel_gpu.o
	$(CXX) $(LDFLAGS) main.o kernel_gpu.o video.o -o gauss_gpu 

clean:
	rm -f *.o gauss_gpu
