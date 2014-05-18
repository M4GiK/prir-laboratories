CXX=g++

CUDA_INSTALL_PATH=/usr/local/cuda
CFLAGS= -I. -I$(CUDA_INSTALL_PATH)/include -I/usr/include/opencv 
LDFLAGS= -L$(CUDA_INSTALL_PATH)/lib -lcudart -L/usr/lib -lcxcore -lcv -lhighgui -lcvaux -lml 

#Uncomment the line below if you dont have CUDA enabled GPU
#EMU=-deviceemu

ifdef EMU
CUDAFLAGS+=-deviceemu
endif

all:
	$(CXX) $(CFLAGS) -c main.cpp -o main.o -m32 -arch i386
	nvcc $(CUDAFLAGS) -c kernel_gpu.cu -o kernel_gpu.o
	$(CXX) $(LDFLAGS) main.o kernel_gpu.o -o grayscale -arch i386

clean:
	rm -f *.o grayscale

