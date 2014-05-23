CXX					=	g++

CUDA_INSTALL_PATH	=	/usr/local/cuda
OPENVC_INSTALL_PATH	=	/usr/local/include/opencv
OPENVC2_INSTALL_PATH=	/usr/local/include/opencv2
OPENVC_PATH			=	/usr/local/share/OpenCV
OPENVC_LIB			=	/usr/local/lib
CFLAGS				= 	-I. -I$(CUDA_INSTALL_PATH)/include -I$(OPENVC_INSTALL_PATH)  -I$(OPENVC2_INSTALL_PATH) -I$(OPENVC_PATH)
LDFLAGS				= 	-L$(OPENVC_LIB) -lopencv_highgui -lopencv_core 
CUDAFLAGS			= 	-L$(CUDA_INSTALL_PATH)/lib64 -lcudart   -arch sm_20

#Uncomment the line below if you dont have CUDA enabled GPU
#EMU=-deviceemu

#ifdef EMU
#CUDAFLAGS+=-deviceemu
#endif

all:
	$(CXX) 	$(CFLAGS) 	$(LDFLAGS) 		-c main.cpp 			-o main.o  
	$(CXX) 	$(CFLAGS) 	$(LDFLAGS) 		-c VideoOperations.cpp 	-o video.o
	nvcc 	$(CFLAGS) 	$(CUDAFLAGS) 	-c kernel_gpu.cu 		-o kernel_gpu.o		
	$(CXX) 	$(LDFLAGS) 	main.o video.o kernel_gpu.o 			-o gauss_gpu

clean:
	rm -f *.o gauss_gpu

rebuild: 
	clean all
	
#all: gauss_gpu
#
#gauss_gpu: video.o kernel_gpu.o main.o
#	$(CXX) $(LDFLAGS) video.o main.o kernel_gpu.o -o gauss_gpu
#
#main.o:
#	$(CXX) $(CFLAGS) main.cpp  kernel_gpu.h -o main.o
#
#video.o:
#	$(CXX) $(CFLAGS) VideoOperations.cpp VideoOperations.hpp -o video.o
#
#kernel_gpu.o:
#	nvcc $(CFLAGS) $(CUDAFLAGS) kernel_gpu.cu VideoOperations.hpp kernel_gpu.h -o kernel_gpu.o
#
#clean:
#	rm -f *.o gauss_gpu
	