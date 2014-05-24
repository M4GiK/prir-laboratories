COMPLIATOR			= 	nvcc
CXX					= 	g++
OPENVC_INSTALL_PATH	=	/usr/local/include/opencv
OPENVC_LIB			=	/usr/local/lib
CFLAGS				= 	-I$(OPENVC_INSTALL_PATH) 
LDFLAGS				= 	-L$(OPENVC_LIB) -lopencv_highgui -lopencv_core 
CUDAFLAGS			= 	-gencode arch=compute_20,code=sm_20 

all: gauss_gpu

gauss_gpu: video.o kernel_gpu.o
	$(COMPLIATOR) main.cpp VideoOperations.o kernel_gpu.o -o gauss_gpu $(CUDAFLAGS) $(LDFLAGS) $(CFLAGS)
	
video.o:
	$(CXX) -c VideoOperations.cpp
	
kernel_gpu.o:
	$(COMPLIATOR) -c kernel_gpu.cu $(CUDAFLAGS) $(LDFLAGS) $(CFLAGS)
	
main.o:

clean:
	rm -f *.o gauss_gpu

rebuild: 
	clean all
