CONFIG += c++11

QMAKE_CXXFLAGS += -std=c++11

QMAKE_CXXFLAGS += -Wall -Werror

QMAKE_CXXFLAGS += -fopenmp
QMAKE_LFLAGS += -fopenmp

TARGET = pi_omp

SOURCES += \
    main.cpp
