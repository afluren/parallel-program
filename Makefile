# CXX = g++
# CXXFLAGS = -Iinclude -O2 -Wall -std=c++11 -fopenmp -lpthread -MMD

# CXX = mpic++
# CXXFLAGS = -Iinclude -O2 -Wall -std=c++11 -fopenmp -MMD

CXX = nvcc
CXXFLAGS = -Iinclude -O2 -Xcompiler -fopenmp -Xcompiler -lpthread -MMD -m64 -std=c++11

SRC = $(wildcard src/*.cc) $(wildcard src/*.cu) main.cc
OBJ = $(SRC:.cc=.o)
OBJ := $(OBJ:.cu=.o)
TARGET = main

all: $(TARGET)

$(TARGET): $(OBJ)
	$(CXX) $(CXXFLAGS) -o $@ $^

%.o: %.cc
	$(CXX) $(CXXFLAGS) -c $< -o $@

%.o: %.cu
	$(CXX) $(CXXFLAGS) -c $< -o $@

-include $(OBJ:.o=.d)

clean:
	rm -f $(OBJ) $(TARGET) $(OBJ:.o=.d)
