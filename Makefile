CXX = g++
CXXFLAGS = -Iinclude -O2 -Wall -std=c++11 -fopenmp -lpthread -MMD

SRC = $(wildcard src/*.cc) main.cc
OBJ = $(SRC:.cc=.o)
TARGET = main

all: $(TARGET)

$(TARGET): $(OBJ)
	$(CXX) $(CXXFLAGS) -o $@ $^

%.o: %.cc
	$(CXX) $(CXXFLAGS) -c $< -o $@

-include $(OBJ:.o=.d)

clean:
	rm -f $(OBJ) $(TARGET) $(OBJ:.o=.d)

