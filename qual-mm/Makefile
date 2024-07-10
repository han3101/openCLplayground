# Variables
CC = g++
CFLAGS = -I/usr/local/cuda-12.3/targets/x86_64-linux/include -Iinclude -std=c++11
LDFLAGS = -L/usr/lib/x86_64-linux-gnu -lOpenCL
SRCS = matrix_multiply.cpp
OBJS = $(SRCS:.cpp=.o)
EXEC = main

# Default target
all: $(EXEC)

# Rule to build the executable
$(EXEC): $(OBJS)
	$(CC) $(OBJS) -o $@ $(LDFLAGS)

# Rule to build object files
%.o: %.cpp
	$(CC) $(CFLAGS) -c $< -o $@

# Clean up build files
clean:
	rm -f $(OBJS) $(EXEC)

# Phony targets
.PHONY: all clean
