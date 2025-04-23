# Makefile for HD Computing with Multiple Datasets

# Compiler and flags
CC = gcc
CFLAGS = -Wall -Wextra -O2
LDFLAGS = -lm

# Directories
SRC_DIR = .
BUILD_DIR = build
OUTPUT_DIR = output

# Ensure build and output directories exist
$(shell mkdir -p $(BUILD_DIR))
$(shell mkdir -p $(OUTPUT_DIR))

# Source files
SRCS = $(wildcard $(SRC_DIR)/*.c)

# Explicitly list all source files to make sure none are missed
SRC_FILES = \
	$(SRC_DIR)/main.c \
	$(SRC_DIR)/dataset.c \
	$(SRC_DIR)/hd_core.c \
	$(SRC_DIR)/hd_binding.c \
	$(SRC_DIR)/hd_bundling.c \
	$(SRC_DIR)/hd_inference.c \
	$(SRC_DIR)/hd_level.c \
	$(SRC_DIR)/hd_mapping.c \
	$(SRC_DIR)/hd_similarity.c \
	$(SRC_DIR)/hd_training.c \
	$(SRC_DIR)/hd_error.c \
	$(SRC_DIR)/mnist_loader.c \
	$(SRC_DIR)/ucihar_loader.c \
	$(SRC_DIR)/isolet_loader.c \
	$(SRC_DIR)/cifar10_loader.c \
	$(SRC_DIR)/fmnist_loader.c \
	$(SRC_DIR)/connect4_loader.c

# Object files
OBJS = $(patsubst $(SRC_DIR)/%.c, $(BUILD_DIR)/%.o, $(SRC_FILES))

# Executable name
TARGET = hd_computing

# Main build target
all: $(TARGET)

# Linking object files into executable
$(TARGET): $(OBJS)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

# Compiling source files into object files
$(BUILD_DIR)/%.o: $(SRC_DIR)/%.c
	$(CC) $(CFLAGS) -c $< -o $@

# Clean build files
clean:
	rm -f $(BUILD_DIR)/*.o $(TARGET)

# Clean all generated files
cleanall: clean
	rm -f $(OUTPUT_DIR)/*_model.h

# Run with MNIST dataset
run_mnist: $(TARGET)
	./$(TARGET) mnist

# Run with UCI HAR dataset
run_ucihar: $(TARGET)
	./$(TARGET) ucihar

# Run with ISOLET dataset
run_isolet: $(TARGET)
	./$(TARGET) isolet

# Run with CIFAR-10 dataset
run_cifar10: $(TARGET)
	./$(TARGET) cifar10

# Run with Fashion-MNIST dataset
run_fmnist: $(TARGET)
	./$(TARGET) fmnist

# Run with Connect-4 dataset
run_connect4: $(TARGET)
	./$(TARGET) connect4

# Dependencies
$(BUILD_DIR)/main.o: $(SRC_DIR)/main.c $(SRC_DIR)/config.h $(SRC_DIR)/hd_core.h $(SRC_DIR)/dataset.h
$(BUILD_DIR)/dataset.o: $(SRC_DIR)/dataset.c $(SRC_DIR)/dataset.h $(SRC_DIR)/config.h
$(BUILD_DIR)/hd_core.o: $(SRC_DIR)/hd_core.c $(SRC_DIR)/hd_core.h $(SRC_DIR)/config.h $(SRC_DIR)/dataset.h
$(BUILD_DIR)/hd_binding.o: $(SRC_DIR)/hd_binding.c $(SRC_DIR)/hd_binding.h $(SRC_DIR)/hd_level.h $(SRC_DIR)/hd_mapping.h
$(BUILD_DIR)/hd_bundling.o: $(SRC_DIR)/hd_bundling.c $(SRC_DIR)/hd_bundling.h $(SRC_DIR)/hd_binding.h
$(BUILD_DIR)/hd_inference.o: $(SRC_DIR)/hd_inference.c $(SRC_DIR)/hd_inference.h $(SRC_DIR)/dataset.h
$(BUILD_DIR)/hd_level.o: $(SRC_DIR)/hd_level.c $(SRC_DIR)/hd_level.h
$(BUILD_DIR)/hd_mapping.o: $(SRC_DIR)/hd_mapping.c $(SRC_DIR)/hd_mapping.h $(SRC_DIR)/hd_level.h
$(BUILD_DIR)/hd_similarity.o: $(SRC_DIR)/hd_similarity.c $(SRC_DIR)/hd_similarity.h $(SRC_DIR)/hd_inference.h
$(BUILD_DIR)/hd_training.o: $(SRC_DIR)/hd_training.c $(SRC_DIR)/hd_training.h $(SRC_DIR)/hd_bundling.h
$(BUILD_DIR)/hd_error.o: $(SRC_DIR)/hd_error.c $(SRC_DIR)/hd_error.h $(SRC_DIR)/config.h

.PHONY: all clean cleanall run_mnist run_ucihar run_isolet run_cifar10 run_fmnist run_connect4