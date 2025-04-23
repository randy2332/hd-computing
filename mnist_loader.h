// mnist_loader.h
#ifndef MNIST_LOADER_H

#define MNIST_LOADER_H
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

// 定義MNIST數據集結構
typedef struct {
    int number_of_images; //圖片總數
    int number_of_rows; //圖片row, 這裡是28
    int number_of_cols; // 圖片column, 這裡是28
    unsigned char **images;  // 二維數組存儲圖像
    unsigned char *labels;   // 一維數組存儲標籤
} MNIST_Dataset;

// 函數聲明
uint32_t swap_endian(uint32_t value);
MNIST_Dataset* load_mnist(const char* image_path, const char* label_path);
void free_mnist(MNIST_Dataset* dataset);

#endif