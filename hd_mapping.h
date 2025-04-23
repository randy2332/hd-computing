// hd_mapping.h
#ifndef HD_MAPPING_H
#define HD_MAPPING_H

#include "mnist_loader.h"
#include "hd_level.h"

// 定義映射結構
typedef struct {
    int input_min;      // 輸入範圍最小值 (0)
    int input_max;      // 輸入範圍最大值 (255)
    int n_levels;       // level數量
    int* thresholds;    // 儲存每個level的閾值
} HDMapping;

// 函數聲明
HDMapping* init_mapping(int input_min, int input_max, int n_levels);
int get_level_index(HDMapping* mapping, int value);
char* get_level_vector(HDLevelVectors* hd, int value, HDMapping* mapping);
void free_mapping(HDMapping* mapping);
void encode_mnist_image(HDLevelVectors* hd, unsigned char* image, char** encoded_image, 
                       int image_size, HDMapping* mapping);

#endif

