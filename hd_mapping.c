// hd_mapping.c
#include "hd_mapping.h"
#include <stdlib.h>



int get_level_index(HDMapping* mapping, int value) {
    // 檢查是否在範圍內
    if (value < mapping->input_min) return 0;
    if (value >= mapping->input_max) return mapping->n_levels - 1;

    // 計算level索引
    int level = 0;
    for (int i = 1; i <= mapping->n_levels; i++) {
        if (value < mapping->thresholds[i]) {
            level = i - 1;
            break;
        }
    }

  
    return level;
}

char* get_level_vector(HDLevelVectors* hd, int value, HDMapping* mapping) {
    int level_index = get_level_index(mapping, value);
    

    
    return hd->vectors[level_index];
}

void encode_mnist_image(HDLevelVectors* hd, unsigned char* image, char** encoded_image, 
                       int image_size, HDMapping* mapping) {
    //printf("Encoding image:\n");
    for (int i = 0; i < image_size; i++) {
        encoded_image[i] = get_level_vector(hd, image[i], mapping);
        if (i < 5) {  // 只顯示前5個像素的映射信息
            printf("Pixel[%d] = %d -> Level vector[%d]\n", 
                   i, image[i], get_level_index(mapping, image[i]));
        }
    }
}

HDMapping* init_mapping(int input_min, int input_max, int n_levels) {
    HDMapping* mapping = (HDMapping*)malloc(sizeof(HDMapping));
    if (!mapping) return NULL;

    mapping->input_min = input_min;
    mapping->input_max = input_max;
    mapping->n_levels = n_levels;

    // 分配閾值數組內存
    mapping->thresholds = (int*)malloc((n_levels + 1) * sizeof(int));
    if (!mapping->thresholds) {
        free(mapping);
        return NULL;
    }

    // 計算每個level的閾值
    double range = (double)(input_max - input_min + 1);
    double step = range / n_levels;
    
    printf("\nInitializing mapping thresholds:\n");
    for (int i = 0; i <= n_levels; i++) {
        mapping->thresholds[i] = (int)(input_min + i * step);
        printf("Threshold[%d] = %d\n", i, mapping->thresholds[i]);
    }

    return mapping;
}

void free_mapping(HDMapping* mapping) {
    if (mapping) {
        free(mapping->thresholds);
        free(mapping);
    }
}

