# ifndef HD_LEVEL_H
# define HD_LEVEL_H

#include <stdio.h>
#include <stdlib.h>
#include <time.h>

// 定義向量結構
typedef struct {
    int levels;         // 總共的level數量
    int dimension;      // 向量維度
    int randomness;      //  randomness
    char **vectors;     // 存儲所有level的向量
} HDLevelVectors;

// 函數聲明
HDLevelVectors* init_level_vectors(int levels, int dimension, float randomness);
void free_level_vectors(HDLevelVectors* hd);
void print_vector(char* vector, int dimension);

#endif