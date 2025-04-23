#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>



typedef struct {
    int levels;        
    int dimension;     
    float randomness;  // 隨機参数 (0 到 1 間)
    char **vectors;    
} HDLevelVectors;

void free_level_vectors(HDLevelVectors* hd);
void generate_random_vector(char *vector, int dimension);
void interpolate_vectors(char *result, const char *vec1, const char *vec2, 
                         const float *threshold, float t, int dimension);
int hamming_distance(char* vec1, char* vec2, int dimension);
void print_vector(char* vector, int dimension);

// 生成随机二进制向量
void generate_random_vector(char *vector, int dimension) {
    for (int i = 0; i < dimension; i++) {
        vector[i] = rand() % 2;
    }
}

// 根据阈值向量进行插值，类似torchhd中的torch.where函数
void interpolate_vectors(char *result, const char *vec1, const char *vec2, 
                         const float *threshold, float t, int dimension) {
    for (int i = 0; i < dimension; i++) {
        // 如果阈值小于t，选择vec1的值，否则选择vec2的值
        result[i] = (threshold[i] < t) ? vec1[i] : vec2[i];
    }
}

// TorchHD风格的初始化函数
HDLevelVectors* init_level_vectors(int num_vectors, int dimension, float randomness) {
    if (num_vectors <= 0 || dimension <= 0 || randomness < 0 || randomness > 1) {
        return NULL;
    }
    
    // 分配主结构内存
    HDLevelVectors* hd = (HDLevelVectors*)malloc(sizeof(HDLevelVectors));
    if (!hd) return NULL;
    
    // 初始化结构体
    hd->levels = num_vectors;
    hd->dimension = dimension;
    hd->randomness = randomness;
    
    // 分配向量数组内存
    hd->vectors = (char**)malloc(num_vectors * sizeof(char*));
    if (!hd->vectors) {
        free(hd);
        return NULL;
    }
    
    // 为每个level分配内存
    for (int i = 0; i < num_vectors; i++) {
        hd->vectors[i] = (char*)malloc(dimension * sizeof(char));
        if (!hd->vectors[i]) {
            // 清理已分配的内存
            for (int j = 0; j < i; j++) {
                free(hd->vectors[j]);
            }
            free(hd->vectors);
            free(hd);
            return NULL;
        }
    }
    
    // 初始化隨機生成器
    srand(time(NULL));

    // 計算span,可以參考torchhd實現方式
    float levels_per_span = (1 - randomness) * (num_vectors - 1) + randomness * 1;
    levels_per_span = (levels_per_span < 1) ? 1 : levels_per_span; // 至少為1
    float span = (num_vectors - 1) / levels_per_span;
    int span_count = (int)ceilf(span + 1);

    // 生成基礎向量 (類似span_hv)
    char **span_vectors = (char**)malloc(span_count * sizeof(char*));
    if (!span_vectors) {
        free_level_vectors(hd);
        return NULL;
    }
    
    for (int i = 0; i < span_count; i++) {
        span_vectors[i] = (char*)malloc(dimension * sizeof(char));
        if (!span_vectors[i]) {
            // 清理
            for (int j = 0; j < i; j++) {
                free(span_vectors[j]);
            }
            free(span_vectors);
            free_level_vectors(hd);
            return NULL;
        }
        generate_random_vector(span_vectors[i], dimension);
    }
    
    // 生成閥值向量 (類似threshold_v)
    float *threshold = (float*)malloc(dimension * sizeof(float));
    if (!threshold) {
        // 清理
        for (int i = 0; i < span_count; i++) {
            free(span_vectors[i]);
        }
        free(span_vectors);
        free_level_vectors(hd);
        return NULL;
    }
    
    for (int i = 0; i < dimension; i++) {
        threshold[i] = (float)rand() / RAND_MAX; // 0到1之間之隨機值
    }
    
    // 爲每個level生成向量
    for (int i = 0; i < num_vectors; i++) {
        int span_idx = (int)(i / levels_per_span);
        
        // 特殊情況：如果在span邊界上
        if (fabs(fmod(i, levels_per_span)) < 1e-12) {
            // 直接使用該span的正交向量
            for (int j = 0; j < dimension; j++) {
                hd->vectors[i][j] = span_vectors[span_idx][j];
            }
        } else {
            // 計算在當前span内的位置
            float level_within_span = fmod(i, levels_per_span);
            // 從起始向量角度計算的閥值
            float t = 1 - (level_within_span / levels_per_span);
            
            // 在兩個基礎向量之間進行插值
            interpolate_vectors(hd->vectors[i], 
                              span_vectors[span_idx], 
                              span_vectors[span_idx + 1], 
                              threshold, t, dimension);
        }
    }
    
    // 清理記憶體
    for (int i = 0; i < span_count; i++) {
        free(span_vectors[i]);
    }
    free(span_vectors);
    free(threshold);
    
    return hd;
}

// 釋放記憶體
void free_level_vectors(HDLevelVectors* hd) {
    if (hd) {
        if (hd->vectors) {
            for (int i = 0; i < hd->levels; i++) {
                free(hd->vectors[i]);
            }
            free(hd->vectors);
        }
        free(hd);
    }
}

// 
void print_vector(char* vector, int dimension) {
    printf("[");
    for (int i = 0; i < dimension; i++) {
        printf("%d", vector[i]);
        if (i < dimension - 1) printf(",");
    }
    printf("]\n");
}