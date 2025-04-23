// hd_training.c (Binary Version)
#include "hd_training.h"
#include <stdlib.h>
#include <stdio.h>
#include <math.h>

ClassVectors* init_class_vectors(int n_classes, int dimension) {
    ClassVectors* cv = (ClassVectors*)malloc(sizeof(ClassVectors));
    if (!cv) return NULL;

    cv->n_classes = n_classes;
    cv->dimension = dimension;

    // 分配類別計數器內存
    cv->class_counts = (int*)calloc(n_classes, sizeof(int));
    if (!cv->class_counts) {
        free(cv);
        return NULL;
    }

    // 分配累加器內存
    cv->accumulators = (int**)malloc(n_classes * sizeof(int*));
    if (!cv->accumulators) {
        free(cv->class_counts);
        free(cv);
        return NULL;
    }

    // 分配每個類別的累加器內存
    for (int i = 0; i < n_classes; i++) {
        cv->accumulators[i] = (int*)calloc(dimension, sizeof(int));
        if (!cv->accumulators[i]) {
            for (int j = 0; j < i; j++) {
                free(cv->accumulators[j]);
            }
            free(cv->accumulators);
            free(cv->class_counts);
            free(cv);
            return NULL;
        }
    }

    // 分配類別HV內存
    cv->class_hvs = (char**)malloc(n_classes * sizeof(char*));
    if (!cv->class_hvs) {
        for (int i = 0; i < n_classes; i++) {
            free(cv->accumulators[i]);
        }
        free(cv->accumulators);
        free(cv->class_counts);
        free(cv);
        return NULL;
    }

    // 分配每個類別HV的內存
    for (int i = 0; i < n_classes; i++) {
        cv->class_hvs[i] = (char*)malloc(dimension * sizeof(char));
        if (!cv->class_hvs[i]) {
            for (int j = 0; j < i; j++) {
                free(cv->class_hvs[j]);
            }
            for (int j = 0; j < n_classes; j++) {
                free(cv->accumulators[j]);
            }
            free(cv->class_hvs);
            free(cv->accumulators);
            free(cv->class_counts);
            free(cv);
            return NULL;
        }
    }

    return cv;
}

void free_class_vectors(ClassVectors* cv) {
    if (cv) {
        if (cv->class_counts) free(cv->class_counts);
        if (cv->accumulators) {
            for (int i = 0; i < cv->n_classes; i++) {
                free(cv->accumulators[i]);
            }
            free(cv->accumulators);
        }
        if (cv->class_hvs) {
            for (int i = 0; i < cv->n_classes; i++) {
                free(cv->class_hvs[i]);
            }
            free(cv->class_hvs);
        }
        free(cv);
    }
}

void accumulate_training_vector(ClassVectors* cv, int class_label, BundledVector* bundle) {
    if (class_label >= 0 && class_label < cv->n_classes) {
        // 累加
        for (int i = 0; i < cv->dimension; i++) {
            cv->accumulators[class_label][i] += bundle->final_vector[i];
        }
        cv->class_counts[class_label]++;
        
        // 二值化: 大於等於類別樣本數量一半的為1，否則為0
        int threshold = cv->class_counts[class_label] / 2;
        for (int i = 0; i < cv->dimension; i++) {
            cv->class_hvs[class_label][i] = (cv->accumulators[class_label][i] > threshold) ? 1 : 0;
        }
    }
}

// 修改統計信息顯示函數
void print_class_vector_stats(ClassVectors* cv) {
    printf("\n類別向量統計:\n");
    for (int c = 0; c < cv->n_classes; c++) {
        printf("類別 %d:\n", c);
        printf("  樣本數量: %d\n", cv->class_counts[c]);
        
        if (cv->class_counts[c] > 0) {
            // 計算1的比例
            int ones_count = 0;
            for (int i = 0; i < cv->dimension; i++) {
                if (cv->class_hvs[c][i] == 1) {
                    ones_count++;
                }
            }
            
            double ones_ratio = (double)ones_count / cv->dimension * 100;
            printf("  1的比例: %.2f%%\n", ones_ratio);

            // 打印前10個位元值
            printf("  前10個位元值: ");
            for (int i = 0; i < 10; i++) {
                printf("%d ", cv->class_hvs[c][i]);
            }
            printf("\n");
        }
    }
}