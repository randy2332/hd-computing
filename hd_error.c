// hd_error.c - Implementation of error handling for HD Computing
#include "hd_error.h"
#include "config.h"
#include <string.h>
#include <stdarg.h>

// Global error state
HDErrorCode hd_last_error = HD_SUCCESS;
char hd_error_message[256] = {0};

// Set error code and message
void hd_set_error(HDErrorCode code, const char* message) {
    hd_last_error = code;
    
    if (message) {
        strncpy(hd_error_message, message, sizeof(hd_error_message) - 1);
        hd_error_message[sizeof(hd_error_message) - 1] = '\0';
    } else {
        strcpy(hd_error_message, "Unknown error");
    }
    
    // Print error message to stderr
    fprintf(stderr, "HD Error: %s (Code: %d)\n", hd_error_message, code);
}

// Get the last error message
const char* hd_get_error_message() {
    return hd_error_message;
}

// Get the last error code
HDErrorCode hd_get_error_code() {
    return hd_last_error;
}

// Clear the error state
void hd_clear_error() {
    hd_last_error = HD_SUCCESS;
    hd_error_message[0] = '\0';
}

// Debug print function
void hd_debug_print(const char* format, ...) {
    if (HD_DEBUG_PRINT) {
        va_list args;
        va_start(args, format);
        vprintf(format, args);
        va_end(args);
    }
}