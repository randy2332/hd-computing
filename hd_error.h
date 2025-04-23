// hd_error.h - Error handling for HD Computing
#ifndef HD_ERROR_H
#define HD_ERROR_H

#include <stdio.h>

// Error codes
typedef enum {
    HD_SUCCESS = 0,                 // Operation succeeded
    HD_ERROR_MEMORY_ALLOCATION,     // Memory allocation failed
    HD_ERROR_INVALID_PARAMETER,     // Invalid parameter
    HD_ERROR_FILE_IO,               // File I/O error
    HD_ERROR_NOT_INITIALIZED,       // Component not initialized
    HD_ERROR_NOT_TRAINED,           // Model not trained yet
    HD_ERROR_BINDING_FAILED,        // Binding operation failed
    HD_ERROR_BUNDLING_FAILED,       // Bundling operation failed
    HD_ERROR_ENCODING_FAILED,       // Encoding operation failed
    HD_ERROR_UNKNOWN                // Unknown error
} HDErrorCode;

// Global error state
extern HDErrorCode hd_last_error;
extern char hd_error_message[256];

// Error handling functions
void hd_set_error(HDErrorCode code, const char* message);
const char* hd_get_error_message();
HDErrorCode hd_get_error_code();
void hd_clear_error();

// Debug print function
void hd_debug_print(const char* format, ...);

#endif // HD_ERROR_H