#ifndef WIN_UTILS_GPU_H
#define WIN_UTILS_GPU_H

// assert function on GPU
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=false)
{
    if (code != cudaSuccess) {
        fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}

#define gpuErrChk(ans) { gpuAssert((ans), __FILE__, __LINE__); }

#endif
