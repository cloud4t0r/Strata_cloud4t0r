// src/step35/step35_hostread_probe.cpp - can a kernel read the GGUF mapping directly?
//
// A device-side MoE (one kernel, the eight experts' matvecs, no host round trip) needs the expert bytes to be
// DEVICE-ADDRESSABLE.  On this engine they live in the file's mapping (`GgufModel::shard().tensor_data()`), a plain
// mmap, and every miss copies them to VRAM with a host-issued cudaMemcpy.  The round trip is what a gathered kernel
// would remove, and whether it can remove it at all depends on one question the documentation does not answer for
// this driver: does HIP let a kernel read that mapping?
//
// This probe answers it with numbers - register a mapped window, get its device pointer, read it from a kernel, and
// compare against the same bytes copied to VRAM first.  It is a diagnostic, not a feature: it prints what it finds and
// exits 0 unless the basic CUDA calls fail.
#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <vector>

namespace {

__global__ void k_read_sum(const uint8_t* p, size_t n, unsigned long long* out) {
    unsigned long long acc = 0;
    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += (size_t) gridDim.x * blockDim.x) acc += p[i];
    atomicAdd(out, acc);
}

double time_ms(void (*launch)(void*), void* arg) {
    cudaEvent_t a = nullptr, b = nullptr;
    cudaEventCreate(&a);
    cudaEventCreate(&b);
    cudaEventRecord(a);
    launch(arg);
    cudaEventRecord(b);
    cudaEventSynchronize(b);
    float ms = 0;
    cudaEventElapsedTime(&ms, a, b);
    cudaEventDestroy(a);
    cudaEventDestroy(b);
    return ms;
}

struct ReadArgs {
    const uint8_t* ptr;
    size_t n;
    unsigned long long* out;
};

void launch_read(void* p) {
    const ReadArgs* a = (const ReadArgs*) p;
    k_read_sum<<<256, 256>>>(a->ptr, a->n, a->out);
}

}  // namespace

int main(int argc, char** argv) {
    // A file to read: the caller's, or this binary (the point is the memory kind, not the content).
    const char* path = argc > 1 ? argv[1] : "/proc/self/exe";
    const int fd = open(path, O_RDONLY);
    if (fd < 0) {
        std::printf("probe: cannot open %s\n", path);
        return 1;
    }
    struct stat st {};
    fstat(fd, &st);
    size_t n = (size_t) st.st_size;
    if (n > (size_t) 256 << 20) n = (size_t) 256 << 20;   // a quarter GB is plenty to time
    void* map = mmap(nullptr, n, PROT_READ, MAP_PRIVATE, fd, 0);
    if (map == MAP_FAILED) {
        std::printf("probe: mmap failed\n");
        close(fd);
        return 1;
    }
    std::printf("probe: %s, %.1f MB mapped read-only (a plain mmap, like the model's shards)\n", path,
                (double) n / 1e6);

    unsigned long long* d_out = nullptr;
    if (cudaMalloc((void**) &d_out, sizeof(unsigned long long)) != cudaSuccess ||
        cudaMemset(d_out, 0, sizeof(unsigned long long)) != cudaSuccess) {
        std::printf("probe: no CUDA device\n");
        return 1;
    }

    // ---- 1: the bytes copied to VRAM first (what the engine does today)
    uint8_t* d_buf = nullptr;
    if (cudaMalloc((void**) &d_buf, n) != cudaSuccess) {
        std::printf("probe: out of device memory for %.1f MB\n", (double) n / 1e6);
        return 1;
    }
    cudaMemcpy(d_buf, map, n, cudaMemcpyHostToDevice);
    ReadArgs copied{d_buf, n, d_out};
    const double t_copy_read = time_ms(launch_read, &copied);

    // ---- 2: the mapping registered as host memory the device can read (what a gathered kernel would need)
    const cudaError_t reg = cudaHostRegister(map, n, cudaHostRegisterMapped);
    if (reg != cudaSuccess) {
        std::printf("probe: cudaHostRegister on a file mapping FAILS (%s): a device-side MoE cannot read the GGUF\n",
                    cudaGetErrorString(reg));
        std::printf("probe: copied-to-VRAM read: %.3f ms for %.1f MB (%.1f GB/s)\n", t_copy_read, (double) n / 1e6,
                    (double) n / t_copy_read / 1e6);
        cudaFree(d_buf);
        cudaFree(d_out);
        return 0;
    }
    void* dev_view = nullptr;
    if (cudaHostGetDevicePointer(&dev_view, map, 0) != cudaSuccess) {
        std::printf("probe: registered, but not device-addressable: a device-side MoE cannot read the GGUF\n");
        cudaHostUnregister(map);
        cudaFree(d_buf);
        cudaFree(d_out);
        return 0;
    }
    ReadArgs direct{(const uint8_t*) dev_view, n, d_out};
    const double t_direct = time_ms(launch_read, &direct);
    std::printf("probe: registered and device-addressable - a kernel read %.1f MB from the mapping in %.3f ms "
                "(%.1f GB/s), against %.3f ms after a copy to VRAM (%.1f GB/s of copy)\n",
                (double) n / 1e6, t_direct, (double) n / t_direct / 1e6, t_copy_read, (double) n / t_copy_read / 1e6);

    cudaHostUnregister(map);
    cudaFree(d_buf);
    cudaFree(d_out);
    munmap(map, n);
    close(fd);
    return 0;
}
