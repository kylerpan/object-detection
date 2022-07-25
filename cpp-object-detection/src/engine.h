#pragma once
#pragma warning(disable: 4244)

#include "NvInfer.h"
#include "buffers.h"

// Options for the network
struct Options
{
    bool fp16 = false;
    std::vector<int32_t> opt_batch_sizes;
    int32_t max_batch_size = 1;
    int gpu_device_idx = 0;

    // Max allowable GPU memory to be used for model conversion, in bytes.
    // Applications should allow the engine builder as much workspace as they can afford;
    // at runtime, the SDK allocates no more than this and typically less.
    size_t max_work_space = (size_t)6000000000;
};

// Class to extend TensorRT logger
class Logger : public nvinfer1::ILogger
{
    void log (Severity severity, const char* msg) noexcept override
    {
        if (severity <= Severity::kWARNING) std::cout << msg << std::endl;
    }
};

// input/output tensor definition
typedef struct tTensor
{
    int32_t b, c, h, w;
    int32_t num_elems;
    float* data;

    tTensor() { b = h = w = c = 1; num_elems = 0; data = nullptr; }
    void updateNumElems() { num_elems = b * h * w * c; }
} Tensor;

// Engine class
class Engine
{
public:
    Engine(const Options& options) : options_(options) {}
    ~Engine() { if (cuda_stream_) cudaStreamDestroy(cuda_stream_); }

    // Build TensorRT network from onnx
    bool build(std::string onnx_path);

    // Load and prepare the TensorRT network for inference
    bool loadNetwork();

    std::vector<Tensor> getTensors(std::vector<std::string> tensors_name);
    bool execute();

private:
    // Converts the engine options into a string
    std::string generateEngineName(const std::string& file_path);
    std::string getGpuName();
    int32_t getNumGpus();
    bool doesFileExist(const std::string& file_path)
    {
        std::ifstream f(file_path.c_str());
        return f.good();
    }

private:
    std::unique_ptr<nvinfer1::ICudaEngine> engine_ = nullptr;
    std::unique_ptr<nvinfer1::IExecutionContext> context_ = nullptr;

    std::vector<samplesCommon::ManagedBuffer> buffers_;
    std::vector<void*> buffers_binding_;
    cudaStream_t cuda_stream_ = nullptr;

    const Options options_;
    std::string engine_name_;
    Logger logger_;
};
