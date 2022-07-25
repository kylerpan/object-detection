#include <iostream>
#include <fstream>

#include "engine.h"
#include "NvOnnxParser.h"

bool Engine::build(std::string onnx_path)
{
    if (!doesFileExist(onnx_path))
    {
        std::cout << "ONNX model not found: " << onnx_path << std::endl;
        return false;
    }

    // Only regenerate the engine file if it has not already been generated for the specified options
    engine_name_ = generateEngineName(onnx_path);
    if (doesFileExist(engine_name_))
    {
        std::cout << "Engine found, not regenerating..." << std::endl;
        return true;
    }

    std::cout << "Engine not found, generating..." << std::endl;

    // Create our engine builder.
    auto builder = std::unique_ptr<nvinfer1::IBuilder>(nvinfer1::createInferBuilder(logger_));
    if (!builder) return false;

    // Set the max supported batch size
    builder->setMaxBatchSize(options_.max_batch_size);

    // Define an explicit batch size and then create the network.
    // More info here: https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index.html#explicit-implicit-batch
    auto explicitBatch = 1U << static_cast<uint32_t>(NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
    auto network = std::unique_ptr<nvinfer1::INetworkDefinition>(builder->createNetworkV2(explicitBatch));
    if (!network) return false;

    // Create a parser for reading the onnx file.
    auto parser = std::unique_ptr<nvonnxparser::IParser>(nvonnxparser::createParser(*network, logger_));
    if (!parser) return false;

    // We are going to first read the onnx file into memory, then pass that buffer to the parser.
    std::ifstream file(onnx_path, std::ios::binary | std::ios::ate);
    std::streamsize size = file.tellg();
    file.seekg(0, std::ios::beg);

    std::vector<char> buffer(size);
    if (!file.read(buffer.data(), size))
    {
        printf("Unable to read engine file\n");
        return false;
    }

    auto parsed = parser->parse(buffer.data(), buffer.size());
    if (!parsed) return false;

    // Save the input height, width, and channels.
    const auto input = network->getInput(0);
    const auto output = network->getOutput(0);
    const auto input_name = input->getName();
    const auto input_dims = input->getDimensions();
    int32_t in_c = input_dims.d[1];
    int32_t in_h = input_dims.d[2];
    int32_t in_w = input_dims.d[3];

    auto config = std::unique_ptr<nvinfer1::IBuilderConfig>(builder->createBuilderConfig());
    if (!config) return false;

    // Specify the optimization profiles
    IOptimizationProfile* defaultProfile = builder->createOptimizationProfile();
    defaultProfile->setDimensions(input_name, OptProfileSelector::kMIN, Dims4(1, in_c, in_h, in_w));
    defaultProfile->setDimensions(input_name, OptProfileSelector::kOPT, Dims4(1, in_c, in_h, in_w));
    defaultProfile->setDimensions(input_name, OptProfileSelector::kMAX, Dims4(options_.max_batch_size, in_c, in_h, in_w));
    config->addOptimizationProfile(defaultProfile);

    // Specify all the optimization profiles.
    for (const auto& opt_batch_size: options_.opt_batch_sizes)
    {
        if (opt_batch_size == 1) continue;

        if (opt_batch_size > options_.max_batch_size)
        {
            printf("opt_batch_size cannot be greater than max_batch_size!\n");
            continue;
        }

        IOptimizationProfile* profile = builder->createOptimizationProfile();
        profile->setDimensions(input_name, OptProfileSelector::kMIN, Dims4(1, in_c, in_h, in_w));
        profile->setDimensions(input_name, OptProfileSelector::kOPT, Dims4(opt_batch_size, in_c, in_h, in_w));
        profile->setDimensions(input_name, OptProfileSelector::kMAX, Dims4(options_.max_batch_size, in_c, in_h, in_w));
        config->addOptimizationProfile(profile);
    }

    config->setMaxWorkspaceSize(options_.max_work_space);

    if (options_.fp16) config->setFlag(BuilderFlag::kFP16);

    // CUDA stream used for profiling by the builder.
    auto profileStream = samplesCommon::makeCudaStream();
    if (!profileStream) return false;
    config->setProfileStream(*profileStream);

    // Build the engine
    std::unique_ptr<IHostMemory> plan {builder->buildSerializedNetwork(*network, *config)};
    if (!plan) return false;

    // Write the engine to disk
    std::ofstream outfile(engine_name_, std::ofstream::binary);
    outfile.write(reinterpret_cast<const char*>(plan->data()), plan->size());

    std::cout << "Success, saved engine to " << engine_name_ << std::endl;
    return true;
}

bool Engine::loadNetwork()
{
    // Read the serialized model from disk
    std::ifstream file(engine_name_, std::ios::binary | std::ios::ate);
    std::streamsize size = file.tellg();
    file.seekg(0, std::ios::beg);

    std::vector<char> buffer(size);
    if (!file.read(buffer.data(), size))
    {
        printf("Unable to read engine file");
        return false;
    }

    std::unique_ptr<IRuntime> runtime {createInferRuntime(logger_)};
    if (!runtime)
    {
        printf("fail to create run time\n");
        return false;
    }

    // Set the device index
    auto ret = cudaSetDevice(options_.gpu_device_idx);
    if (ret != 0)
    {
        printf("fail to set GPU device %d (%d)\n", options_.gpu_device_idx, getNumGpus());
        return false;
    }

    engine_ = std::unique_ptr<nvinfer1::ICudaEngine>(runtime->deserializeCudaEngine(buffer.data(), buffer.size()));
    if (!engine_)
    {
        printf("fail to create cuda engine from %s\n", engine_name_.c_str());
        return false;
    }

    context_ = std::unique_ptr<nvinfer1::IExecutionContext>(engine_->createExecutionContext());
    if (!context_)
    {
        printf("fail to create execution context\n");
        return false;
    }

    ret = cudaStreamCreate(&cuda_stream_);
    if (ret != 0)
    {
        printf("fail to create cuda stream\n");
        return false;
    }

    return true;
}

std::vector<Tensor> Engine::getTensors(std::vector<std::string> tensors_name)
{
    Tensor tensor;
    std::vector<Tensor> tensors(tensors_name.size(), tensor);

    // loop through all network inputs/outputs
    for (int tensor_idx = 0; tensor_idx < engine_->getNbBindings(); tensor_idx++)
    {
        // get tensor dimension and build tensor
        auto dims = engine_->getBindingDimensions(tensor_idx);

        for (int i = 0; i < dims.nbDims; i++)
        {
            int n = dims.nbDims - 1 - i;
            if (n == 0)      tensor.w = dims.d[i];
            else if (n == 1) tensor.h = dims.d[i];
            else if (n == 2) tensor.c = dims.d[i];
        }
        tensor.updateNumElems();

        // allocate buffer
        samplesCommon::ManagedBuffer buffer;
        buffer.hostBuffer.resize(tensor.num_elems);
        buffer.deviceBuffer.resize(tensor.num_elems);
        tensor.data = (float*)buffer.hostBuffer.data();

        // push to vectors
        buffers_binding_.push_back(buffer.deviceBuffer.data());
        buffers_.push_back(std::move(buffer));

        // find tensor name
        auto name = engine_->getBindingName(tensor_idx);
        for (int i = 0; i < tensors_name.size(); i++)
        {
            if (name == tensors_name[i])
            {
                tensors[i] = tensor;
                break;
            }
        }
    }

    // check if we get all tensors from tensors_name
    for (int i = 0; i < tensors.size(); i++)
    {
        if (tensors[i].data == nullptr)
            printf("ERROR: fail to get tensor for %s\n", tensors_name[i].c_str());
    }

    return tensors;
}

bool Engine::execute()
{
    // Copy from CPU to GPU
    for (int i = 0; i < buffers_.size(); i++)
    {
        if (!engine_->bindingIsInput(i)) continue;

        auto ret = cudaMemcpyAsync(buffers_[i].deviceBuffer.data(), buffers_[i].hostBuffer.data(),
                   buffers_[i].hostBuffer.nbBytes(), cudaMemcpyHostToDevice, cuda_stream_);
        if (ret != 0)
        {
            printf("Unable to copy from CPU(%p) to GPU(%p) %lldbytes\n",
                   buffers_[i].hostBuffer.data(), buffers_[i].deviceBuffer.data(), buffers_[i].deviceBuffer.nbBytes());
            return false;
        }
    }

    // Run inference.
    bool status = context_->enqueueV2(buffers_binding_.data(), cuda_stream_, nullptr);
    if (!status)
    {
        std::cout << "fail in inference" << std::endl;
        return false;
    }

    // Copy the results back to CPU memory
    for (int i = 0; i < buffers_.size(); i++)
    {
        if (engine_->bindingIsInput(i)) continue;
        auto ret = cudaMemcpyAsync(buffers_[i].hostBuffer.data(), buffers_[i].deviceBuffer.data(),
            buffers_[i].deviceBuffer.nbBytes(), cudaMemcpyDeviceToHost, cuda_stream_);
        if (ret != 0)
        {
            printf("Unable to copy from GPU(%p) to CPU(%p) %lldbytes\n",
                buffers_[i].deviceBuffer.data(), buffers_[i].hostBuffer.data(), buffers_[i].deviceBuffer.nbBytes());
            return false;
        }
    }

    auto ret = cudaStreamSynchronize(cuda_stream_);
    if (ret != 0)
    {
        std::cout << "Unable to synchronize cuda stream" << std::endl;
        return false;
    }

    return true;
}

std::string Engine::generateEngineName(const std::string& file_path)
{
    std::string engine_name = "engine";

    // checks if the gpu index in options is greater the number of gpus in machine
    if (options_.gpu_device_idx >= getNumGpus())
    {
        printf("Error, provided device index is out of range!");
        return "";
    }

    // Serialize the specified options into the filename
    engine_name += "-" + getGpuName();

    if (options_.fp16) engine_name += "-fp16";
    else engine_name += "-fp32";

    engine_name += "-" + std::to_string(options_.max_batch_size);
    for (int i = 0; i < options_.opt_batch_sizes.size(); ++i)
    {
        engine_name += "_" + std::to_string(options_.opt_batch_sizes[i]);
    }

    engine_name += "-" + std::to_string(options_.max_work_space/1000000) + "m";

    size_t pos = file_path.find_last_of("\\/");
    std::string file_name = file_path;
    if (pos != std::string::npos) file_name = file_path.substr(pos + 1);
    engine_name += "-" + file_name + ".trt";

    return engine_name;
}

std::string Engine::getGpuName()
{
    int device = 0;
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device);

    std::string gpu_name = prop.name;
    for (int i = 0; i < gpu_name.size(); i++)
    {
        if (gpu_name[i] == ' ') gpu_name[i] = '_';
    }
    return gpu_name;
}

int32_t Engine::getNumGpus()
{
    int num_gpus;
    cudaGetDeviceCount(&num_gpus);
    return num_gpus;
}
