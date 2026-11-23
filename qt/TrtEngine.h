#pragma once
#include <opencv2/core.hpp>
#include <string>
#include <vector>
#include <unordered_map>
#include <cuda_runtime.h>
#include <NvInfer.h>

// 通用 TRT 推理封装：多输入 / 动态输出，所有行为模型共用
// 工业惯例：一个引擎类管理加载、IO 绑定、同步推理、结果回传
// 动态输入（batch 维度）：setInputShape 声明实际 shape 后 enqueue
class TrtEngine {
public:
    TrtEngine() = default;
    ~TrtEngine();

    bool load(const std::string &enginePath);
    bool ready() const { return context_ != nullptr; }

    // 动态输入的实际 shape（如 [8,3,256,192]），必须在 setInput 之前调用
    bool setInputShape(const std::string &name, const nvinfer1::Dims &dims);

    // 上传输入（host float 数组，按输入 shape 的元素数）
    bool setInput(const std::string &name, const float *data, size_t nFloats);

    // 同步推理
    bool infer();

    // 输出实际 shape（推理后有效，动态输出）
    nvinfer1::Dims outShape(const std::string &name) const;
    // 下载输出到调用方复用缓冲（高频路径无堆分配；out 只扩不缩）
    void getOutput(const std::string &name, std::vector<float> &out);
    // 便捷版：内部临时缓冲
    std::vector<float> getOutput(const std::string &name);

    nvinfer1::Dims inShape(const std::string &name) const;
    bool hasTensor(const std::string &name) const;

private:
    struct Tensor {
        std::string name;
        bool isInput = false;
        nvinfer1::Dims dims;
        void *dev = nullptr;
        size_t maxElems = 0;
    };
    nvinfer1::IRuntime   *runtime_  = nullptr;
    nvinfer1::ICudaEngine *engine_  = nullptr;
    nvinfer1::IExecutionContext *context_ = nullptr;
    cudaStream_t stream_ = nullptr;
    std::vector<Tensor> tensors_;
    std::unordered_map<std::string, int> idx_;

    class Logger;
    Logger *logger_ = nullptr;

    static size_t volume(const nvinfer1::Dims &d);
    Tensor *find(const std::string &name);
};
