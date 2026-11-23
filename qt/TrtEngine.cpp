#include "TrtEngine.h"
#include <NvInferPlugin.h>
#include <fstream>
#include <iostream>
#include <algorithm>
#include <cstdint>
#include <mutex>

// process-wide GPU serialization: with multi-channel workers the GPU is a
// single shared resource — concurrent enqueue from separate threads serializes
// at the driver anyway (observed 50x slowdown with free-for-all access), so
// take one lock and keep multi-channel throughput fair and predictable
// (same model as DeepStream: one inference pipeline, batches in sequence)
static std::mutex g_gpuMtx;
// engine deserialization + TRT plugin registry are NOT thread-safe for
// concurrent users: load engines one at a time (one-off startup cost)
static std::mutex g_loadMtx;
static std::once_flag g_pluginOnce;

class TrtEngine::Logger : public nvinfer1::ILogger {
public:
    void log(Severity sev, const char *msg) noexcept override {
        if (sev == Severity::kERROR || sev == Severity::kWARNING)
            std::cerr << "[TRT] " << msg << "\n";
    }
};

size_t TrtEngine::volume(const nvinfer1::Dims &d) {
    size_t n = 1;
    for (int i = 0; i < d.nbDims; ++i) n *= (size_t)(d.d[i] > 0 ? d.d[i] : 1);
    return n;
}

TrtEngine::Tensor *TrtEngine::find(const std::string &name) {
    auto it = idx_.find(name);
    return it == idx_.end() ? nullptr : &tensors_[it->second];
}

TrtEngine::~TrtEngine() {
    for (auto &t : tensors_) if (t.dev) cudaFree(t.dev);
    if (stream_) cudaStreamDestroy(stream_);
    delete context_;
    delete engine_;
    delete runtime_;
    delete logger_;
}

bool TrtEngine::load(const std::string &enginePath) {
    std::lock_guard<std::mutex> lk(g_loadMtx);
    std::ifstream f(enginePath, std::ios::binary | std::ios::ate);
    if (!f) { std::cerr << "[TrtEngine] cannot open " << enginePath << "\n"; return false; }
    size_t size = (size_t)f.tellg();
    f.seekg(0);
    std::vector<char> buf(size);
    f.read(buf.data(), size);

    logger_ = new Logger();
    std::call_once(g_pluginOnce, [] {          // register plugins exactly once
        initLibNvInferPlugins(nullptr, "");
    });
    runtime_ = nvinfer1::createInferRuntime(*logger_);
    if (!runtime_) return false;
    engine_ = runtime_->deserializeCudaEngine(buf.data(), size);
    if (!engine_) {
        std::cerr << "[TrtEngine] deserialize failed: " << enginePath << "\n";
        return false;
    }
    context_ = engine_->createExecutionContext();
    if (!context_) return false;
    cudaStreamCreate(&stream_);

    // discover IO tensors, allocate device buffers, bind addresses
    for (int i = 0; i < engine_->getNbIOTensors(); ++i) {
        const char *name = engine_->getIOTensorName(i);
        Tensor t;
        t.name = name;
        t.isInput = engine_->getTensorIOMode(name) == nvinfer1::TensorIOMode::kINPUT;
        t.dims = engine_->getTensorShape(name);
        // dynamic tensors: allocate for the profile MAX shape so batch>1 works
        bool dyn = false;
        for (int k = 0; k < t.dims.nbDims; ++k) if (t.dims.d[k] <= 0) { dyn = true; break; }
        nvinfer1::Dims maxD = t.dims;
        if (dyn)
            maxD = engine_->getProfileShape(name, 0, nvinfer1::OptProfileSelector::kMAX);
        size_t vol = volume(maxD);
        // dynamic outputs: reserve generous pool; x2 headroom for Int64 bindings
        t.maxElems = t.isInput ? vol : std::max<size_t>(vol, 1u << 20) * 2;
        if (cudaMalloc(&t.dev, t.maxElems * sizeof(float)) != cudaSuccess) {
            std::cerr << "[TrtEngine] cudaMalloc failed for " << name << "\n";
            return false;
        }
        if (t.isInput)
            context_->setInputTensorAddress(name, t.dev);
        else
            context_->setOutputTensorAddress(name, t.dev);
        idx_[t.name] = (int)tensors_.size();
        tensors_.push_back(std::move(t));
    }

    std::cout << "[TrtEngine] loaded " << enginePath << " ("
              << tensors_.size() << " tensors)\n";
    return true;
}

bool TrtEngine::setInputShape(const std::string &name, const nvinfer1::Dims &dims) {
    if (!context_) return false;
    if (!context_->setInputShape(name.c_str(), dims)) return false;
    return context_->setInputTensorAddress(name.c_str(), find(name)->dev);
}

bool TrtEngine::setInput(const std::string &name, const float *data, size_t nFloats) {
    Tensor *t = find(name);
    if (!t || !t->isInput) { std::cerr << "[TrtEngine] no input " << name << "\n"; return false; }
    if (nFloats > t->maxElems) { std::cerr << "[TrtEngine] input too large " << name << "\n"; return false; }
    return cudaMemcpyAsync(t->dev, data, nFloats * sizeof(float),
                           cudaMemcpyHostToDevice, stream_) == cudaSuccess;
}

bool TrtEngine::infer() {
    std::lock_guard<std::mutex> lk(g_gpuMtx);
    if (!context_) return false;
    if (!context_->enqueueV3(stream_)) return false;
    return cudaStreamSynchronize(stream_) == cudaSuccess;
}

nvinfer1::Dims TrtEngine::outShape(const std::string &name) const {
    return context_->getTensorShape(name.c_str());
}

std::vector<float> TrtEngine::getOutput(const std::string &name) {
    Tensor *t = find(name);
    if (!t || t->isInput) return {};
    nvinfer1::Dims d = context_->getTensorShape(name.c_str());
    size_t n = volume(d);
    if (n > t->maxElems) n = t->maxElems;
    std::vector<float> host(n);
    if (n && cudaMemcpyAsync(host.data(), t->dev, n * sizeof(float),
                             cudaMemcpyDeviceToHost, stream_) != cudaSuccess)
        return {};
    cudaStreamSynchronize(stream_);
    return host;
}

void TrtEngine::getOutput(const std::string &name, std::vector<float> &out) {
    Tensor *t = find(name);
    if (!t || t->isInput) { out.clear(); return; }
    nvinfer1::Dims d = context_->getTensorShape(name.c_str());
    size_t n = volume(d);
    if (n > t->maxElems) n = t->maxElems;
    out.resize(n);   // capacity reused across frames (resize never shrinks)
    if (n && cudaMemcpyAsync(out.data(), t->dev, n * sizeof(float),
                             cudaMemcpyDeviceToHost, stream_) != cudaSuccess) {
        out.clear(); return;
    }
    cudaStreamSynchronize(stream_);
}

nvinfer1::Dims TrtEngine::inShape(const std::string &name) const {
    Tensor *t = const_cast<TrtEngine*>(this)->find(name);
    return t ? t->dims : nvinfer1::Dims{};
}

bool TrtEngine::hasTensor(const std::string &name) const {
    return idx_.count(name) > 0;
}
