/*
 * Copyright (c) 2018-2024, NVIDIA CORPORATION. All rights reserved.
 *
 * Based on DeepStream-Yolo by Marcos Luciano
 * https://www.github.com/marcoslucianops
 *
 * Ported to DeepStream-Yolo-Seg by ddasdkimo
 * https://github.com/ddasdkimo/DeepStream-Yolo-Seg
 */

#include <algorithm>

#include "nvdsinfer_custom_impl.h"
#include "nvdsinfer_context.h"

#include "yolo_seg.h"

static bool
getYoloSegNetworkInfo(NetworkInfo& networkInfo, const NvDsInferContextInitParams* initParams)
{
  std::string onnxFilePath = initParams->onnxFilePath;

  if (onnxFilePath.empty()) {
    std::cerr << "ONNX file path is required for YoloSeg" << std::endl;
    return false;
  }

  std::string modelName = onnxFilePath.substr(0, onnxFilePath.find(".onnx"));
  size_t pos = modelName.rfind("/");
  if (pos != std::string::npos) {
    modelName = modelName.substr(pos + 1);
  }

  std::transform(modelName.begin(), modelName.end(), modelName.begin(), [] (uint8_t c) {
    return std::tolower(c);
  });

  networkInfo.inputBlobName = "images";
  networkInfo.onnxFilePath = onnxFilePath;
  networkInfo.batchSize = initParams->maxBatchSize;
  networkInfo.int8CalibPath = initParams->int8CalibrationFilePath;
  networkInfo.deviceType = initParams->useDLA ? "kDLA" : "kGPU";
  networkInfo.numDetectedClasses = initParams->numDetectedClasses;
  networkInfo.clusterMode = initParams->clusterMode;
  networkInfo.scaleFactor = initParams->networkScaleFactor;
  networkInfo.offsets = initParams->offsets;
  networkInfo.workspaceSize = initParams->workspaceSize;
  networkInfo.inputFormat = initParams->networkInputFormat;

  if (initParams->networkMode == NvDsInferNetworkMode_FP32) {
    networkInfo.networkMode = "FP32";
  }
  else if (initParams->networkMode == NvDsInferNetworkMode_INT8) {
    networkInfo.networkMode = "INT8";
  }
  else if (initParams->networkMode == NvDsInferNetworkMode_FP16) {
    networkInfo.networkMode = "FP16";
  }

  if (!fileExists(networkInfo.onnxFilePath)) {
    std::cerr << "ONNX file does not exist: " << networkInfo.onnxFilePath << std::endl;
    return false;
  }

  return true;
}

#if NV_TENSORRT_MAJOR >= 8
extern "C" bool
NvDsInferYoloCudaEngineGet(nvinfer1::IBuilder* const builder, nvinfer1::IBuilderConfig* const builderConfig,
    const NvDsInferContextInitParams* const initParams, nvinfer1::DataType dataType,
    nvinfer1::ICudaEngine*& cudaEngine);

extern "C" bool
NvDsInferYoloCudaEngineGet(nvinfer1::IBuilder* const builder, nvinfer1::IBuilderConfig* const builderConfig,
    const NvDsInferContextInitParams* const initParams, nvinfer1::DataType dataType, nvinfer1::ICudaEngine*& cudaEngine)
#else
extern "C" bool
NvDsInferYoloCudaEngineGet(nvinfer1::IBuilder* const builder, const NvDsInferContextInitParams* const initParams,
    nvinfer1::DataType dataType, nvinfer1::ICudaEngine*& cudaEngine);

extern "C" bool
NvDsInferYoloCudaEngineGet(nvinfer1::IBuilder* const builder, const NvDsInferContextInitParams* const initParams,
    nvinfer1::DataType dataType, nvinfer1::ICudaEngine*& cudaEngine)
#endif
{
  NetworkInfo networkInfo;
  if (!getYoloSegNetworkInfo(networkInfo, initParams))
    return false;

  YoloSeg yoloSeg(networkInfo);

#if NV_TENSORRT_MAJOR >= 8
  cudaEngine = yoloSeg.createEngine(builder, builderConfig);
#else
  cudaEngine = yoloSeg.createEngine(builder);
#endif

  if (cudaEngine == nullptr) {
    std::cerr << "Failed to build CUDA engine for YoloSeg" << std::endl;
    return false;
  }

  return true;
}
