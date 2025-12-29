/*
 * Copyright (c) 2018-2024, NVIDIA CORPORATION. All rights reserved.
 *
 * Based on DeepStream-Yolo by Marcos Luciano
 * https://www.github.com/marcoslucianops
 *
 * Ported to DeepStream-Yolo-Seg by ddasdkimo
 * https://github.com/ddasdkimo/DeepStream-Yolo-Seg
 */

#ifndef __YOLO_SEG_H__
#define __YOLO_SEG_H__

#include <string>
#include <vector>
#include <cassert>
#include <iostream>
#include <fstream>

#include "NvInfer.h"
#include "nvdsinfer_custom_impl.h"

struct NetworkInfo
{
  std::string inputBlobName;
  std::string onnxFilePath;
  int batchSize;
  std::string int8CalibPath;
  std::string deviceType;
  int numDetectedClasses;
  int clusterMode;
  std::string networkMode;
  float scaleFactor;
  const float* offsets;
  size_t workspaceSize;
  int inputFormat;
};

class YoloSeg
{
public:
  YoloSeg(const NetworkInfo& networkInfo);
  ~YoloSeg();

#if NV_TENSORRT_MAJOR >= 8
  nvinfer1::ICudaEngine* createEngine(nvinfer1::IBuilder* builder, nvinfer1::IBuilderConfig* config);
#else
  nvinfer1::ICudaEngine* createEngine(nvinfer1::IBuilder* builder);
#endif

private:
  std::string m_InputBlobName;
  std::string m_OnnxFilePath;
  int m_BatchSize;
  std::string m_Int8CalibPath;
  std::string m_DeviceType;
  int m_NumDetectedClasses;
  int m_ClusterMode;
  std::string m_NetworkMode;
  float m_ScaleFactor;
  const float* m_Offsets;
  size_t m_WorkspaceSize;
  int m_InputFormat;

  int m_InputC;
  int m_InputH;
  int m_InputW;
};

bool fileExists(const std::string fileName, bool verbose = true);

#endif // __YOLO_SEG_H__
