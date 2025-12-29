/*
 * Copyright (c) 2018-2024, NVIDIA CORPORATION. All rights reserved.
 *
 * Based on DeepStream-Yolo by Marcos Luciano
 * https://www.github.com/marcoslucianops
 *
 * Ported to DeepStream-Yolo-Seg by ddasdkimo
 * https://github.com/ddasdkimo/DeepStream-Yolo-Seg
 */

#include "NvOnnxParser.h"

#include "yolo_seg.h"

#ifdef OPENCV
#include "calibrator.h"
#endif

#include <experimental/filesystem>

bool
fileExists(const std::string fileName, bool verbose)
{
  if (!std::experimental::filesystem::exists(std::experimental::filesystem::path(fileName))) {
    if (verbose) {
      std::cout << "\nFile does not exist: " << fileName << std::endl;
    }
    return false;
  }
  return true;
}

YoloSeg::YoloSeg(const NetworkInfo& networkInfo) : m_InputBlobName(networkInfo.inputBlobName),
    m_OnnxFilePath(networkInfo.onnxFilePath), m_BatchSize(networkInfo.batchSize),
    m_Int8CalibPath(networkInfo.int8CalibPath), m_DeviceType(networkInfo.deviceType),
    m_NumDetectedClasses(networkInfo.numDetectedClasses), m_ClusterMode(networkInfo.clusterMode),
    m_NetworkMode(networkInfo.networkMode), m_ScaleFactor(networkInfo.scaleFactor),
    m_Offsets(networkInfo.offsets), m_WorkspaceSize(networkInfo.workspaceSize),
    m_InputFormat(networkInfo.inputFormat), m_InputC(0), m_InputH(0), m_InputW(0)
{
}

YoloSeg::~YoloSeg()
{
}

nvinfer1::ICudaEngine*
#if NV_TENSORRT_MAJOR >= 8
YoloSeg::createEngine(nvinfer1::IBuilder* builder, nvinfer1::IBuilderConfig* config)
#else
YoloSeg::createEngine(nvinfer1::IBuilder* builder)
#endif
{
  assert(builder);

#if NV_TENSORRT_MAJOR < 8
  nvinfer1::IBuilderConfig* config = builder->createBuilderConfig();
  if (m_WorkspaceSize > 0) {
    config->setMaxWorkspaceSize((size_t) m_WorkspaceSize * 1024 * 1024);
  }
#endif

  nvinfer1::NetworkDefinitionCreationFlags flags =
      1U << static_cast<uint32_t>(nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);

  nvinfer1::INetworkDefinition* network = builder->createNetworkV2(flags);
  assert(network);

#if NV_TENSORRT_MAJOR > 8 || (NV_TENSORRT_MAJOR == 8 && NV_TENSORRT_MINOR > 0)
  nvonnxparser::IParser* parser = nvonnxparser::createParser(*network, *builder->getLogger());
#else
  nvinfer1::ILogger& logger = *(nvinfer1::ILogger*)builder;
  nvonnxparser::IParser* parser = nvonnxparser::createParser(*network, logger);
#endif

  if (!parser->parseFromFile(m_OnnxFilePath.c_str(), static_cast<int>(nvinfer1::ILogger::Severity::kWARNING))) {
    std::cerr << "\nCould not parse the ONNX file\n" << std::endl;

#if NV_TENSORRT_MAJOR >= 8
    delete parser;
    delete network;
#else
    parser->destroy();
    config->destroy();
    network->destroy();
#endif

    return nullptr;
  }

  m_InputC = network->getInput(0)->getDimensions().d[1];
  m_InputH = network->getInput(0)->getDimensions().d[2];
  m_InputW = network->getInput(0)->getDimensions().d[3];

  std::cout << "\nModel input shape: " << m_InputC << "x" << m_InputH << "x" << m_InputW << std::endl;

  // Handle dynamic batch size
  if (network->getInput(0)->getDimensions().d[0] == -1) {
    nvinfer1::IOptimizationProfile* profile = builder->createOptimizationProfile();
    assert(profile);
    for (int i = 0; i < network->getNbInputs(); ++i) {
      nvinfer1::ITensor* input = network->getInput(i);
      nvinfer1::Dims inputDims = input->getDimensions();
      nvinfer1::Dims dims = inputDims;
      dims.d[0] = 1;
      profile->setDimensions(input->getName(), nvinfer1::OptProfileSelector::kMIN, dims);
      dims.d[0] = m_BatchSize;
      profile->setDimensions(input->getName(), nvinfer1::OptProfileSelector::kOPT, dims);
      dims.d[0] = m_BatchSize;
      profile->setDimensions(input->getName(), nvinfer1::OptProfileSelector::kMAX, dims);
    }
    config->addOptimizationProfile(profile);
  }

  std::cout << "\nBuilding the TensorRT Engine for Segmentation\n" << std::endl;

  if (m_NetworkMode == "FP16") {
    assert(builder->platformHasFastFp16());
    config->setFlag(nvinfer1::BuilderFlag::kFP16);
    std::cout << "Building with FP16 precision" << std::endl;
  }
  else if (m_NetworkMode == "INT8") {
    assert(builder->platformHasFastInt8());
    config->setFlag(nvinfer1::BuilderFlag::kINT8);
    std::cout << "Building with INT8 precision" << std::endl;

    if (m_Int8CalibPath != "") {
#ifdef OPENCV
      fileExists(m_Int8CalibPath);

      std::string calib_image_list;
      int calib_batch_size;

      if (getenv("INT8_CALIB_IMG_PATH")) {
        calib_image_list = getenv("INT8_CALIB_IMG_PATH");
      }
      else {
        std::cerr << "INT8_CALIB_IMG_PATH not set" << std::endl;
        assert(0);
      }

      if (getenv("INT8_CALIB_BATCH_SIZE")) {
        calib_batch_size = std::stoi(getenv("INT8_CALIB_BATCH_SIZE"));
      }
      else {
        std::cerr << "INT8_CALIB_BATCH_SIZE not set" << std::endl;
        assert(0);
      }

      std::cout << "Using calibration images from: " << calib_image_list << std::endl;
      std::cout << "Calibration batch size: " << calib_batch_size << std::endl;
      std::cout << "Calibration table will be saved to: " << m_Int8CalibPath << std::endl;

      nvinfer1::IInt8EntropyCalibrator2* calibrator = new Int8EntropyCalibrator2(calib_batch_size, m_InputC, m_InputH,
          m_InputW, m_ScaleFactor, m_Offsets, m_InputFormat, calib_image_list, m_Int8CalibPath);
      config->setInt8Calibrator(calibrator);
#else
      std::cerr << "OpenCV is required to run INT8 calibrator" << std::endl;
      assert(0);
#endif
    }
  }
  else {
    std::cout << "Building with FP32 precision" << std::endl;
  }

#if NV_TENSORRT_MAJOR > 8 || (NV_TENSORRT_MAJOR == 8 && NV_TENSORRT_MINOR > 0)
  nvinfer1::IRuntime* runtime = nvinfer1::createInferRuntime(*builder->getLogger());
#else
  nvinfer1::IRuntime* runtime = nvinfer1::createInferRuntime(logger);
#endif

  assert(runtime);

  nvinfer1::IHostMemory* serializedEngine = builder->buildSerializedNetwork(*network, *config);

  if (serializedEngine == nullptr) {
    std::cerr << "Failed to build serialized network" << std::endl;
#if NV_TENSORRT_MAJOR >= 8
    delete parser;
    delete network;
#else
    parser->destroy();
    config->destroy();
    network->destroy();
#endif
    return nullptr;
  }

  nvinfer1::ICudaEngine* engine = runtime->deserializeCudaEngine(serializedEngine->data(), serializedEngine->size());

  if (engine) {
    std::cout << "Building complete\n" << std::endl;
  }
  else {
    std::cerr << "Building engine failed\n" << std::endl;
  }

#if NV_TENSORRT_MAJOR >= 8
  delete serializedEngine;
  delete parser;
  delete network;
#else
  serializedEngine->destroy();
  parser->destroy();
  config->destroy();
  network->destroy();
#endif

  return engine;
}
