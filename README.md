# DeepStream-Yolo-Seg (with INT8 Calibration Support)

NVIDIA DeepStream SDK 8.0 / 7.1 / 7.0 / 6.4 / 6.3 / 6.2 / 6.1.1 / 6.1 / 6.0.1 / 6.0 application for YOLO-Seg models

--------------------------------------------------------------------------------------------------
### YOLO object detection models and other infos: https://github.com/marcoslucianops/DeepStream-Yolo
--------------------------------------------------------------------------------------------------
### Important: Please export the ONNX model with the new export file, generate the TensorRT engine again with the updated files, and use the new config_infer_primary file according to your model
--------------------------------------------------------------------------------------------------
### **NEW**: INT8 Calibration Support - This fork adds INT8 quantization support for improved inference performance
--------------------------------------------------------------------------------------------------

### Getting started

* [Supported models](#supported-models)
* [Instructions](#basic-usage)
* [INT8 Calibration](#int8-calibration)
* [YOLOv5-Seg usage](docs/YOLOv5_Seg.md)
* [YOLOv7-Seg usage](docs/YOLOv7_Seg.md)
* [YOLOv7-Mask usage](docs/YOLOv7_Mask.md)
* [YOLOv8-Seg usage](docs/YOLOv8_Seg.md)
* [YOLO11-Seg usage](docs/YOLO11_Seg.md)
* [RF-DETR-Seg usage](docs/RFDETR_Seg.md)
* [NMS configuration](#nms-configuration)
* [Detection threshold configuration](#detection-threshold-configuration)

##

### Supported models

* [RF-DETR-Seg](https://github.com/roboflow/rf-detr)
* [YOLO11-Seg](https://github.com/ultralytics/ultralytics)
* [YOLOv8-Seg](https://github.com/ultralytics/ultralytics)
* [YOLOv7-Mask](https://github.com/WongKinYiu/yolov7/tree/mask)
* [YOLOv7-Seg](https://github.com/WongKinYiu/yolov7/tree/u7/seg)
* [YOLOv5-Seg](https://github.com/ultralytics/yolov5)

##

### Instructions

#### 1. Download the DeepStream-Yolo-Seg repo

```
git clone https://github.com/marcoslucianops/DeepStream-Yolo-Seg.git
cd DeepStream-Yolo-Seg
```

#### 2. Compile the libs

2.1. Set the `CUDA_VER` according to your DeepStream version

```
export CUDA_VER=XY.Z
```

* x86 platform

  ```
  DeepStream 8.0 = 12.8
  DeepStream 7.1 = 12.6
  DeepStream 7.0 / 6.4 = 12.2
  DeepStream 6.3 = 12.1
  DeepStream 6.2 = 11.8
  DeepStream 6.1.1 = 11.7
  DeepStream 6.1 = 11.6
  DeepStream 6.0.1 / 6.0 = 11.4
  ```

* Jetson platform

  ```
  DeepStream 8.0 = 13.0
  DeepStream 7.1 = 12.6
  DeepStream 7.0 / 6.4 = 12.2
  DeepStream 6.3 / 6.2 / 6.1.1 / 6.1 = 11.4
  DeepStream 6.0.1 / 6.0 = 10.2
  ```

2.2. Make the libs

**Standard build (FP32/FP16 only):**
```
make -C nvdsinfer_custom_impl_Yolo_seg clean && make -C nvdsinfer_custom_impl_Yolo_seg
```

**Build with INT8 calibration support (requires OpenCV):**
```
make -C nvdsinfer_custom_impl_Yolo_seg clean && make -C nvdsinfer_custom_impl_Yolo_seg OPENCV=1
```

#### 3. Run

```
deepstream-app -c deepstream_app_config.txt
```

**NOTE**: The TensorRT engine file may take a very long time to generate (sometimes more than 10 minutes).

##

### NMS configuration

For now, the NMS is configured in the ONNX exporter file.

**NOTE**: Make sure to set `cluster-mode=4` in the config_infer file.

##

### Detection threshold configuration

The minimum detection confidence threshold is configured in the ONNX exporter file. The `pre-cluster-threshold` should be >= the value used in the ONNX model.

```
[class-attrs-all]
pre-cluster-threshold=0.25
```

##

### INT8 Calibration

INT8 quantization can significantly improve inference performance (typically 10-30% faster than FP16) with minimal accuracy loss.

#### Prerequisites

1. **OpenCV** - Required for image preprocessing during calibration
2. **Calibration images** - Typically 500-1000 representative images from your dataset

#### Step 1: Build with OpenCV support

```bash
export CUDA_VER=12.8  # or your CUDA version
make -C nvdsinfer_custom_impl_Yolo_seg clean && make -C nvdsinfer_custom_impl_Yolo_seg OPENCV=1
```

#### Step 2: Prepare calibration images

Create a text file listing the paths to calibration images (one per line):

```bash
# Create calibration.txt
find /path/to/calibration/images -name "*.jpg" | head -1000 > calibration.txt
```

#### Step 3: Configure config_infer file

Update your `config_infer_primary_*.txt`:

```ini
[property]
...
network-mode=1                                    # 0=FP32, 1=INT8, 2=FP16
int8-calib-file=calib.table                       # Calibration table output
model-engine-file=model_int8.engine               # Engine file name
engine-create-func-name=NvDsInferYoloCudaEngineGet  # Required for custom engine
...
```

#### Step 4: Run calibration

Set environment variables and run DeepStream:

```bash
export INT8_CALIB_IMG_PATH=/path/to/calibration.txt
export INT8_CALIB_BATCH_SIZE=8

deepstream-app -c deepstream_app_config.txt
```

The first run will:
1. Load calibration images
2. Run inference to collect activation statistics
3. Generate `calib.table` file
4. Build and save the INT8 engine

Subsequent runs will reuse the `calib.table` and engine file.

#### Calibration Table Portability

The `calib.table` file contains per-layer quantization scale factors and is **portable across different GPUs**. You can:

1. Generate `calib.table` once on any GPU
2. Copy `calib.table` + ONNX model to other machines (RTX 4090, DGX, etc.)
3. Build INT8 engine on target hardware (engine files are NOT portable)

This saves calibration time when deploying to multiple machines.

##

My projects: https://www.youtube.com/MarcosLucianoTV
