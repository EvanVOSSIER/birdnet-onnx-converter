# BirdNET ONNX Converter

Convert and optimize BirdNET models for ONNX Runtime inference on various platforms including GPUs, desktop CPUs, and embedded devices like Raspberry Pi.

## Features

- **TFLite to ONNX conversion** - Convert BirdNET TFLite models to ONNX format
- **GPU optimization** - Replace unsupported ops (RFFT2D, ReverseSequence) with GPU-friendly alternatives
- **Multiple precision formats**:
  - FP32 - Standard precision for GPU/desktop
  - FP16 - Half precision for devices with FP16 support (RPi 5, modern GPUs)
  - INT8 - Quantized for maximum performance on CPU
  - INT8-ARM - ARM-compatible quantization (avoids ConvInteger)

## Installation

```bash
pip install onnx onnx-simplifier onnxslim numpy onnxruntime onnxconverter-common
pip install tensorflow tf2onnx  # For TFLite conversion
```

## Usage

### Step 1: Convert TFLite to ONNX

```bash
python convert.py --input BirdNET_Model.tflite --output-dir ./ --onnx-only
```

### Step 2: Optimize ONNX Model

```bash
# Output all formats (FP32, FP16, INT8)
python optimize.py --input BirdNET_Model.onnx --output BirdNET

# Output only FP32
python optimize.py --input BirdNET_Model.onnx --output BirdNET --fp32-only

# Include ARM-compatible INT8
python optimize.py --input BirdNET_Model.onnx --output BirdNET --int8-arm
```

### Output Files

| File | Description | Recommended Use |
|------|-------------|-----------------|
| `*_fp32.onnx` | Full precision | GPU (CUDA/TensorRT), Desktop CPU |
| `*_fp16.onnx` | Half precision | RPi 5, Modern GPUs |
| `*_int8.onnx` | INT8 quantized | Intel CPUs |
| `*_int8_arm.onnx` | ARM-compatible INT8 | Raspberry Pi, ARM devices |

## Key Optimizations

The optimizer applies the following transformations:

1. **RFFT2D → MatMul** - Replaces TensorFlow's RFFT2D with precomputed DFT matrix multiplication
2. **ReverseSequence → Slice** - Replaces with negative-stride slice operation
3. **GlobalAveragePool + Squeeze → ReduceMean** - Combines into single operation
4. **Transpose fusion** - Merges consecutive transpose operations
5. **Cast removal** - Removes redundant type casts
6. **Graph optimization** - Uses onnx-simplifier and onnxslim for further optimization

## Supported Models

This tool supports conversion and optimization of:

- **Official BirdNET models** - Download from [Zenodo](https://zenodo.org/records/15050749) (v2.4 with 6522 species)
- **Custom classifiers** - Models trained with [BirdNET Analyzer](https://github.com/birdnet-team/BirdNET-Analyzer) for additional species or regional variants
- **Any BirdNET variant** - Models using the same backbone architecture

## Platform-Specific Notes

### Raspberry Pi 5

Use FP16 model for best performance:
```python
import onnxruntime as ort
session = ort.InferenceSession("BirdNET_fp16.onnx")
```

### Raspberry Pi 3/4

Use INT8-ARM model (FP16 not supported):
```python
session = ort.InferenceSession("BirdNET_int8_arm.onnx")
```

### Intel CPUs

Use INT8 model for ~2x speedup with oneDNN:
```python
session = ort.InferenceSession("BirdNET_int8.onnx")
```

### NVIDIA GPUs

Use FP32 or FP16 with CUDA/TensorRT:
```python
session = ort.InferenceSession(
    "BirdNET_fp16.onnx",
    providers=['CUDAExecutionProvider']
)
```

## Acknowledgments

This project is based on [Justin Chu's optimized ONNX conversion](https://huggingface.co/justinchuby/BirdNET-onnx) of the BirdNET v2.4 model. His work on replacing TensorFlow-specific operations with ONNX-compatible alternatives made GPU inference possible.

## References

- [BirdNET Analyzer](https://github.com/birdnet-team/BirdNET-Analyzer) - Official BirdNET tool for training custom classifiers
- [BirdNET Models on Zenodo](https://zenodo.org/records/15050749) - Official model downloads
- [Justin Chu's BirdNET-ONNX](https://huggingface.co/justinchuby/BirdNET-onnx) - Original ONNX conversion work
- [ONNX Runtime](https://onnxruntime.ai/)

## License

MIT License
