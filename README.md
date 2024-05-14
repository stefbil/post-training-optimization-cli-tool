# Model Compression and ONNX Conversion

Onnx convertion is being investigated due to issues.

This code allows you to optimize a pretrained VisionEncoderDecoderModel from Hugging Face Transformers.

## Prerequisites

- A GPU with CUDA support is recommended to load the VisionTransformer models
- Python 3.10.6
- CUDA Drivers


## Usage

Prepare and activate a python environment with:

```python
python -m venv optimenv
```

Install required packaged:
```python
pip install -r requirements.txt
```

The code can be run with:

```bash
python cli.py --load PATH_TO_MODEL --save PATH_TO_SAVE  --gpt2
```

The following arguments are available:

- `--gpt2`: Indicated the model that's gonna get optimized
- `--load`: Path to pretrained model. Can be a Hugging Face model link or local path.
- `--save`: Path to save the compressed model files.
<!-- - `--onnx`: Add this flag to also convert the model to ONNX after compressing. -->

For example:

```bash
python cli.py --gpt2 --load nlpconnect/vit-gpt2-image-captioning --save PATH_TO_SAVE
```
or

```cmd
python cli.py -gpt2 -l nlpconnect/vit-gpt2-image-captioning -s PATH_TO_SAVE
```

This will compress the model and save it to `PATH_TO_SAVE`. It will also convert the model to ONNX format and save it to `PATH_TO_SAVE/onnx`.

## Model Compression

The model compression first loads the VisionEncoderDecoderModel and converts it to fp16 to reduce the size.

It then saves the compressed model in the binary .bin format to the specified save path.

The compression typically reduces the model size by 2-4x.

<!-- ## ONNX Conversion

If the `--onnx` flag is added, the model is also converted to the ONNX format after compression. 

This is done using the ORTModelForVision2Seq class to export the model.

The ONNX model is saved to the `onnx` subfolder in the save path.

This allows the model to be used for optimized inference with ONNX Runtime. -->
