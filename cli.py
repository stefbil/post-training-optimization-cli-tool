import argparse
from transformers import VisionEncoderDecoderModel, AutoTokenizer, pipeline
import torch
from transformers.onnx import FeaturesManager
# from optimum.onnxruntime import ORTModelForVision2Seq
import logging


logger = logging.getLogger(__name__)

def compress_model(load_model_path, save_path):
    logger.info(f"Loading model from {load_model_path}")
    model = VisionEncoderDecoderModel.from_pretrained(load_model_path, torch_dtype=torch.float16)
    # save_path = "D:\\Stef\\Network compression\Models"
    print(model.get_memory_footprint()/1000000000)
    save_path += "\\bin"
    logger.info(f"Saving .bin model to {save_path}")
    model.save_pretrained(save_path)


# def convert_onnx(load_model_path, save_path):
#     # Load the model from the hub and export it to the ONNX format
#     model = ORTModelForVision2Seq.from_pretrained(load_model_path, export=True)
#     # Save the converted model
#     save_path += "\\onnx"
#     logger.info(f"Saving .onnx model to {save_path}")
#     model.save_pretrained(save_path)
    
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpt2","-gpt2", action="store_true")
    parser.add_argument("--load", "-l",help="Path to pretrained model (huggingface link or local path)",type=str)
    parser.add_argument("--save","-s",help="Path to save compressed model",type=str)
    # parser.add_argument("--onnx","-onnx",action="store_true")
    return parser.parse_args()

def main():
    args = parse_args()

    load_path = (args.load)
    save_path = (args.save)

    if args.gpt2:
        compress_model(load_path, save_path)
        # if args.onnx:
        #     convert_onnx(load_path, save_path)
    else:
        exit()


if __name__ == "__main__":
    main()
