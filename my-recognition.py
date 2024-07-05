#!/usr/bin/pyamethon3
import jetson_inference
import jetson_inference_python
import jetson_utils
from jetson_inference import imageNet
from jetson_utils import loadImage
import argparse


parser = argparse.ArgumentParser()
parser.add_argument("filename" , type=str, help="filename of the image to process")
arg = parser.parse_args()

img = loadImage(arg.filename)
net = imageNet(model="resnet18.onnx", labels="labels.txt", input_blob="input_0", output_blob="output_0")

class_idx, confidence = net.Classify(img)
class_desc = net.GetClassDesc(class_idx)
print("Image is recognized as " + str(class_desc) + " (class #) " + str(class_idx) + " with " + str(confidence * 100) + " % confidence.")
