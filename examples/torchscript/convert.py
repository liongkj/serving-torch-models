# convert model to a torch script module
import argparse
import os
import sys
from ast import parse
from inspect import trace
from re import M

sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))

import cv2
import torch
import torchvision
from examples.torchscript.tools.det_infer import DetInfer
from examples.torchscript.tools.ocr_infer import OCRInfer, init_args
from examples.torchscript.tools.processing import Dataprocess, RecProcess
from torchocr.networks import build_model


def main(args):
    example = cv2.imread(args["img_path"])
    print(example.shape)
    # 1. transcript the det model
    det_jit,det_config = load_det_jit_model(args)
    rec_jit, rec_config = load_rec_jit_model(args,example)
    data_process = Dataprocess(args,det_config)
    rec_data_process = RecProcess(args,rec_config)
    tensor,data = data_process.preprocess(example)
    out = det_jit(tensor)
    if len(out) == 0:
        print("no result")
        return
    # 2. transcript the rec model
    
    imgs, draw_box_list, score_list = data_process.midprocess(out,example,data_shape=data["shape"])
    parts_of_img =  rec_data_process.postProcess(imgs)
    texts = rec_jit(imgs)
    results =  data_process.postprocess(example,texts,draw_box_list, score_list )
    print(results[0])

def load_rec_jit_model(args,example):
    rec_script_path = f"script_{args['rec_path']}"
    ckpt = torch.load(args["rec_path"], map_location="cpu")
    state_dict = {k.replace("module.", ""): v for k, v in ckpt["state_dict"].items()}

    model = build_model(ckpt["cfg"]["model"])
    model.load_state_dict(state_dict)


    scripted_rec_model = torch.jit.trace(model,example)
    # torch.jit.save(scripted_rec_model, rec_script_path)
    return scripted_rec_model, ckpt["cfg"]

def load_det_jit_model(args):
    # det_model = torch.load(args["det_path"])
    det_script_path = f"script_{args['det_path']}"
    ckpt = torch.load(args["det_path"], map_location="cpu")
    state_dict = {k.replace("module.", ""): v for k, v in ckpt["state_dict"].items()}

    model = build_model(ckpt["cfg"]["model"])
    model.load_state_dict(state_dict)

    # det_model = DetInfer()
    scripted_det_model = torch.jit.script(model)
    # torch.jit.save(scripted_det_model, det_script_path)
    return scripted_det_model, ckpt["cfg"]


if __name__ == "__main__":
    args = init_args()
    main(args)
