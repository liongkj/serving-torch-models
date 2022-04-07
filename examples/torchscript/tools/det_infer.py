# -*- coding: utf-8 -*-
# @Time    : 2020/6/16 10:57
# @Author  : zhoujun
import os
import pathlib
import sys

# 将 torchocr路径加到python陆经里
__dir__ = pathlib.Path(os.path.abspath(__file__))
sys.path.append(str(__dir__))
sys.path.append(str(__dir__.parent.parent))

import torch
from torch import nn
from torchocr.datasets.det_modules import ResizeFixedSize, ResizeShortSize


class DetInfer(nn.Module):
    # def __init__(self, cfg, model):
    #     self.model = model
        
    #     self.model.cpu()
    #     # self.model.eval()
        
    def forward(self, img_tensor):
        # 预处理根据训练来
        
        return self.model(img_tensor)
        

    def filter_unimportant(self, box_list, score_list):
        boxes, scores = [], []
        Y_AREA = 60
        for box, score in zip(box_list, score_list):
            if box[0][1] < Y_AREA or box[1][1] < Y_AREA:
                continue
            boxes.append(box)
            scores.append(score)
        return boxes, scores


def init_args():
    import argparse

    parser = argparse.ArgumentParser(description="PytorchOCR infer")
    parser.add_argument("--model_path", required=True, type=str, help="rec model path")
    parser.add_argument(
        "--img_path", required=True, type=str, help="img dir for predict"
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    import time

    import cv2
    from matplotlib import pyplot as plt
    from torchocr.utils import draw_bbox

    args = init_args()

    model = DetInfer(args.model_path)
    names = next(os.walk(args.img_path))[2]
    st = time.time()
    for name in names:
        path = os.path.join(args.img_path, name)
        img = cv2.imread(path)
        box_list, score_list = model.predict(img)
        out_path = os.path.join(args.img_path, "res", name)
        img = draw_bbox(img, box_list)
        cv2.imwrite(out_path[:-4] + "_res.jpg", img)
    print((time.time() - st) / len(names))
