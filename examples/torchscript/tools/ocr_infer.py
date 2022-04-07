import argparse

import cv2
import numpy as np
import torch
import torch.nn as nn
from examples.torchscript.tools.det_infer import DetInfer
from examples.torchscript.tools.rec_infer import RecInfer
from line_profiler import LineProfiler
from memory_profiler import profile


# @torch.jit.script
class OCRInfer(nn.Module):
    def __init__(
        self,
        det_path,
        rec_path,
        rec_batch_size=16,
        # time_profile=False,
        # mem_profile=False,
    ):
        super().__init__()
        self.det_model = det_path
        # self.rec_model = torch.jit.script(RecInfer(rec_path, rec_batch_size))
        # self.det_model = DetInfer(det_path)
        self.rec_model = RecInfer(rec_path, rec_batch_size)
        # assert not (
        #     time_profile and mem_profile
        # ), "can not profile memory and time at the same time"
        # self.line_profiler = None
        # if time_profile:
        #     self.line_profiler = LineProfiler()
        #     self.predict = self.predict_time_profile
        # if mem_profile:
        #     self.predict = self.predict_mem_profile

    def do_predict(self, tensor):
        box_list, score_list = self.det_model.predict(tensor)
        


        
        texts = self.rec_model.predict(imgs)
        
        return texts, score_list, debug_img

    def forward(self, img):
        return self.do_predict(img)

    # def predict_mem_profile(self, img):
    #     wapper = profile(self.do_predict)
    #     return wapper(img)

    # def predict_time_profile(self, img):
    #     # run multi time
    #     for i in range(8):
    #         print("*********** {} profile time *************".format(i))
    #         lp = LineProfiler()
    #         lp_wrapper = lp(self.do_predict)
    #         ret = lp_wrapper(img)
    #         lp.print_stats()
    #     return ret


def init_args():
    import argparse

    parser = argparse.ArgumentParser(description="OCR infer")
    parser.add_argument(
        "--det_path",
        type=str,
        help="det model path",
        default="models/det_db_mbv3_new.pth",
    )
    parser.add_argument(
        "--rec_path",
        type=str,
        help="rec model path",
        default="models/ch_rec_moblie_crnn_mbv3.pth",
    )
    parser.add_argument(
        "--img_path",
        type=str,
        help="img path for predict",
        default="data/ic2.jpg",
    )
    parser.add_argument("--rec_batch_size", type=int, help="rec batch_size", default=16)
    parser.add_argument(
        "-time_profile", action="store_true", help="enable time profile mode"
    )
    parser.add_argument(
        "-mem_profile", action="store_true", help="enable memory profile mode"
    )
    args,_ = parser.parse_known_args()

    return vars(args)


def post_process(debug_img):
    (
        h,
        w,
        _
    ) = debug_img.shape
    raido = 600.0 / w if w > 1200 else 1
    debug_img = cv2.resize(debug_img, (int(w * raido), int(h * raido)))
    return debug_img

if __name__ == "__main__":
    import cv2

    args = init_args()
    img = cv2.imread(args["img_path"])
    model = OCRInfer(**args)
    txts, boxes, debug_img = model.predict(img)
    debug_img = post_process(debug_img)
    if not (args["mem_profile"] or args["time_profile"]):
        cv2.imshow("debug", debug_img)
        cv2.waitKey()
