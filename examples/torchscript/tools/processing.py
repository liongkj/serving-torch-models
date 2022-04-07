import os
import pathlib

import numpy as np
import torch
from torchocr.datasets.det_modules import ResizeFixedSize, ResizeShortSize
from torchocr.datasets.RecDataSet import RecDataProcess
from torchocr.networks import build_model
from torchocr.postprocess import build_post_process
from torchocr.utils import CTCLabelConverter
from torchocr.utils.vis import draw_ocr_box_txt
from torchvision import transforms


class Dataprocess:
    def __init__(self, args,cfg):
        self.args = args
        # det model preprocess
        self.resize = ResizeFixedSize(736, False)
        self.post_process = build_post_process(cfg["post_process"])
        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=cfg["dataset"]["train"]["dataset"]["mean"],
                    std=cfg["dataset"]["train"]["dataset"]["std"],
                ),
            ]
        )
        
    def preprocess(self,img):
        data = {"img": img, "shape": [img.shape[:2]], "text_polys": []}
        data = self.resize(data)
        tensor = self.transform(data["img"])
        tensor = tensor.unsqueeze(dim=0)
        return tensor,data

    def midprocess(self,out,img,data_shape):
        out = out.cpu().numpy()
        box_list, score_list = self.post_process(out, data_shape)
        box_list, score_list = box_list[0], score_list[0]
        box_list, score_list = self.filter_unimportant(box_list, score_list)
        if len(box_list) > 0:
            idx = [x.sum() > 0 for x in box_list]
            for i, v in enumerate(idx):
                if v:
                    box_list[i] = box_list[i]
                    score_list[i] = score_list[i]

            # box_list = [box_list[i] for i, v in enumerate(idx) if v]
            # score_list = [score_list[i] for i, v in enumerate(idx) if v]
        else:
            box_list, score_list = [], []
        # return box_list, score_list
        draw_box_list = [tuple(map(tuple, box)) for box in box_list]
        imgs = [get_rotate_crop_image(img, box) for box in box_list]
        return imgs, draw_box_list, score_list

    def postprocess(self,img,texts,draw_box_list, score_list ):
        texts = [txt[0][0] for txt in texts]
        debug_img = draw_ocr_box_txt(img, draw_box_list, texts)
        return debug_img

class RecProcess():
    def __init__(self,cfg) -> None:
        # rec model preprocess
        self.process = RecDataProcess(cfg["dataset"]["train"]["dataset"])
        new_path = os.path.join(*pathlib.Path(cfg["dataset"]["alphabet"]).parts[1:])
        self.converter = CTCLabelConverter(new_path)
        
    def postProcess(self,imgs, rec_model):
        if not isinstance(imgs, list):
            imgs = [imgs]
        imgs = [
            self.process.normalize_img(self.process.resize_with_specific_height(img))
            for img in imgs
        ]
        widths = np.array([img.shape[1] for img in imgs])
        idxs = np.argsort(widths)
        txts = []
        for idx in range(0, len(imgs), self.batch_size):
            batch_idxs = idxs[idx : min(len(imgs), idx + self.batch_size)]
            batch_imgs = [
                self.process.width_pad_img(imgs[idx], imgs[batch_idxs[-1]].shape[1])
                for idx in batch_idxs
            ]
            batch_imgs = np.stack(batch_imgs)
            tensor = torch.from_numpy(batch_imgs.transpose([0, 3, 1, 2])).float()
            tensor = tensor.to(self.device)
            with torch.no_grad():
                out = rec_model(tensor)
                out = out.softmax(dim=2)
            out = out.cpu().numpy()
            txts.extend([self.converter.decode(np.expand_dims(txt, 0)) for txt in out])
        # 按输入图像的顺序排序
        idxs = np.argsort(idxs)
        out_txts = [txts[idx] for idx in idxs]
        return out_txts

def get_rotate_crop_image(img, points):
    """
    img_height, img_width = img.shape[0:2]
    left = int(np.min(points[:, 0]))
    right = int(np.max(points[:, 0]))
    top = int(np.min(points[:, 1]))
    bottom = int(np.max(points[:, 1]))
    img_crop = img[top:bottom, left:right, :].copy()
    points[:, 0] = points[:, 0] - left
    points[:, 1] = points[:, 1] - top
    """
    points = points.astype(np.float32)
    img_crop_width = int(
        max(
            np.linalg.norm(points[0] - points[1]), np.linalg.norm(points[2] - points[3])
        )
    )
    img_crop_height = int(
        max(
            np.linalg.norm(points[0] - points[3]), np.linalg.norm(points[1] - points[2])
        )
    )
    pts_std = np.float32(
        [
            [0, 0],
            [img_crop_width, 0],
            [img_crop_width, img_crop_height],
            [0, img_crop_height],
        ]
    )
    M = cv2.getPerspectiveTransform(points, pts_std)
    dst_img = cv2.warpPerspective(
        img,
        M,
        (img_crop_width, img_crop_height),
        borderMode=cv2.BORDER_REPLICATE,
        flags=cv2.INTER_CUBIC,
    )
    dst_img_height, dst_img_width = dst_img.shape[0:2]
    if dst_img_height * 1.0 / dst_img_width >= 1.5:
        dst_img = np.rot90(dst_img)
    return dst_img
