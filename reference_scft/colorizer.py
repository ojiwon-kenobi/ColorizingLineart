import torch
import torch.nn as nn
import argparse
import cv2 as cv
import numpy as np

from pathlib import Path
from tqdm import tqdm
from model import Generator


def preprocess(path: Path) -> torch.Tensor:
    img = cv.imread(str(path))
    img = (img[:, :, ::-1].transpose(2, 0, 1) - 127.5) / 127.5
    img = np.expand_dims(img, axis=0)
    img = torch.cuda.FloatTensor(img.astype(np.float32))
    print("preprocess")
    return img

def img_save(img: np.array, outdir: Path):
    img = np.clip(img[0] * 127.5 + 127.5, 0, 255).astype(np.uint8).transpose(1, 2, 0)
    img = img[:, :, ::-1]
    print("img_save")

    cv.imwrite(f"{outdir}/inference.png", img)

def single_inference(line_path: Path,
                     style_path: Path,
                     pretrained_path: Path,
                     outdir: Path):
    model = Generator()
    weight = torch.load(pretrained_path)
    model.load_state_dict(weight)
    model.cuda()
    model.eval()
    print("single_inference")

    line = preprocess(line_path)
    style = preprocess(style_path)

    with torch.no_grad():
        y = model(line, style)

    y = y.detach().cpu().numpy()

    img_save(y, outdir)

if __name__ == "__main__":
    print("main")

    parser = argparse.ArgumentParser(description="Style2paint")
    parser.add_argument("--outdir", type=Path, default="inferdir", help="output directory")
    parser.add_argument("--line_path", type=Path, help="data path")
    parser.add_argument("--ref_path", type=Path, help="sketch path")
    parser.add_argument("--pre", type=Path, help="pretrained path")

    args = parser.parse_args()
    outdir = args.outdir
    outdir.mkdir(exist_ok=True)

    single_inference(args.line_path,
                     args.ref_path,
                     args.pre,
                     outdir,
                     )
