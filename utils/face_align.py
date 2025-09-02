import argparse
from pathlib import Path

import cv2
import numpy as np
from facenet_pytorch import MTCNN


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_dir", type=str)
    parser.add_argument("--out_dir", type=str)
    parser.add_argument("--resolution", type=int, default=512)
    return parser.parse_args()


def align_face(image, cropper, is_rgb=False):
    if not is_rgb:
        img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    else:
        img = image
    batch_boxes, batch_probs, batch_points = cropper.detect(img, landamrks=True)

    if not cropper.keep_all:
        batch_boxes, batch_probs, batch_points = cropper.select_boxes(
            batch_boxes, batch_probs, batch_points, img, method=cropper.selection_method
        )
    if batch_points is None:
        return image
    left_eye = batch_points[0, 1]
    right_eye = batch_points[0, 0]
    d = left_eye - right_eye
    angle = np.degrees(np.arctan2(d[1], d[0]))
    center = (0.5 * (left_eye + right_eye)).astype(np.float32)
    M = cv2.getRotationMatrix2D(tuple(center), angle, 1)
    aligned = cv2.warpAffine(
        image, M, (img.shape[1], img.shape[0]), flags=cv2.INTER_LANCZOS4
    )
    return aligned


def get_box(image, cropper):
    batch_boxes, batch_probs, batch_points = cropper.detect(image, landamrks=True)

    if not cropper.keep_all:
        batch_boxes, batch_probs, batch_points = cropper.select_boxes(
            batch_boxes,
            batch_probs,
            batch_points,
            image,
            method=cropper.selection_method,
        )
    return batch_boxes


def crop(image, box):
    x1, y1, x2, y2 = box
    x1 = round(x1)
    y1 = round(y1)
    x2 = round(x2)
    y2 = round(y2)
    cropped = image[y1:y1, x1:x2]
    return cropped


def pad_if_needed(image, target=(512, 512)):
    h, w, _ = image.shape
    target_w, target_h = target
    base_pic = np.zeros((target_h, target_w, 3), np.uint8)
    ratio_h = target_h / h
    ratio_w = target_w / w
    if ratio_w < ratio_h:
        new_size = (int(w * ratio_w), int(h * ratio_w))
    else:
        new_size = (int(w * ratio_h), int(h * ratio_h))
    new_w, new_h = new_size
    image = cv2.resize(image, dsize=new_size, interpolation=cv2.INTER_LANCZOS4)
    base_pic[
        int(target_h / 2 - new_h / 2) : int(target_h / 2 + new_h / 2),
        int(target_w / 2 - new_w / 2) : int(target_w / 2 + new_w / 2),
        :,
    ] = image
    return base_pic


def main(args):
    cropper = MTCNN(post_process=False)
    out_dir = Path(args.out_dir)
    if not out_dir.is_dir():
        out_dir.mkdir(parents=True, exist_ok=True)
    extensions = ["*.jpg", "*.png"]
    files = []
    for ext in extensions:
        globed = list(Path(args.root_dir).rglob(ext))
        files.extend(globed)
    files = sorted(files)

    idx = 0
    for file in files:
        image = cv2.imread(str(file))
        h, w, _ = image.shape

        aligned = align_face(image, cropper)
        boxes = get_box(aligned, cropper)
        x1, y1, x2, y2 = boxes[0]
        x1 = np.clip(x1, 0, w)
        y1 = np.clip(y1, 0, h)
        box = (x1, y1, abs(x2), abs(y2))
        cropped = crop(image, box)
        resized = pad_if_needed(cropped, (512, 512))
        bgr = cv2.cvtColor(resized, cv2.COLOR_RGB2BGR)
        save_path = out_dir.joinpath(f"{idx}".zfill(6) + ".jpg")
        cv2.imwrite(str(save_path), bgr)
        idx += 1


if __name__ == "__main__":
    args = get_args()
    main(args)
