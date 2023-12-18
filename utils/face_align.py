import argparse
from pathlib import Path
import cv2
from facenet_pytorch import MTCNN

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_dir", type=str)
    parser.add_argument("--resolution", type=int, default=512)
    return parser.parse_args()

def face_align():
    pass

def face_crop():
    pass

def pad_if_needed():
    pass

def main():
    cropper = MTCNN(post_process=False, image_size=)

if __name__ == "__main__":
    main()