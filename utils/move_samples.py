import argparse
import random
from pathlib import Path

import cv2
from tqdm import tqdm


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_dir", type=str)
    parser.add_argument("--n_samples", type=int, default=100)
    parser.add_argument("--out_dir", type=str)
    parser.add_argument("--resolution", type=int, default=512)
    return parser.parse_args()


def main(args):
    root_dir = Path(args.root_dir)
    out_dir = Path(args.out_dir)
    if not out_dir.is_dir():
        out_dir.mkdir(parents=True, exist_ok=True)

    extensions = ["*.jpg", "*.png"]
    files = []
    for ext in tqdm(extensions, desc="glob"):
        globed = sorted(root_dir.rglob(ext))
        files.extend(globed)

    samples = random.sample(files, args.n_samples)
    idx = 0
    for sample in tqdm(samples):
        image = cv2.imread(str(sample))
        image = cv2.resize(
            image, (args.resolution, args.resolution), interpolation=cv2.INTER_LANCZOS4
        )
        out_file = out_dir.joinpath(f"{idx}".zfill(6) + ".jpg")
        cv2.imwrite(str(out_file), image)
        idx += 1


if __name__ == "__main__":
    args = get_args()
    main(args)
