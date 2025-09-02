import argparse
from pathlib import Path

# from facenet_pytorch import InceptionResnetV1
import numpy as np
import torch
from PIL import Image
from torchvision.models import MobileNet_V3_Large_Weights, mobilenet_v3_large
from torchvision.transforms import v2
from tqdm import tqdm


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_dir", type=str)
    parser.add_argument("--save_dir", type=str)
    args = parser.parse_args()
    return args


class LabelMaker:
    def __init__(self, length=37, value_range=(-1, 1)):
        self.length = length
        self.value_range = value_range
        # self.encoder = InceptionResnetV1(pretrained="vggface2").eval()
        self.encoder = self.load_encoder()
        self.prerprocess_transform = v2.Compose(
            [
                v2.Resize(224, interpolation=v2.InterpolationMode.LANCZOS),
                v2.ToImage(),
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )

    def write(self, path, label):
        with open(path, "w", encoding="utf-8") as f:
            for line in label:
                f.write(f"{line}\n")

    def save_npy(self, path, label):
        np.save(path, label.detach().cpu().numpy())

    def prerprocess(self, image_path):
        image = Image.open(image_path).convert("RGB")
        image = self.prerprocess_transform(image)
        return image

    def embed(self, image_path):
        data = self.prerprocess(image_path).to("cuda:0")
        data = data.unsqueeze(0)
        embed = self.encoder(data)
        return embed.squeeze(0)

    def load_encoder(self):
        model = mobilenet_v3_large(weights=MobileNet_V3_Large_Weights.IMAGENET1K_V2)
        model.classifier = torch.nn.Sequential(torch.nn.Identity())
        model.eval()
        model.to("cuda:0")
        model.compile()
        return model


def main(args):
    maker = LabelMaker()

    image_dir = Path(args.image_dir)
    label_dir = Path(args.save_dir)
    label_dir.mkdir(parents=True, exist_ok=True)

    image_files = []
    for ext in ["*.jpg", "*.png"]:
        globed = image_dir.rglob(ext)
        image_files.extend(sorted(globed))

    image_files = sorted(image_files)
    for file in tqdm(image_files):
        label = maker.embed(file)
        label_path = label_dir.joinpath(file.name.replace(file.suffix, ".npy"))
        maker.save_npy(str(label_path), label)


if __name__ == "__main__":
    args = get_args()
    main(args)
