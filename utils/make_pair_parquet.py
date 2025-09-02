from pathlib import Path

import polars


def make_pair_parquet(pairs: tuple[tuple[str, str]], save_path: str | Path):
    images = []
    labels = []

    for image_dir, label_dir in pairs:
        exts = ["*.jpg", "*.png"]
        image_dir = Path(image_dir)
        for ext in exts:
            image_files = sorted(image_dir.rglob(ext))
            for image_file in image_files:
                label_file = Path(label_dir).joinpath(image_file.stem + ".npy")
                if label_file.exists():
                    images.append(str(image_file))
                    labels.append(str(label_file))

    df = polars.DataFrame({"image": images, "label": labels})
    df.write_parquet(save_path)


if __name__ == "__main__":
    # pairs = (
    #     (
    #         "/home/kangnam/projects/face_to_parameter/data/v1/images",
    #         "/home/kangnam/projects/face_to_parameter/data/v1/labels",
    #     ),
    #     (
    #         "/home/kangnam/projects/face_to_parameter/data/v2/images",
    #         "/home/kangnam/projects/face_to_parameter/data/v2/labels",
    #     ),
    # )
    # save_path = "data/dataset.parquet"
    # make_pair_parquet(pairs, save_path)

    file = polars.read_parquet("data/dataset.parquet")
    a = file.row(100)
    print(a)
