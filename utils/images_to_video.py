from pathlib import Path

import cv2
from tqdm import tqdm


def convert_to_video(images_path: Path, output_path: Path, fps: int = 240) -> None:
    filenames = images_path.glob("*.jpg")
    filenames = sorted(filenames, key=lambda x: int(x.stem))

    img = cv2.imread(str(filenames[0]))
    height, width, _ = img.shape
    out = cv2.VideoWriter(
        str(output_path), cv2.VideoWriter_fourcc(*"DIVX"), fps, (width, height)
    )

    progress_bar = tqdm(filenames)
    for filename in progress_bar:
        progress_bar.set_description("Making a video")
        img = cv2.imread(str(filename))
        out.write(img)

    out.release()
