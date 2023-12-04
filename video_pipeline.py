import argparse
import shutil
import subprocess
from pathlib import Path

import numpy as np

from utils import convert_to_images, convert_to_video, draw_trajectory


def process_video(
    video_path: Path,
    output_path: Path,
    device: str,
    fps: int,
    clean: bool,
) -> None:
    output_path = output_path / video_path.stem

    images_raw_path = output_path / "images_raw"
    images_draw_path = output_path / "images_draw"

    images_raw_path.mkdir(parents=True, exist_ok=True)
    images_draw_path.mkdir(parents=True, exist_ok=True)

    # copy video file
    shutil.copyfile(video_path, output_path / video_path.name)

    # convert video to images
    print("Converting the video to images...")
    convert_to_images(video_path, images_raw_path, video_stride=1)

    # detect balls using YOLO
    detect_cmd = [
        "python3",
        "yolov5/detect.py",
        "--weights",
        "models/yolov5s_basketball.pt",
        "--source",
        f"{images_raw_path}/",
        "--save-txt",
        "--save-conf",
        "--nosave",
        "--project",
        f"{output_path.parent}",
        "--name",
        f"{video_path.stem}",
        "--exist-ok",
        "--device",
        f"{device}",
    ]
    subprocess.run(detect_cmd, check=True)

    # draw trajectory
    trajectory = draw_trajectory(
        output_path / "labels",
        images_raw_path,
        images_draw_path,
        ball_conf=0.5,
        max_distance=30,
    )
    trajectory = np.array(trajectory)
    np.savetxt(output_path / "trajectory.txt", trajectory, fmt="%4d %4d")

    # make video
    convert_to_video(
        images_draw_path,
        output_path / f"output_{video_path.stem}.avi",
        fps=fps,
    )

    if clean:
        shutil.rmtree(output_path / "images_raw")
        shutil.rmtree(output_path / "images_draw")
        (output_path / video_path.name).unlink()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("video", type=str, default=None, help="Path to the video file.")
    parser.add_argument(
        "-o", "--output", type=str, default="output", help="Output directory."
    )
    parser.add_argument(
        "-d",
        "--device",
        type=str,
        default="cuda:0",
        help="Device to be used by YOLOv5 model.",
    )
    parser.add_argument("--fps", type=int, default=30, help="FPS of the output video.")
    parser.add_argument(
        "-c",
        "--clean",
        action="store_true",
        default=True,
        help="Remove saved intermediate files.",
    )

    args = parser.parse_args()

    process_video(
        Path(args.video),
        Path(args.output),
        args.device,
        args.fps,
        args.clean,
    )
