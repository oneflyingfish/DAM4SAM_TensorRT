import os
import glob
import argparse
import shutil
import torch
import numpy as np
import cv2
from PIL import Image

from dam4sam_tracker import DAM4SAMTracker
from utils.visualization_utils import overlay_mask, overlay_rectangle
from utils.box_selector import BoxSelector

init_box = [313, 137, 202, 474]


def run_sequence(dir_path, file_extension, output_dir):
    global init_box
    # Load frames from a given directory
    frames_dir = sorted(glob.glob(os.path.join(dir_path, "*.%s" % file_extension)))

    if len(frames_dir) == 0:
        print("Error: There is no frames in the given directory.")
        exit(-1)

    # Select bounding box using click&hold box drawer
    img0 = cv2.imread(frames_dir[0])

    if not init_box:
        # init by mouse
        box_selector = BoxSelector()
        init_box = box_selector.select_box(img0)

    if not init_box:
        print("Error: Initialization box is not given")
        exit(-1)

    # Create tracker instance
    tracker = DAM4SAMTracker("sam21pp-L")

    # Handle saving output masks to the given output directory
    # Visualize tracking results if output directory is not given
    if output_dir:
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        else:
            shutil.rmtree(output_dir)
            os.makedirs(output_dir)

    # Track frame-by-frame
    print("Segmenting frames...")
    import time

    for i in range(len(frames_dir)):
        img = Image.open(frames_dir[i])
        img_vis = np.array(img)

        start = time.time()
        if i == 0:
            outputs = tracker.initialize(img, None, bbox=init_box)
            if not output_dir:
                overlay_rectangle(img_vis, init_box, color=(255, 0, 0), line_width=2)
                window_name = "win"
                cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
                wait_ = 0
        else:
            outputs = tracker.track(img)
        print(f"cost {i}: {time.time()-start}s.")
        pred_mask = outputs["pred_mask"]  # type: np.ndarray

        if output_dir:
            frame_name = os.path.basename(frames_dir[i])
            dot_idx = frame_name.find(".")
            frame_name = frame_name[:dot_idx]
            output_path = os.path.join(output_dir, "%s.png" % frame_name)

            pred_mask = np.expand_dims(pred_mask, axis=2)
            # pred_mask = np.where(pred_mask > 0.2, 1.0, 0)
            img = cv2.cvtColor(img_vis, cv2.COLOR_RGB2BGR)
            img[:, :, 0:1] = (
                img[:, :, 0:1] * (1 - pred_mask)
                + img[:, :, 0:1] * pred_mask * 0.6
                + pred_mask * 102
            ).astype(np.uint8)
            # mask_img = (
            #     img * (1 - pred_mask) + pred_mask * img * 0.8 + pred_mask * 51
            # ).astype(np.uint8)
            # cv2.imwrite(output_path, pred_mask * 255)
            cv2.imwrite(output_path, img)

    print("Segmentation: Done.")


def main():
    parser = argparse.ArgumentParser(description="Run on a sequence of frames.")
    parser.add_argument(
        "--dir",
        type=str,
        required=False,
        default="/home/yutian/workspace/DAM4SAM_TensorRT/test/imgs",
        help="Path to directory with frames.",
    )
    parser.add_argument("--ext", type=str, default="jpg", help="Image file extension.")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/home/yutian/workspace/DAM4SAM_TensorRT/test/output",
        help="Path to the output directory.",
    )

    args = parser.parse_args()

    run_sequence(args.dir, args.ext, args.output_dir)


if __name__ == "__main__":
    main()
