from mazeSolve import MazeSolve, display
import cv2, numpy as np
import time
import sys
import argparse


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Demonstration of different maze solving algorithms"
    )
    parser.add_argument(
        "--visualize",
        "-v",
        action="store_true",
        default=False,
        help="Enable visualization, shows expanded nodes in blue",
    )
    parser.add_argument(
        "--disable-pre-process",
        "-dp",
        action="store_false",
        default=True,
        help="Pre process the image (Remmove rows and columns)",
    )
    parser.add_argument(
        "--speed",
        "-s",
        type=int,
        default=50,
        metavar="X",
        help="Display the image after X frames.Disabled if visualize is disabled",
    )
    parser.add_argument("img", help="Path to input image")
    parser.add_argument(
        "--mitm",
        "-m",
        help="Apply meet in the middle optimization",
        default=False,
        action="store_true",
    )
    parser.add_argument(
        "--algorithm",
        "-a",
        required=True,
        choices=["Astar", "Dijakstra"],
        help="The algorithm to use for solving the maze",
    )
    args = parser.parse_args()
    print(args)
    s = MazeSolve(
        args.img,
        preProcessImage=args.disable_pre_process,
        visualization_speed=args.speed,
        visualization=args.visualize,
    )
    algorithm = ("mitm" if args.mitm else "") + args.algorithm
    start = time.time()
    image = getattr(s, algorithm)()
    print("Elapsed:", time.time() - start)
    display(image, 0)
