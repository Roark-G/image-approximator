"""
Approximate an image with patches from the command line.

Example Usage:
    python main.py path/to/target.jpg
    python main.py path/to/target.jpg -n 500 --start-mode white --patches patch_palettes/basic_patches -o out.png
"""
import argparse

from image_approximator import ImageApproximator, RunParameters


def parse_args():
    parser = argparse.ArgumentParser(description="approximate an image with patches")
    parser.add_argument("target", help="path to the target image")
    parser.add_argument("-n", "--n-patches", type=int, default=250,
                        help="number of patches to place (default: 250)")
    parser.add_argument("-o", "--output", default="approx.png",
                        help="where to save the result (default: approx.png)")
    parser.add_argument("-s", "--start-mode", default="average",
                        help='"white", "average", or a path to a start image (default: average)')
    parser.add_argument("-p", "--patches", default="patch_palettes/basic_patches",
                        help="folder of patch PNGs (default: patch_palettes/basic_patches)")
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="enable verbose output")
    args = parser.parse_args()
    if args.n_patches < 1:
        parser.error("--n-patches must be at least 1")
    return args

def main():
    args = parse_args()

    p = RunParameters()
    p.start_mode = args.start_mode

    approx = ImageApproximator()
    approx.run(args.target, args.patches, n_patches=args.n_patches, parameters=p, verbose=args.verbose)
    approx.export_img(args.output)


if __name__ == "__main__":
    main()