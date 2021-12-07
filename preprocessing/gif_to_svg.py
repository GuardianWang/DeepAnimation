from PIL import Image
import os
from os import listdir
from os.path import isfile, join
import argparse
from tqdm import tqdm
from math import floor


def extract_frames(input_folder_name, output_folder_name, gif_name, num_frames):
    with Image.open(input_folder_name + "/" + gif_name) as im:
        for i in range(num_frames):
            im.seek(int(floor(im.n_frames / num_frames * i)))
            im.save('%s-%i.png' % (output_folder_name + "/pngs/" + gif_name.split(".")[0], i))


if __name__ == "__main__":
    # Parse arguments
    arg_parser = argparse.ArgumentParser(description="Converts animated GIFs to SVG frames")
    arg_parser.add_argument("-i", "--input", type=str, required=True,
                            help="input directory of gifs")
    arg_parser.add_argument("-o", "--output", type=str, required=True,
                            help="output directory of svgs")
    arg_parser.add_argument("-n", "--num_frames", type=str, required=True,
                            help="desired number of frames")
    args = vars(arg_parser.parse_args())

    if not os.path.exists(args["output"]):
        os.makedirs(args["output"])
    if not os.path.exists(args["output"]+"/pngs"):
        os.makedirs(args["output"]+"/pngs")
    if not os.path.exists(args["output"]+"/svgs"):
        os.makedirs(args["output"]+"/svgs")

    gif_files = [f for f in listdir(args["input"]) if isfile(join(args["input"], f))]
    for f in tqdm(gif_files):
        extract_frames(args["input"], args["output"], f, int(args["num_frames"]))

    png_files = [f for f in listdir(args["output"]+"/pngs") if isfile(join(args["output"]+"/pngs", f))]
    for f in png_files:
        print("Converting %s..." % f)
        os.system("vtracer --input %s --output %s -m polygon" % (args["output"] + "/pngs/" + f, args["output"] + "/svgs/" + f.split(".")[0]+".svg"))
