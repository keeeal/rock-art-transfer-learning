
import os, json
from random import sample

import numpy as np
import cairocffi as cairo
from PIL import Image

# https://github.com/googlecreativelab/quickdraw-dataset/issues/19#issuecomment-402247262
def vector_to_raster(vector_images, side=28, line_diameter=16, padding=16, bg_color=(0,0,0), fg_color=(1,1,1)):
    """
    padding and line_diameter are relative to the original 256x256 image.
    """

    original_side = 256.

    surface = cairo.ImageSurface(cairo.FORMAT_ARGB32, side, side)
    ctx = cairo.Context(surface)
    ctx.set_antialias(cairo.ANTIALIAS_BEST)
    ctx.set_line_cap(cairo.LINE_CAP_ROUND)
    ctx.set_line_join(cairo.LINE_JOIN_ROUND)
    ctx.set_line_width(line_diameter)

    # scale to match the new size
    # add padding at the edges for the line_diameter
    # and add additional padding to account for antialiasing
    total_padding = padding * 2. + line_diameter
    new_scale = float(side) / float(original_side + total_padding)
    ctx.scale(new_scale, new_scale)
    ctx.translate(total_padding / 2., total_padding / 2.)

    raster_images = []
    for vector_image in vector_images:
        # clear background
        ctx.set_source_rgb(*bg_color)
        ctx.paint()

        bbox = np.hstack(vector_image).max(axis=1)
        offset = ((original_side, original_side) - bbox) / 2.
        offset = offset.reshape(-1,1)
        centered = [stroke + offset for stroke in vector_image]

        # draw strokes, this is the most cpu-intensive part
        ctx.set_source_rgb(*fg_color)
        for xv, yv in centered:
            ctx.move_to(xv[0], yv[0])
            for x, y in zip(xv, yv):
                ctx.line_to(x, y)
            ctx.stroke()

        data = surface.get_data()
        raster_image = np.copy(np.asarray(data)[::4])
        raster_images.append(raster_image)

    return raster_images

def main(path, samples):
    for file in os.listdir(path):
        if file.endswith('.ndjson'):
            dir_path = os.path.join(path, os.path.splitext(file)[0])
            if not os.path.isdir(dir_path): os.mkdir(dir_path)
            with open(os.path.join(path, file)) as f:
                data = f.readlines()

            side = 224
            data = sample(data, samples)
            decode = json.JSONDecoder().decode
            vector_images = [decode(line)['drawing'] for line in data]
            raster_images = vector_to_raster(vector_images, side)
            for n, img in enumerate(raster_images):
                img = Image.fromarray(img.reshape(side, side))
                img.save(os.path.join(dir_path, str(n) + '.png'))

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('path')
    parser.add_argument('--samples', '-n', type=int, default=100)
    main(**vars(parser.parse_args()))
