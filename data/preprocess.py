
import os
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm

def main(path):
    image_files = []
    
    for root, folders, files in os.walk(path):
        for file in map(lambda f: Path(root) / f, files):
            if file.suffix.lower() in {'.jpg', '.png'}:
                image_files.append(file)

    for image_path in tqdm(image_files, desc=str(path)):

        # load image as 32 bit float array
        image = np.array(Image.open(image_path)).astype(np.float32)

        # remove third dimension if present
        if len(image.shape) == 3: image = image.max(2)

        # normalise to range [0, 1]
        image = image/image.max()

        # set background colour to black
        if image.reshape(-1)[0] > 0.5: image = 1 - image

        # convert to 8 bit unsigned mask in range (0, 255)
        image = 255.*(image >= image.mean())
        image = Image.fromarray(image.astype(np.uint8))

        # scale
        image = image.resize((224, 224))

        # save as png
        os.remove(image_path)
        new_path = os.path.splitext(image_path)[0] + '.png'
        image.save(new_path, 'PNG')

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('path', type=Path)
    main(**vars(parser.parse_args()))
