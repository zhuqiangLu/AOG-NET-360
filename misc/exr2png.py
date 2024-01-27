import os 
os.environ["OPENCV_IO_ENABLE_OPENEXR"]="1"
import cv2
import numpy as np 
from PIL import Image 
from einops import rearrange
from tqdm import tqdm
from joblib import Parallel, delayed
from rich.progress import track
import argparse

class TonemapHDR(object):
    """
        Tonemap HDR image globally. First, we find alpha that maps the (max(numpy_img) * percentile) to max_mapping.
        Then, we calculate I_out = alpha * I_in ^ (1/gamma)
        input : nd.array batch of images : [H, W, C]
        output : nd.array batch of images : [H, W, C]
    """

    def __init__(self, gamma=2.4, percentile=50, max_mapping=0.5):
        self.gamma = gamma
        self.percentile = percentile
        self.max_mapping = max_mapping  # the value to which alpha will map the (max(numpy_img) * percentile) to

    def __call__(self, numpy_img, clip, alpha=None, gamma=True):
        if gamma:
            power_numpy_img = np.power(numpy_img, 1 / self.gamma)
        else:
            power_numpy_img = numpy_img
        non_zero = power_numpy_img > 0
        if non_zero.any():
            r_percentile = np.percentile(power_numpy_img[non_zero], self.percentile)
        else:
            r_percentile = np.percentile(power_numpy_img, self.percentile)
        if alpha is None:
            alpha = self.max_mapping / (r_percentile + 1e-10)
        tonemapped_img = np.multiply(alpha, power_numpy_img)

        if clip:
            tonemapped_img = np.clip(tonemapped_img, 0, 1)

        return tonemapped_img.astype('float32')


def process_exr(source, dest):
    processor = TonemapHDR()
    image = cv2.imread(source, cv2.IMREAD_UNCHANGED) 
    image = image[..., ::-1]
    image = processor(image, True)
    # image = rearrange(image, 'h w c -> c h w')
    image = (image * 255).astype(np.uint8)
    image = Image.fromarray(image)
    image.save(dest)
    print(dest)
    



if __name__ == '__main__':
    parser = argparse.ArgumentParser() 
    parser.add_argument("--image_dir", type=str, default=None, required=True)
    parser.add_argument("--out_dir", type=str, default=None, required=True)
    parser.add_argument("--num_workers", type=int, default=60, required=False)
    args = parser.parse_args() 

    source = args.image_dir
    dest = args.out_dir
    os.makedirs(dest, exist_ok=True)
    executor = Parallel(n_jobs=args.num_workers)
    tasks = (delayed(process_exr)(os.path.join(source, exr), os.path.join(dest, exr.replace('.exr', '.png'))) for exr in os.listdir(source))
    executor(tasks)


