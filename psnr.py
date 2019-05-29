import argparse
import cv2
import numpy as np

def psnr(img1, img2):
    mse = np.mean((img1 - img2)**2)
    if mse == 0:
        return 100
    return 20 * np.log10(255.0/np.sqrt(mse))

def get_args():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("img1", type=str)
    parser.add_argument("img2", type=str)
    return parser.parse_args()

def main(args):
    img1 = cv2.imread(args.img1)
    img2 = cv2.imread(args.img2)
    result = psnr(img1, img2)
    print('PSNR: {:.2f}'.format(result))

if __name__ == '__main__':
    main(get_args())
