import argparse
import cv2
import numpy as np

def nc(img1, img2):
    result = np.sum(img1 * img2) / np.sum(img1 * img1)
    return result

def get_args():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("img1", type=str)
    parser.add_argument("img2", type=str)
    return parser.parse_args()

def main(args):
    img1 = cv2.imread(args.img1)
    img2 = cv2.imread(args.img2)
    result = nc(img1, img2)
    print('NC: {:.2f}'.format(result))

if __name__ == '__main__':
    main(get_args())

