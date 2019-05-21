#!/usr/bin/env python

import argparse
import cv2
import json
import numpy as np

def inverse_torus_automorphism_permutation(img, m, k):
    img_result = np.zeros_like(img)

    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            new_i = ((k+1)*i - j) % m
            new_j = (-k*i + j) % m
            img_result[new_i, new_j] = img[i, j]

    return img_result

def get_args():
    parser = argparse.ArgumentParser(description="Watermark Extractor")
    parser.add_argument("-i", "--image", required=True, type=str)
    parser.add_argument("-s", "--secret-file", required=True, type=str)
    parser.add_argument("-o", "--output-image", required=True, type=str)
    return parser.parse_args()

def main(args):
    with open(args.secret_file) as f:
        key_info = json.load(f)

    k = key_info['k']
    m = key_info['m']
    M = key_info['M']

    img = cv2.imread(args.image, cv2.IMREAD_COLOR)
    img = cv2.cvtColor(np.float32(img), cv2.COLOR_BGR2YUV)
    img_y = img[:, :, 0]

    wbits = np.zeros((m, m))
    cnt = 0
    for blk in key_info['blocks']:
        i, j = blk['blk_pos']
        dct = cv2.dct(img_y[i*8:(i+1)*8, j*8:(j+1)*8])
        for pos in blk['dct_pos']:
            if (np.abs(dct[tuple(pos)]) % M) < (M / 2):
                bit = 0
            else:
                bit = 1
            wbits[cnt // m, cnt % m] = bit
            cnt += 1

    wbits = inverse_torus_automorphism_permutation(wbits, m, k)

    cv2.imwrite(args.output_image, wbits*255)

if __name__ == '__main__':
    main(get_args())
