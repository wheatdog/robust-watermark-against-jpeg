#!/usr/bin/env python

import argparse
import cv2
import random
import numpy as np
import json

jpeg_quant = np.array(
    [
        [16, 11, 10, 16, 24, 40, 51, 61],
        [12, 12, 14, 19, 26, 58, 60, 55],
        [14, 13, 16, 24, 40, 57, 69, 56],
        [14, 17, 22, 29, 51, 87, 80, 62],
        [18, 22, 37, 56, 68, 109, 103, 77],
        [24, 35, 55, 64, 81, 104, 113, 92],
        [49, 64, 78, 87, 103, 121, 120, 101],
        [72, 92, 95, 98, 112, 100, 103, 99]
    ]
)

choosable = np.zeros_like(jpeg_quant)
choosable[0,2] = 1
choosable[1,1] = 1
choosable[2,0] = 1
choosable[0,3] = 1
choosable[1,2] = 1
choosable[2,1] = 1
choosable[3,0] = 1

def get_args():
    parser = argparse.ArgumentParser(description="Watermark Embedder")
    parser.add_argument("-i", "--host-image", required=True, type=str)
    parser.add_argument("-w", "--watermark-image", required=True, type=str)
    parser.add_argument("-o", "--output-image", required=True, type=str)
    parser.add_argument("-s", "--secret-file", required=True, type=str)
    parser.add_argument('-r', '--random-seed', default=0, type=int)
    return parser.parse_args()

def torus_automorphism_permutation(img, m, k=1):
    img_result = np.zeros_like(img)

    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            new_i = (i + j) % m
            new_j = (k*i + (k+1)*j) % m
            img_result[new_i, new_j] = img[i, j]

    return img_result

def main(args):

    random.seed(args.random_seed)

    key_info = {}
    img_host = cv2.imread(args.host_image, cv2.IMREAD_COLOR)
    img_mark = cv2.imread(args.watermark_image, cv2.IMREAD_GRAYSCALE)

    _, img_mark = cv2.threshold(img_mark, 0, 1, cv2.THRESH_BINARY)

    for i in range(32):
        for j in range(32):
            print(img_mark[i,j], end=' ')
        print()

    # STEP1
    k = 1
    m = img_mark.shape[0]
    img_mark = torus_automorphism_permutation(img_mark, m=m, k=k)
    key_info['k'] = k
    key_info['m'] = m

    # STEP2
    img_host = cv2.cvtColor(np.float32(img_host), cv2.COLOR_BGR2YUV)
    img_host_y = img_host[:, :, 0]

    # STEP3
    blocks = {}
    for i in range(img_host_y.shape[0] // 8):
        for j in range(img_host_y.shape[1] // 8):
            dct = cv2.dct(img_host_y[i*8:(i+1)*8, j*8:(j+1)*8])
            dct_quant = np.int32(dct / jpeg_quant)
            blocks[(i, j)] = {
                'cnt': np.count_nonzero(dct_quant),
                'dct_q': dct_quant,
                'dct': dct,
            }

    # STEP4
    M = 18
    key_info['M'] = M

    cnt = 0
    blk_info = []
    for blk in sorted(blocks.items(), key=lambda x: x[1]['cnt'], reverse=True):
        #print(blk)
        candidate = np.stack(np.nonzero(blk[1]['dct_q'] * choosable), 1)
        candidate = candidate.tolist()
        # TODO: should be flexible, random smaller than 2
        if len(candidate) == 0:
            continue
        com = random.sample(candidate, 2)
        blk_info.append({
            'blk_pos': blk[0],
            'dct_pos': com,
        })

        #import ipdb; ipdb.set_trace()
        for c in com:
            C = blk[1]['dct'][tuple(c)]
            r = np.abs(C) % M
            q = np.abs(C) // M
            sign = 1 if C >= 0 else -1
            wbit = img_mark[cnt // img_mark.shape[1], cnt % img_mark.shape[1]]

            if wbit:
                r_ = 3 * M / 4
                C_low = sign * ((q-1) * M + r_)
                C_high = sign * (q * M + r_)
            else:
                r_ = M / 4
                C_low = sign * (q * M + r_)
                C_high = sign * ((q+1) * M + r_)

            C_star = C_low if np.abs(C_low - C) <= np.abs(C_high - C) else C_high
            blk[1]['dct'][tuple(c)] = C_star
            cnt += 1
            if cnt >= img_mark.size:
                break

        if cnt >= img_mark.size:
            break

    key_info['blocks'] = blk_info

    # STEP 6
    img_host_y_new = np.zeros_like(img_host_y)
    for blk in blocks.items():
        img_host_y_new[blk[0][0]*8:(blk[0][0]+1)*8, blk[0][1]*8:(blk[0][1]+1)*8] = cv2.idct(blk[1]['dct'])

    # STEP 7
    f = 0
    D = img_host_y - img_host_y_new
    img_host_y_new = img_host_y_new + f * D

    # STEP 8
    img_host = np.concatenate((np.expand_dims(img_host_y_new,-1), img_host[:,:,1:]), -1)
    img_host = cv2.cvtColor(img_host, cv2.COLOR_YUV2BGR)
    cv2.imwrite(args.output_image, img_host)

    with open(args.secret_file, 'w') as f:
        json.dump(key_info, f)

if __name__ == '__main__':
    main(get_args())
