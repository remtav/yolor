import csv
import random
import shutil
from pathlib import Path

import cv2
#import pandas
import torch
from tqdm import tqdm

csv_file = Path('/home/remi/Documents/sherbrooke_citoyen/training_img_14oct/bingo_trn.csv')
out_root = Path('/media/data/bingo')
val_perc = 25  # percentage of training data going to validation rather than training
random.seed(1234)

#bingo_data = pandas.read_csv(csv_file, sep=';', header=None)

out_img_dir = out_root / 'images'
out_img_dir.mkdir(exist_ok=True, parents=True)
out_gt_dir = out_root / 'labels'
out_gt_dir.mkdir(exist_ok=True, parents=True)

with open(csv_file, 'r') as f:
    reader = csv.reader(f, delimiter=';')
    for index, row in tqdm(enumerate(reader)):
        # determine which dataset it will belong to
        num = random.randint(0, 100)
        dataset = 'val' if num < val_perc else 'trn'
        out_img_dir_dataset = out_img_dir / dataset
        out_img_dir_dataset.mkdir(exist_ok=True)
        out_gt_dir_dataset = out_gt_dir / dataset
        out_gt_dir_dataset.mkdir(exist_ok=True)

        # read info from csv starting with file path
        row_splits = row[0].split('.png')
        file = Path(f"{row_splits[0]}.png")  # add ".png" is a work around because csv is not well written
        # check file's existence
        if not file.is_file():
            raise FileNotFoundError(f'Not found: {file}')
        # copy to destination folder
        if not (out_img_dir_dataset / file.name).is_file():
            shutil.copy(file, out_img_dir)
            print(f'Copied: {file}')
        # start preparing label files
        img = cv2.imread(str(file))
        # print(img.shape)
        out_lbl_lines = []
        lbl_file = out_gt_dir_dataset / f'{file.stem}.txt'
        if not lbl_file.is_file():
            with open(out_gt_dir_dataset / f'{file.stem}.txt', 'w') as out_lbl:
                for bbox in eval(row_splits[1]):
                    # make bboxes relative to image shape and place xy's in center of box, not upper right corner
                    # (see coco dataset convention for xywh labeling)
                    x_rel = (bbox[0] + bbox[2]/2) / img.shape[1]
                    y_rel = (bbox[1] + bbox[3]/2) / img.shape[0]
                    w_rel = bbox[2] / img.shape[1]
                    h_rel = bbox[3] / img.shape[0]
                    xywh_rel = f'0 {x_rel} {y_rel} {w_rel} {h_rel}\n'
                    out_lbl_lines.append(xywh_rel)
                    # print(bbox)
                    # print(xywh_rel)
                    #cv2.rectangle(img, (int((x_rel-w_rel/2)*img.shape[1]), int((y_rel-h_rel/2)*img.shape[0])), (int((x_rel+w_rel/2)*img.shape[1]), int((y_rel+h_rel/2)*img.shape[0])), (0, 255, 0), 2)
                # for debugging
                # cv2.imshow('img', img)
                # cv2.waitKey()
                # cv2.destroyAllWindows()
                out_lbl.writelines(out_lbl_lines)
                print('\n')

for dataset in ['trn', 'val']:
    with open(out_root / f'{dataset}.txt', 'a') as f2:
        for file in (out_img_dir / dataset).iterdir():
            f2.write(f'{file}\n')



