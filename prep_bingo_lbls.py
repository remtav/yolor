import csv
import random
import shutil
from pathlib import Path

import cv2
import torch

csv_file = Path('/home/remi/Documents/sherbrooke_citoyen/training_img_14oct/bingo_trn.csv')
out_dir = Path('/media/data/bingo')
random.seed(1234)

with open(csv_file, 'r') as f:
    reader = csv.reader(f, delimiter='[')
    for index, row in enumerate(reader):
        num = random.randint(0, 100)
        dataset = 'val' if num < 10 else 'trn'
        file = Path(row[0][:-1])
        #shutil.copy(file, out_dir / 'images' / dataset)
        print(f'File: {file}')
        img = cv2.imread(str(file))
        print(img.shape)
        out_lbl_lines = []
        with open(out_dir / 'labels' / dataset / f'{file.stem}.txt', 'w') as out_lbl:
            for bbox in eval('[' + row[1]):
                x_rel = bbox[0] / img.shape[1]
                y_rel = bbox[1] / img.shape[0]
                w_rel = bbox[2] / img.shape[1]
                h_rel = bbox[3] / img.shape[0]
                xywh_rel = f'0 {x_rel} {y_rel} {w_rel} {h_rel}\n'
                out_lbl_lines.append(xywh_rel)
                print(bbox)
                print(xywh_rel)
                cv2.rectangle(img, (int(x_rel*img.shape[1]), int(y_rel*img.shape[0])), (int((x_rel+w_rel)*img.shape[1]), int((y_rel+h_rel)*img.shape[0])), (0, 255, 0), 2)
            cv2.imshow('img', img)
            cv2.waitKey()
            cv2.destroyAllWindows()
            out_lbl.writelines(out_lbl_lines)
            print('\n')


