"""
Usage:
python small_satrn_inference.py vedastr/test.jpg
"""
import argparse
import os
import sys

import cv2

from vedastr.runners import InferenceRunner
from vedastr.utils import Config

from pathlib import Path
from vedastr.helper import download_drive_file

home = str(Path.home())


def load_model():
    cfg_path = os.path.join(
        os.path.dirname(__file__), "vedastr/configs/small_satrn.py"
    )
    cfg = Config.fromfile(cfg_path)

    deploy_cfg = cfg['deploy']
    common_cfg = cfg.get('common')

    checkpoint = os.path.join(
        home, 
        *["OCR", "vedastr", "small_satrn.pth"]
    )
    if not os.path.exists(checkpoint):
        if not os.path.exists(os.path.dirname(checkpoint)):
            os.makedirs(os.path.dirname(checkpoint))
        download_drive_file(file_id="1bcKtEcYGIOehgPfGi_TqPkvrm6rjOUKR", output=checkpoint)

    runner = InferenceRunner(deploy_cfg, common_cfg)
    runner.load_checkpoint(checkpoint)

    # print("model loaded..")
    return runner

def main():
    parser = argparse.ArgumentParser(description='Inference')
    parser.add_argument('image', type=str, help='input image path')
    args = parser.parse_args()

    runner = load_model()

    image_filepath = args.image
    image = cv2.imread(image_filepath)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    pred_str, probs = runner(image)
    # runner.logger.info('predict string: {} \t of {}'.format(pred_str, img))
    print('predict string: {} \t of {}'.format(pred_str, image_filepath))

if __name__ == '__main__':
    main()