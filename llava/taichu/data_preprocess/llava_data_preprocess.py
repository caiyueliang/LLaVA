import json
import os
import random
import pandas as pd
from loguru import logger
import argparse

from data_preprocess import DataPreprocess

class LlavaDataPreprocess(DataPreprocess):
    def __init__(self) -> None:
        super().__init__()

    def data_exchange(self, src_data_list: list):
        pass

def parse_argvs():
    parser = argparse.ArgumentParser(description='data preprocess')
    parser.add_argument("--input_data", type=str, default="./data/开发数据集统一格式/07-医疗问诊/result.json")
    parser.add_argument("--output_data", type=str, default="./result_temp.json")
    # parser.add_argument("--replace_dict", type={}, default={"question": "user", "answer": "assistant"})

    args = parser.parse_args()
    logger.info('[args] {}'.format(args))

    return parser, args


if __name__ == "__main__":
    parser, args = parse_argvs()

    # replace_dict = {"question": "user", "answer": "assistant"}
    LlavaDataPreprocess().data_preprocess(input_path=args.input_data, output_path=args.output_data)