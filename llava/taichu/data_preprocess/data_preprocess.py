import json
import os
import random
import pandas as pd
import argparse
from loguru import logger

class DataPreprocess(object):
    def __init__(self) -> None:
        pass

    def sample_preset_data(self, src_lines: list, preset_lines: list, preset_data_ratio: float) -> list:
        logger.info("[sample_preset_data] len: {}; [preset_lines] len: {}; [ratio] {}".format(
            len(src_lines), len(preset_lines), preset_data_ratio))

        num = int(len(src_lines) * preset_data_ratio)
        num = num if num < len(preset_lines) else len(preset_lines)
        preset_lines = random.sample(preset_lines, num)

        src_lines = src_lines + preset_lines
        logger.info("[sample_preset_data] len: {}; [preset_lines] len: {}; [ratio] {}".format(
            len(src_lines), len(preset_lines), preset_data_ratio))
        return src_lines


    def data_list_to_dict(self, data_list):
        data_dict = dict()
        data_keys = ["id", "conversations"]
        for data_key in data_keys:
            data_dict[data_key] = list()
            for idx in range(len(data_list)):
                data_dict[data_key].append(data_list[idx][data_key])
        return data_dict


    def data_exchange(self, src_data_list: list) -> list:
        raise NotImplementedError("subclass must override data_exchange")
    
    def data_preprocess(self, input_path, save_json=False, output_path=None, preset_data_path=None, preset_data_ratio=1.0):
        if os.path.exists(input_path):
            with open(input_path, mode="r", encoding="utf-8") as fr:
                src_lines = fr.readlines()

            if preset_data_path is not None:            # 如果preset_data_path不为空，则会对数据进行扩展
                if os.path.exists(preset_data_path) is True:
                    with open(preset_data_path, mode="r", encoding="utf-8") as fr:
                        preset_lines = fr.readlines()
                        src_lines = self.sample_preset_data(src_lines=src_lines,
                                                            preset_lines=preset_lines,
                                                            preset_data_ratio=preset_data_ratio)
                else:
                    logger.warning("[data_preprocess] preset_data_path: {}，文件不存在".format(preset_data_path))

            new_data_list = self.data_exchange(src_data_list=src_lines)
            if save_json and output_path:
                with open(output_path, mode="w", encoding="utf-8") as fw:
                    json.dump(obj=new_data_list, fp=fw, ensure_ascii=False, indent=4)
                logger.info("[data_preprocess] save finish, output_path: {}".format(output_path))
            logger.info("[data_preprocess] exchange finish")
            return new_data_list
        else:
            logger.warning("[data_preprocess] input_path: {}，文件不存在".format(input_path))


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
    DataPreprocess().data_preprocess(input_path=args.input_data, output_path=args.output_data)
