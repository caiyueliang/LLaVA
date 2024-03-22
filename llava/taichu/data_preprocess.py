import json
import os
import random
import pandas as pd
import argparse
from loguru import logger


def sample_preset_data(src_lines: list, preset_lines: list, preset_data_ratio: float) -> list:
    logger.info("[src_lines] len: {}; [preset_lines] len: {}; [ratio] {}".format(
        len(src_lines), len(preset_lines), preset_data_ratio))

    num = int(len(src_lines) * preset_data_ratio)
    num = num if num < len(preset_lines) else len(preset_lines)
    preset_lines = random.sample(preset_lines, num)

    src_lines = src_lines + preset_lines
    logger.info("[src_lines] len: {}; [preset_lines] len: {}; [ratio] {}".format(
        len(src_lines), len(preset_lines), preset_data_ratio))
    return src_lines


def data_exchange(src_data_list: list):
    new_data_list = list()
    src_data_len = len(src_data_list)
    logger.info("[data_exchange] src_data_len: {}".format(src_data_len))

    i = 0
    for conversation_text in src_data_list:
        try:
            if i == 0:
                logger.info("[data_exchange][conversation_text] {}".format(conversation_text))
            conversation_dict = json.loads(conversation_text)
            new_conversation_list = list()
            if i == 0:
                logger.info("[data_exchange][conversation_dict] before: {}".format(conversation_dict))

            if "conversations" in conversation_dict.keys():
                for dialog_dict in conversation_dict["conversations"]:
                    if dialog_dict["from"] == "question":
                        new_conversation_list.append({"role": "user", "content": dialog_dict["value"]})
                    elif dialog_dict["from"] == "answer":
                        new_conversation_list.append({"role": "assistant", "content": dialog_dict["value"]})
                    else:
                        logger.warning("[data_exchange][dialog_dict]{}，不为：question 或 answer".format(dialog_dict))
            else:
                logger.warning("[data_exchange][conversation_dict]{}，不存在：conversations".format(conversation_dict))

            conversation_dict["conversations"] = new_conversation_list
            if i == 0:
                logger.info("[data_exchange][conversation_dict] after: {}".format(conversation_dict))

            i += 1
            new_data_list.append(conversation_dict)
        except Exception as e:
            logger.exception(e)
            logger.info("[data_exchange][conversation_text] {}".format(conversation_text))

    logger.info("[data_exchange] finish, src_data_len: {}, new_data_len: {}".format(
        src_data_len, len(new_data_list)))

    return new_data_list

def data_list_to_dict(data_list):
    data_dict = dict()
    data_keys = ["id", "conversations"]
    for data_key in data_keys:
        data_dict[data_key] = list()
        for idx in range(len(data_list)):
            data_dict[data_key].append(data_list[idx][data_key])
    return data_dict


def data_preprocess(input_path, save_json=False, output_path=None, preset_data_path=None, preset_data_ratio=1.0):
    if os.path.exists(input_path):
        with open(input_path, mode="r", encoding="utf-8") as fr:
            src_lines = fr.readlines()

        if preset_data_path is not None:            # 如果preset_data_path不为空，则会对数据进行扩展
            if os.path.exists(preset_data_path) is True:
                with open(preset_data_path, mode="r", encoding="utf-8") as fr:
                    preset_lines = fr.readlines()
                    src_lines = sample_preset_data(src_lines=src_lines,
                                                   preset_lines=preset_lines,
                                                   preset_data_ratio=preset_data_ratio)
            else:
                logger.warning("[preprocess] preset_data_path: {}，文件不存在".format(preset_data_path))

        new_data_list = data_exchange(src_data_list=src_lines)
        if save_json and output_path:
            with open(output_path, mode="w", encoding="utf-8") as fw:
                json.dump(obj=new_data_list, fp=fw, ensure_ascii=False, indent=4)
            logger.info("[preprocess] save finish, output_path: {}".format(output_path))
        logger.info("[preprocess] exchange finish")
        return new_data_list
    else:
        logger.warning("[preprocess] input_path: {}，文件不存在".format(input_path))


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

    replace_dict = {"question": "user", "answer": "assistant"}
    preprocess(input_path=args.input_data, output_path=args.output_data, replace_dict=replace_dict)
