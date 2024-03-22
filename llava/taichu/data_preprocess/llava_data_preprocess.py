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
        self.input_file_is_json = True

    def data_exchange(self, src_data_list: list) -> list:
        new_data_list = list()
        src_data_len = len(src_data_list)
        logger.info("[data_exchange] src_data_len: {}".format(src_data_len))

        i = 0
        for conversation_dict in src_data_list:
            try:
                if i == 0:
                    logger.info("[data_exchange][conversation_dict] before: {}".format(conversation_dict))
                new_conversation_list = list()

                if "conversations" in conversation_dict.keys():
                    for dialog_dict in conversation_dict["conversations"]:
                        if "question" not in dialog_dict.keys() or "answer" not in dialog_dict.keys():
                            logger.warning("[data_exchange][dialog_dict] {}，中不存在： question 或 answer".format(dialog_dict))

                        if "image" in conversation_dict.keys() and conversation_dict["image"] is not None:
                            if "<image>" not in dialog_dict["question"]:
                                dialog_dict["question"] = "<image>\n" + dialog_dict["question"]
                        if "question" in dialog_dict.keys():
                            new_conversation_list.append({"from": "human", "value": dialog_dict["question"]})
                        if "answer" in dialog_dict.keys():
                            new_conversation_list.append({"from": "gpt", "value": dialog_dict["answer"]})
                else:
                    logger.warning("[data_exchange][conversation_dict]{}，不存在：conversations".format(conversation_dict))

                conversation_dict["conversations"] = new_conversation_list
                if i == 0:
                    logger.info("[data_exchange][conversation_dict] after: {}".format(conversation_dict))

                i += 1
                new_data_list.append(conversation_dict)
            except Exception as e:
                logger.exception(e)
                logger.info("[data_exchange][conversation_dict] {}".format(conversation_dict))

        logger.info("[data_exchange] finish, src_data_len: {}, new_data_len: {}".format(
            src_data_len, len(new_data_list)))

        return new_data_list

def parse_argvs():
    parser = argparse.ArgumentParser(description='data preprocess')
    parser.add_argument("--input_data", type=str, default="/mnt/publish-data/train_data/llava_data/01/result.json")
    parser.add_argument("--output_data", type=str, default="./result_llava_temp.json")

    args = parser.parse_args()
    logger.info('[args] {}'.format(args))

    return parser, args


if __name__ == "__main__":
    parser, args = parse_argvs()

    LlavaDataPreprocess().data_preprocess(input_path=args.input_data, output_path=args.output_data)