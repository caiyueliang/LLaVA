import json
import os
import random
import pandas as pd
from loguru import logger
import argparse

from .data_preprocess import DataPreprocess

class LLMDataPreprocess(DataPreprocess):
    def __init__(self) -> None:
        super().__init__()
        self.input_file_is_json = False

    def data_exchange(self, src_data_list: list):
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
    

def parse_argvs():
    parser = argparse.ArgumentParser(description='data preprocess')
    parser.add_argument("--input_data", type=str, default="/mnt/publish-data/train_data/llm_data_2/weather/result.json")
    parser.add_argument("--output_data", type=str, default="./result_llm_temp.json")
    # parser.add_argument("--replace_dict", type={}, default={"question": "user", "answer": "assistant"})

    args = parser.parse_args()
    logger.info('[args] {}'.format(args))

    return parser, args


if __name__ == "__main__":
    parser, args = parse_argvs()

    # replace_dict = {"question": "user", "answer": "assistant"}
    LLMDataPreprocess().data_preprocess(input_path=args.input_data, output_path=args.output_data)