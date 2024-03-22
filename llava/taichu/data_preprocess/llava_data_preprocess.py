import json
import os
import random
import pandas as pd
from loguru import logger

from data_preprocess import DataPreprocess

class LlavaDataPreprocess(DataPreprocess):
    def __init__(self) -> None:
        super().__init__()

    def data_exchange(self, src_data_list: list):
        pass