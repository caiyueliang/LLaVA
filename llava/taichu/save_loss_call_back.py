import os
import json
import numpy as np
from loguru import logger
from transformers.trainer_callback import TrainerCallback

  
class SaveLossCallback(TrainerCallback):
    def __init__(self, loss_file_path=None):
        self.loss_list = []
        self.loss_metrics = {'train': []}
        if loss_file_path:
            os.makedirs(name=loss_file_path, exist_ok=True)
        self.loss_file = os.path.join(loss_file_path, "loss.json")
        logger.info(f"[SaveLossCallback] loss_file: {self.loss_file}")
        # print(f"[SaveLossCallback] loss_file: {self.loss_file}", flush=True)

    def on_epoch_end(self, args, state, control, **kwargs):
        # 自定义在每个epoch结束时执行的操作
        logger.info("-" * 80)
        # print("-" * 80, flush=True)

    def on_log(self, args, state, control, logs=None, **kwargs):
        # 检查logs中是否有loss和step信息，并打印它们
        if logs is not None and args.local_rank == 0:
            try:
                if "loss" in logs:
                    # logger.info(f"[on_log][logs] {logs}, [state] {state}")
                    self.loss_list.append(float(logs['loss']))
                    step_per_epoch = int(state.max_steps / state.num_train_epochs)
                    cur_epoch = int(logs['epoch'] - 0.001) if logs['epoch'] - 0.001 >= 0 else 0
                    cur_epoch += 1
                    metrics = {
                        'epoch': cur_epoch,
                        'step': int(state.global_step - (cur_epoch - 1) * step_per_epoch),
                        'global_step': state.global_step,
                        'loss': logs['loss'],
                        'lr': logs['learning_rate'],
                        'mean_loss': np.mean(self.loss_list)
                    }
                    logger.info(f"[metrics] {metrics}")

                    self.loss_metrics['train'].append(metrics)

                    with open(self.loss_file, 'w', encoding="utf-8") as file:
                        json.dump(self.loss_metrics, file, indent=4, ensure_ascii=False)
            except Exception as e:
                logger.info(f"[on_log][logs] {logs}, [state] {state}")
                logger.exception(e)