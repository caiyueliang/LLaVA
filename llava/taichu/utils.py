import torch
from loguru import logger
import transformers
from peft import AutoPeftModelForCausalLM
from transformers import AutoTokenizer, AutoModelForCausalLM
from llava import LlavaLlamaForCausalLM
from llava.model.language_model.llava_llama import LlavaConfig
from llava.mm_utils import get_model_name_from_path
from llava.model.builder import load_pretrained_model


def merge_save_model(model_base, path_to_adapter, new_model_directory, max_shard_size=4):
    # if trainer.args.should_save and trainer.args.local_rank == 0:
        logger.info("[merge_save_model] start")
        # import ipdb
        # ipdb.set_trace()

        # from llava.model.language_model.llava_llama import LlavaConfig
        # cfg_pretrained = LlavaConfig.from_pretrained(path_to_adapter)

        tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=False)
        cfg_pretrained = LlavaConfig.from_pretrained(path_to_adapter)
        # model = LlavaLlamaForCausalLM.from_pretrained(model_base, low_cpu_mem_usage=True, config=cfg_pretrained)
        model = LlavaLlamaForCausalLM.from_pretrained(model_base, config=cfg_pretrained)

        if getattr(model, "quantization_method", None):
            model = model.to("cpu")
        elif hasattr(model.config, "torch_dtype"):
            model = model.to(getattr(model.config, "torch_dtype")).to("cpu")
        else:
            model = model.to(torch.float16).to("cpu")
            setattr(model.config, "torch_dtype", torch.float16)

        # model = AutoPeftModelForCausalLM.from_pretrained(
        #     path_to_adapter,  # path to the output directory
        #     config=cfg_pretrained,
        #     device_map=device_map,
        #     trust_remote_code=True
        # ).eval()

        # model = AutoModelForCausalLM.from_pretrained(
        #     path_to_adapter,  # path to the output directory
        #     device_map=device_map,
        #     trust_remote_code=True
        # ).eval()

        # merged_model = model.merge_and_unload()
        # max_shard_size and safe serialization are not necessary.
        # They respectively work for sharding checkpoint and save the model to safetensors
        model.save_pretrained(new_model_directory, 
                              max_shard_size="{}GB".format(max_shard_size), 
                              safe_serialization=True,
                              trust_remote_code=True)
        tokenizer.save_pretrained(new_model_directory)

        logger.info("[merge_save_model] end")

def export_model(model_path, model_base, export_dir, model_name="llava-v1.6-7b-lora", max_shard_size=4):
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        model_path=model_path,
        model_base=model_base,
        model_name=model_name
    )
    
    model.save_pretrained(export_dir, 
                          max_shard_size="{}GB".format(max_shard_size), 
                          safe_serialization=True,
                          trust_remote_code=True)
    tokenizer.save_pretrained(export_dir)


if __name__ == "__main__":
    # merge_save_model(
    #      model_base="/mnt/publish-data/pretrain_models/llava/llava-v1.6-vicuna-7b/",
    #      path_to_adapter="/mnt/publish-data/outputs/llava-v1.6-7b-lora/tmp/", 
    #      new_model_directory="/mnt/publish-data/outputs/llava-v1.6-7b-lora/")

    model_path="/mnt/publish-data/outputs/llava-v1.6-7b-lora/tmp/"
    model_base="/mnt/publish-data/pretrain_models/llava/llava-v1.6-vicuna-7b/"
    # model_name = get_model_name_from_path(model_path)
    model_name = "llava-v1.6-7b-lora"
    export_model(
         model_path=model_path,
         model_base=model_base,
         model_name=model_name,
         export_dir="/mnt/publish-data/outputs/llava-v1.6-7b-lora/"
    )