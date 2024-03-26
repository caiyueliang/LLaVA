from loguru import logger
from llava.model.builder import load_pretrained_model

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