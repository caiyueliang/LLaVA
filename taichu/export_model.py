import argparse
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


def parse_argvs():
    parser = argparse.ArgumentParser(description='test post')
    parser.add_argument("--model_path", type=str, default="/mnt/publish-data/outputs/llava-v1.6-7b-lora/tmp/")
    parser.add_argument("--model_base", type=str, default="/mnt/publish-data/pretrain_models/llava/llava-v1.6-vicuna-7b/")
    parser.add_argument("--model_name", type=str, default="llava-v1.6-7b-lora")
    parser.add_argument("--export_dir", type=str, default="/mnt/publish-data/outputs/llava-v1.6-7b-lora/")

    args = parser.parse_args()
    logger.info('[args] {}'.format(args))

    return parser, args

if __name__ == "__main__":
    parser, args = parse_argvs()

    # model_path="/mnt/publish-data/outputs/llava-v1.6-7b-lora/tmp/"
    # model_base="/mnt/publish-data/pretrain_models/llava/llava-v1.6-vicuna-7b/"
    # model_name = get_model_name_from_path(model_path)
    # model_name = "llava-v1.6-7b-lora"
    export_model(
         model_path=args.model_path,
         model_base=args.model_base,
         model_name=args.model_name,
         export_dir=args.export_dir
    )