import argparse
from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path
from llava.eval.run_llava import eval_model
from loguru import logger


def parse_argvs():
    parser = argparse.ArgumentParser(description='exchange_torch2trt')
    parser.add_argument("--model_path", type=str, default="/mnt/publish-data/pretrain_models/llava/llava-v1.6-vicuna-7b/")
    parser.add_argument("--prompt", type=str, default="图片中讲了什么内容？")
    parser.add_argument("--image_file", type=str, default="https://llava-vl.github.io/static/images/view.jpg")

    args = parser.parse_args()
    logger.info('[args] {}'.format(args))

    return parser, args


if __name__ == "__main__":
    parser, args = parse_argvs()

    model_path = args.model_path

    tokenizer, model, image_processor, context_len = load_pretrained_model(
        model_path=model_path,
        model_base=None,
        model_name=get_model_name_from_path(model_path)
    )

    # model_path = "liuhaotian/llava-v1.5-7b"
    # image_file = "https://llava-vl.github.io/static/images/view.jpg"

    args = type('Args', (), {
        "model_path": model_path,
        "model_base": None,
        "model_name": get_model_name_from_path(model_path),
        "query": args.prompt,
        "conv_mode": None,
        "image_file": args.image_file,
        "sep": ",",
        "temperature": 0,
        "top_p": None,
        "num_beams": 1,
        "max_new_tokens": 512
    })()

    eval_model(args)
