import argparse
import requests
from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path
from llava.eval.run_llava import eval_model
from loguru import logger


def parse_argvs():
    parser = argparse.ArgumentParser(description='exchange_torch2trt')
    parser.add_argument("--url", type=str, default="http://localhost:10000/worker_generate_stream")
    parser.add_argument("--model_path", type=str, default="/mnt/publish-data/pretrain_models/llava/llava-v1.6-vicuna-7b/")
    parser.add_argument("--prompt", type=str, default="图片<image>中讲了什么内容？")
    parser.add_argument("--image_file", type=str, default="./img.png")
    parser.add_argument("--stop", type=str, default="")

    args = parser.parse_args()
    logger.info('[args] {}'.format(args))

    return parser, args


if __name__ == "__main__":
    parser, args = parse_argvs()

    model_path = args.model_path

    # tokenizer, model, image_processor, context_len = load_pretrained_model(
    #     model_path=model_path,
    #     model_base=None,
    #     model_name=get_model_name_from_path(model_path)
    # )
    #
    # # model_path = "liuhaotian/llava-v1.5-7b"
    # # image_file = "https://llava-vl.github.io/static/images/view.jpg"
    #
    # args = type('Args', (), {
    #     "model_path": model_path,
    #     "model_base": None,
    #     "model_name": get_model_name_from_path(model_path),
    #     "query": args.prompt,
    #     "conv_mode": None,
    #     "image_file": args.image_file,
    #     "sep": ",",
    #     "temperature": 0,
    #     "top_p": None,
    #     "num_beams": 1,
    #     "max_new_tokens": 512
    # })()
    #
    # eval_model(args)
    from image_to_base64 import image_to_base64

    base64_string = image_to_base64(args.image_file)

    params = {
        "model": "llava-v1.6-vicuna-7b",
        "prompt": args.prompt,
        "images": [base64_string],
        "temperature": 0,
        "top_p": 1.0,
        "max_new_tokens": 256,
        "stop": args.stop
    }
    response = requests.post(url=args.url,
                             json=params, stream=True, timeout=5)
    for chunk in response.iter_lines(decode_unicode=False, delimiter=b"\0"):
        if chunk:
            print(chunk)
            # yield chunk + b"\0"
