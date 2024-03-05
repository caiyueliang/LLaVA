import argparse
import requests
from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path
from llava.eval.run_llava import eval_model
from loguru import logger
import time

headers = {"User-Agent": "LLaVA Client"}


def parse_argvs():
    parser = argparse.ArgumentParser(description='exchange_torch2trt')
    parser.add_argument("--url", type=str, default="http://localhost:40000/worker_generate_stream")
    parser.add_argument("--model_path", type=str, default="/mnt/publish-data/pretrain_models/llava/llava-v1.6-vicuna-7b/")
    parser.add_argument("--prompt", type=str, default="图片<image>中讲了什么内容？")
    parser.add_argument("--image_file", type=str, default="./img.png")
    parser.add_argument("--stop", type=str, default="")

    args = parser.parse_args()
    logger.info('[args] {}'.format(args))

    return parser, args


if __name__ == "__main__":
    parser, args = parse_argvs()

    # wmodel_path = args.model_path

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

    # params = {
    #     "model": "llava-v1.6-vicuna-7b",
    #     "prompt": args.prompt,
    #     "images": [base64_string],
    #     "temperature": 0,
    #     "top_p": 1.0,
    #     "max_new_tokens": 256,
    #     "stop": args.stop
    # }
    # response = requests.post(url=args.url,
    #                          json=params, stream=True, timeout=5)
    
    # Make requests
    pload = {
        "model": "llava-v1.6-vicuna-7b",
        "prompt": "A chat between a curious human and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the human's questions. USER: <image>\n图片里有几辆车 ASSISTANT:",
        "temperature": 0.2,
        "top_p": 0.7,
        "max_new_tokens": 1024,
        "stop": "</s>",
        "images": f'List of {len(state.get_images())} images: {all_image_hash}',
    }
    logger.info(f"==== request ====\n{pload}")

    pload['images'] = state.get_images()
    
    # state.messages[-1][-1] = "▌"

    logger.warning(f"[headers] {headers}")
    logger.warning(f"  [pload] {pload}")
    # logger.warning(f"  [state] {state}")

    response = requests.post(url=args.url,
            headers=headers, json=pload, stream=True, timeout=10)
    for chunk in response.iter_lines(decode_unicode=False, delimiter=b"\0"):
            if chunk:
                data = json.loads(chunk.decode())
                if data["error_code"] == 0:
                    output = data["text"][len(prompt):].strip()
                    # state.messages[-1][-1] = output + "▌"
                    print(output)
                else:
                    output = data["text"] + f" (error_code: {data['error_code']})"
                    # state.messages[-1][-1] = output
                    print(output)
                time.sleep(0.03)