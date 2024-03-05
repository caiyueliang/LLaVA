import time
import json
import argparse
import requests
from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path
from llava.eval.run_llava import eval_model
from loguru import logger

headers = {"User-Agent": "LLaVA Client"}


def parse_argvs():
    parser = argparse.ArgumentParser(description='exchange_torch2trt')
    parser.add_argument("--url", type=str, default="http://localhost:40000/worker_generate_stream")
    parser.add_argument("--model_path", type=str, default="/mnt/publish-data/pretrain_models/llava/llava-v1.6-vicuna-7b/")
    parser.add_argument("--question", type=str, default="图片中讲了什么内容？")
    parser.add_argument("--image_file", type=str, default="./img.png")
    parser.add_argument("--stop", type=str, default="")

    args = parser.parse_args()
    logger.info('[args] {}'.format(args))

    return parser, args


if __name__ == "__main__":
    parser, args = parse_argvs()

    from image_to_base64 import image_to_base64

    base64_string = image_to_base64(args.image_file)

    
    # Make requests
    prompt = "A chat between a curious human and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the human's questions. USER: <image>\n{question} ASSISTANT:".format(question=args.question)
    pload = {
        "model": "llava-v1.6-vicuna-7b",
        "prompt": prompt,
        "temperature": 0.2,
        "top_p": 0.7,
        "max_new_tokens": 1024,
        "stop": "</s>",
        "images": [base64_string],
    }
    logger.info(f"==== request ====\n{pload}")

    # pload['images'] = state.get_images()

    logger.warning(f"[headers] {headers}")
    # logger.warning(f"  [pload] {pload}")

    response = requests.post(url=args.url,
            headers=headers, json=pload, stream=True, timeout=10)
    for chunk in response.iter_lines(decode_unicode=False, delimiter=b"\0"):
            if chunk:
                data = json.loads(chunk.decode())
                if data["error_code"] == 0:
                    output = data["text"][len(prompt):].strip()
                    print(output)
                else:
                    output = data["text"] + f" (error_code: {data['error_code']})"
                    print(output)
                time.sleep(0.03)