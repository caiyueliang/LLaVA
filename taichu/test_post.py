import time
import json
import argparse
import requests
from loguru import logger
import base64

headers = {"User-Agent": "LLaVA Client"}

def image_to_base64(image_file):
    with open(image_file, 'rb') as image_file:
        image_data = image_file.read()
    # 将图像数据编码为Base64字符串
    base64_string = base64.b64encode(image_data).decode('utf-8')
    return base64_string

def parse_argvs():
    parser = argparse.ArgumentParser(description='test post')
    parser.add_argument("--url", type=str, default="http://localhost:40000/worker_generate_stream")
    parser.add_argument("--model_path", type=str, default="/mnt/publish-data/pretrain_models/llava/llava-v1.6-vicuna-7b/")
    parser.add_argument("--question", type=str, default="图片中讲了什么内容？")
    parser.add_argument("--image_file", type=str, default="./img.png")
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=0.7)
    parser.add_argument("--max_new_tokens", type=str, default=1024)
    parser.add_argument("--stop", type=str, default="</s>")

    args = parser.parse_args()
    logger.info('[args] {}'.format(args))

    return parser, args


if __name__ == "__main__":
    parser, args = parse_argvs()

    base64_string = image_to_base64(image_file=args.image_file)
    images = [base64_string]

    # Make requests
    prompt = "A chat between a curious human and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the human's questions. USER: <image>\n{question} ASSISTANT:".format(question=args.question)
    pload = {
        "model": "llava-v1.6-vicuna-7b",
        "prompt": prompt,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "max_new_tokens": args.max_new_tokens,
        "stop": args.stop,
        "images": [],
    }
    logger.info(f"[headers] {headers}")
    logger.info(f"  [pload] {pload}")

    logger.info(f"==== request ====\n")

    pload['images'] = images

    response = requests.post(url=args.url,
            headers=headers, json=pload, stream=True)
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
