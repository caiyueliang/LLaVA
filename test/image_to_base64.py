from PIL import Image
import base64
import argparse
from io import BytesIO
from loguru import logger


def parse_argvs():
    parser = argparse.ArgumentParser(description='exchange_torch2trt')
    parser.add_argument("--model_path", type=str, default="/mnt/publish-data/pretrain_models/llava/llava-v1.6-vicuna-7b/")
    parser.add_argument("--prompt", type=str, default="图片中讲了什么内容？")
    parser.add_argument("--image_file", type=str, default="./img.png")

    args = parser.parse_args()
    logger.info('[args] {}'.format(args))

    return parser, args


def image_to_base64(image_path):
    # 打开图片文件
    with Image.open(image_path) as image:
        # 将图片转换为字节
        buffered = BytesIO()
        # image.save(buffered, format="JPEG")  # 可以根据需要调整格式
        image.save(buffered, format="PNG")  # 可以根据需要调整格式

        # 获取字节数据
        img_data = buffered.getvalue()
        # 将字节数据编码为 base64 字符串
        img_base64 = base64.b64encode(img_data)
        # Python 3.x 返回的是 bytes，需要解码为字符串
        return img_base64.decode('utf-8')


if __name__ == "__main__":
    parser, args = parse_argvs()

    # 使用示例
    image_path = args.image_file
    base64_string = image_to_base64(image_path)
    print()
    print(base64_string)
