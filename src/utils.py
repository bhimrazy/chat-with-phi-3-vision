import base64
import concurrent.futures
import os
import re
import json
from io import BytesIO
from typing import List

import requests
from decord import VideoReader, cpu
from litserve.specs.openai import ChatCompletionRequest, ResponseFormat
from PIL import Image

from src.config import IMAGE_EXTENSIONS, MAX_NUM_FRAMES, VIDEO_EXTENSIONS


def get_file_extension(filename):
    return os.path.splitext(filename)[1].lower()


def is_image(filename):
    return get_file_extension(filename) in IMAGE_EXTENSIONS


def all_images(files):
    return all(is_image(file.name) for file in files)


def is_video(filename):
    return get_file_extension(filename) in VIDEO_EXTENSIONS


def all_videos(files):
    return all(is_video(file.name) for file in files)


def read_image(source):
    """
    Read an image from a real image URL or a base64-encoded URL.

    Parameters:
    source (str): The image source. Can be a real image URL or a base64 URL string.

    Returns:
    Image or None: The Image object if the source is valid, otherwise None.
    """
    try:
        if re.match(r"^https?://", source):
            # It's a real image URL
            return Image.open(requests.get(source, stream=True).raw).convert("RGB")
        elif re.match(r"^data:image/.+;base64,", source):
            # It's a base64 image URL
            base64_image = source.split(",")[1]
            image_data = base64.b64decode(base64_image)
            return Image.open(BytesIO(image_data)).convert("RGB")
        else:
            return Image.open(source).convert("RGB")
    except Exception as e:
        print(f"An error occurred: {e}")
        return None


def encode_image(image_source):
    """
    Encode an image to a base64 data URL based object.

    Parameters:
    image_source (str or Image): The image source. Can be a real image URL, an Image instance, or a base64 URL string.

    Returns:
    str or None: The base64-encoded data URL of the image if successful, otherwise None.
    """
    try:
        if isinstance(image_source, str):
            image = read_image(image_source)
            if image is None:
                return None
        elif isinstance(image_source, Image.Image):
            image = image_source
        else:
            image = Image.open(image_source).convert("RGB")

        # resize to max_size
        max_size = 720
        if max(image.size) > max_size:
            w, h = image.size
            if w > h:
                new_w = max_size
                new_h = int(h * max_size / w)
            else:
                new_h = max_size
                new_w = int(w * max_size / h)
            image = image.resize((new_w, new_h), resample=Image.BICUBIC)

        buffered = BytesIO()
        # Use image format or default to "JPEG"
        image_format = image.format if image.format else "JPEG"
        image.save(buffered, format=image_format)
        mime_type = f"image/{image_format.lower()}"
        encoded_image = base64.b64encode(buffered.getvalue()).decode("utf-8")
        url = f"data:{mime_type};base64,{encoded_image}"

        image_object = {
            "type": "image_url",
            "image_url": {
                "url": url,
            },
        }
        return image_object
    except Exception as e:
        print(f"Error encoding image: {e}")
        return None


def prepare_content_with_images(content: str, images: List[object]):
    """Prepare content with images."""
    content = [
        {
            "type": "text",
            "text": content,
        },
        *images,
    ]
    return content


def encode_video(video):
    """Encode a video to a list of base64 data URLs."""

    def uniform_sample(l, n):  # noqa: E741
        gap = len(l) / n
        idxs = [int(i * gap + gap / 2) for i in range(n)]
        return [l[i] for i in idxs]

    if hasattr(video, "path"):
        vr = VideoReader(video.path, ctx=cpu(0))
    else:
        vr = VideoReader(video, ctx=cpu(0))
    sample_fps = round(vr.get_avg_fps() / 1)  # FPS
    frame_idx = [i for i in range(0, len(vr), sample_fps)]
    if len(frame_idx) > MAX_NUM_FRAMES:
        frame_idx = uniform_sample(frame_idx, MAX_NUM_FRAMES)
    video = vr.get_batch(frame_idx).asnumpy()

    def process_frame(frame):
        return encode_image(Image.fromarray(frame.astype("uint8")))

    # Use ThreadPoolExecutor to parallelize the encoding process
    with concurrent.futures.ThreadPoolExecutor() as executor:
        video = list(executor.map(process_frame, video))

    return video


def prep_prompt(system_prompt: str, response_format: ResponseFormat):
    """
    Prepare system prompt with response format.

    response format prompt adapted from : https://github.com/SylphAI-Inc/AdalFlow
    """
    response_format_str = ""
    response_format = response_format.model_dump(exclude_none=True, by_alias=True)
    schema = response_format.get("json_schema", None)

    if schema:
        response_format_str = (
            "<RESPONSE_FORMAT>\n"
            "Your output should be formatted as a standard JSON instance with the following schema:\n"
            "```\n"
            f"{json.dumps(schema, indent=4)}\n"
            "```\n"
            "- Make sure to always enclose the JSON output in triple backticks (```). Please do not add anything other than valid JSON output!\n"
            "- Use double quotes for the keys and string values.\n"
            '- DO NOT mistake the "properties" and "type" in the schema as the actual fields in the JSON output.\n'
            "- DO NOT include any additional text, comments, or annotations outside of the JSON object.\n"
            "- Follow the structure and field names as specified in the schema exactly.\n"
            "- Follow the JSON formatting conventions.\n"
            "- DO NOT include schema definitions in the JSON output.\n"
            "- Ensure that the JSON output strictly conforms to the schema provided without deviation.\n"
            "- Do validate your JSON output for syntax correctness and adherence to the schema before submission.\n"
            "- Strictly adhere to the schema provided above.\n"
            "- Return the markdown JSON object as the output without any additional text or comments.\n"
            "</RESPONSE_FORMAT>"
        )
    else:
        response_format_str = (
            "<RESPONSE_FORMAT>\n"
            "Your output should be formatted as a standard JSON instance.\n"
            "- Make sure to always enclose the JSON output in triple backticks (```). Please do not add anything other than valid JSON output!\n"
            "- Use double quotes for the keys and string values.\n"
            '- DO NOT mistake the "properties" and "type" in the schema as the actual fields in the JSON output.\n'
            "- Follow the JSON formatting conventions.\n"
            "</RESPONSE_FORMAT>"
        )
    return f"{system_prompt}\n\n{response_format_str}"


def parse_messages(request: ChatCompletionRequest):
    """
    Parse messages from a ChatCompletionRequest object.
    """
    messages = []
    images = []
    img_count = 1
    response_format = request.response_format
    for message in request.messages:
        content = message.content
        if message.role == "system" and response_format:
            content = prep_prompt(content, response_format)

        if isinstance(content, list):
            prompt = ""
            placeholder = ""
            for content_item in message.content:
                if content_item.type == "text":
                    prompt = content_item.text
                elif content_item.type == "image_url":
                    url = content_item.image_url.url
                    images.append(url)
                    placeholder += f"<|image_{img_count}|>\n"
                    img_count += 1
            content = placeholder + prompt

        messages.append({"role": message.role, "content": content})

    def process_image(image_url):
        image = read_image(image_url)
        # resize if height is greater than 720
        if image.height > 720:
            image = image.resize((int(image.width * 720 / image.height), 720))
        return image

    with concurrent.futures.ThreadPoolExecutor() as executor:
        images = list(executor.map(process_image, images))

    return messages, images
