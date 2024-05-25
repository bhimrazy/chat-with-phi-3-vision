import base64
import requests
from PIL import Image
from io import BytesIO
import re
from litserve.specs.openai import ChatMessage, ChatCompletionRequest


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
            return Image.open(requests.get(source, stream=True).raw)
        elif re.match(r"^data:image/.+;base64,", source):
            # It's a base64 image URL
            base64_image = source.split(",")[1]
            image_data = base64.b64decode(base64_image)
            return Image.open(BytesIO(image_data))
        else:
            return Image.open(source)
    except Exception as e:
        print(f"An error occurred: {e}")
        return None


def encode_image(image_source):
    """
    Encode an image to a base64 data URL.

    Parameters:
    image_source (str or Image): The image source. Can be a real image URL, an Image instance, or a base64 URL string.

    Returns:
    str or None: The base64-encoded data URL of the image if successful, otherwise None.
    """
    if isinstance(image_source, str):
        image = read_image(image_source)
        if image is None:
            return None
    elif isinstance(image_source, Image.Image):
        image = image_source
    else:
        return None

    buffered = BytesIO()
    # Use image format or default to "JPEG"
    format = image.format if image.format else "JPEG"
    image.save(buffered, format=format)
    mime_type = f"image/{format.lower()}"
    encoded_image = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return f"data:{mime_type};base64,{encoded_image}"


def parse_messages(request: ChatCompletionRequest):
    """
    Parse messages from a ChatCompletionRequest object, extracting text content and images.

    Parameters:
    request (ChatCompletionRequest): The request object containing messages.

    Returns:
    Tuple[List[ChatMessage], List[Image]]: A tuple containing a list of ChatMessage objects with updated content
    and a list of PIL Image objects representing the images extracted from the messages.
    """
    messages = []
    images = []

    for message in request.messages:
        if isinstance(message.content, list):
            text_content = ""
            image_token_content = ""
            image_count = 1

            for content_item in message.content:

                if content_item.type == "text":
                    text_content += content_item.text
                elif content_item.type == "image_url":
                    image_url = content_item.image_url
                    if image_url:
                        image_token_content += f"<|image_{image_count}|>\n"
                        image = read_image(image_url)
                        if image:
                            images.append(image)
                        image_count += 1

            text_content = image_token_content + text_content
            messages.append(ChatMessage(role=message.role, content=text_content))
        else:
            messages.append(message)

    return messages, images
