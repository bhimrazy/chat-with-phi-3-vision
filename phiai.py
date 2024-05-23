import json
from io import BytesIO

import requests

BASE_URL = "http://127.0.0.1:8000/predict"


class Phi3VisionAI:
    def __init__(self, base_url=BASE_URL):
        self.base_url = base_url

    def chat(self, messages, image):
        buffered = BytesIO()
        image = image.convert("RGB")
        image.save(buffered, format="JPEG")
        image_bytes = buffered.getvalue()

        # add image token to first messsage with role user, if image token is not already present
        if "<|image_1|>" not in messages[1]["content"]:
            messages[1]["content"] = f"<|image_1|>\n{messages[1]['content']}"

        data = {
            "images": [
                {"image_bytes": image_bytes.hex()},
            ],
            "messages": messages,
        }

        print(messages)
        try:
            response = requests.post(self.base_url, json=json.dumps(data))
            output = response.json()["output"]
            return output
        except Exception as e:
            print(e)
            return None


if __name__ == "__main__":
    client = Phi3VisionAI()
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is shown in this image?"},
    ]
    from PIL import Image

    image = Image.open("uploaded_images/IMG_2237.png")
    response = client.chat(messages, image)
    print(response)
