import copy

from openai import OpenAI

from utils import encode_image


class Phi3VisionAI:
    def __init__(self):
        self.client = OpenAI(
            base_url="http://127.0.0.1:8000/v1",
            api_key="lit",
        )

    def chat(self, messages, image):
        input_messages = copy.deepcopy(messages)
        input_messages[1]["content"] = [
            {"type": "text", "text": messages[1]["content"]},
            {
                "type": "image_url",
                "image_url": encode_image(image),
            },
        ]
        try:
            stream_response = self.client.chat.completions.create(
                model="phi-3-vision", messages=input_messages, stream=True
            )
            # return response.choices[0].message.content
            return stream_response
        except Exception as e:
            print(e)
            return None


if __name__ == "__main__":
    client = Phi3VisionAI()
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is shown in this image?"},
        {
            "role": "assistant",
            "content": "The image shows a wooden boardwalk winding through a lush, green marshland. The sky is partly cloudy with ample sunlight, suggesting it might be a pleasant day. There are trees and shrubs scattered throughout the marsh, and the overall scene is serene and natural.",
        },
        {"role": "user", "content": "How is the weather ?"},
    ]
    from PIL import Image

    image = Image.open("dogs-playing-in-grassy-field.jpg")
    response = client.chat(messages, image)
    print(messages, response)
