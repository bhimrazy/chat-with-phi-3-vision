import os
from threading import Thread

import litserve as ls
from litserve.specs.openai import ChatCompletionRequest
from transformers import AutoModelForCausalLM, AutoProcessor, TextIteratorStreamer

from utils import parse_messages

os.environ["HF_DATASETS_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"


class Phi3VisionLitAPI(ls.LitAPI):
    def setup(self, device):
        model_id = "microsoft/Phi-3-vision-128k-instruct"
        self.model = None
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id, device_map=device, trust_remote_code=True, torch_dtype="auto"
        ).eval()

        self.processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)

    def decode_request(self, request: ChatCompletionRequest):
        # parse messages
        messages, images = parse_messages(request)

        prompt = self.processor.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        model_inputs = self.processor(prompt, images, return_tensors="pt").to(
            self.model.device
        )
        return model_inputs

    def predict(self, model_inputs):
        # without streaming
        # input_len = model_inputs["input_ids"].shape[-1]
        # with torch.inference_mode():
        #     generation_args = {
        #         "max_new_tokens": 1000,
        #     }
        #     generate_ids = self.model.generate(
        #         **model_inputs,
        #         eos_token_id=self.processor.tokenizer.eos_token_id,
        #         **generation_args,
        #     )

        #     generate_ids = generate_ids[:, input_len:]
        #     decoded = self.processor.batch_decode(
        #         generate_ids,
        #         skip_special_tokens=True,
        #         clean_up_tokenization_spaces=False,
        #     )[0]
        #     yield decoded

        # with streaming
        # streaming
        streamer = TextIteratorStreamer(
            self.processor.tokenizer,
            skip_prompt=True,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )

        # Run the generation in a separate thread, so that we can fetch the generated text in a non-blocking way.
        generation_kwargs = dict(
            model_inputs,
            streamer=streamer,
            max_new_tokens=4096,
            eos_token_id=self.processor.tokenizer.eos_token_id,
        )
        thread = Thread(target=self.model.generate, kwargs=generation_kwargs)
        thread.start()
        for text in streamer:
            yield text


if __name__ == "__main__":
    api = Phi3VisionLitAPI()
    server = ls.LitServer(api, spec=ls.OpenAISpec())
    server.run(port=8000)
