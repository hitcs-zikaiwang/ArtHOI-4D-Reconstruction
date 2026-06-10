from openai import OpenAI
import time
import base64
import os
import requests
import dashscope
from dashscope import MultiModalConversation

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
    return encoded_string

# FIXME change this impl to batch-inference or multi-threading for acceleration.

class Qwen:
    """
    Qwen/Qwen-VL-Max
    """
    api_url = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    api_key = "INVALID"  

    def __init__(self):
        self.client = OpenAI(
            api_key=self.api_key,
            base_url=self.api_url,
    )

    def request_with_image(self, prompt, image_path):
        image_base64 = encode_image(image_path)
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{image_base64}"
                        }
                    },
                    {
                        "type": "text",
                        "text": prompt
                    }
                ]
            }
        ]
        completion = self.client.chat.completions.create(
            # model="qwen3-vl-max-latest",
            model="qwen3-vl-flash",
            messages=messages
        )
        return completion.choices[0].message.content


class QwenPerspective:
    def __init__(self):
        self.api_key = os.getenv("QWEN_API_KEY", "INVALID")
        # FIXME change this to a avaliable model (newer and better) for dashscope no longer provides qwen-vl-max-latest
        # or use a third-party provider to keep this model choice.
        # self.model = "qwen-vl-max-latest"
        self.model = "qwen3-vl-flash"

    def request_with_images(self, prompt, image_paths, image_format="jpg"):
        image_contents = []
        for path in image_paths:
            base64_img = encode_image(path)
            image_contents.append({"image": f"data:image/{image_format};base64,{base64_img}"})
            print(path)
        image_contents.append({"text": prompt})

        messages = [
            {"role": "system", "content": [{"text": "You are a helpful assistant."}]},
            {"role": "user", "content": image_contents}
        ]
        response = MultiModalConversation.call(
            api_key=self.api_key,
            model=self.model,
            messages=messages,
            parameters={"vl_high_resolution_images": True}
        )

        # print("response:", response)
        # if not response or not getattr(response, "output", None):
        #     print("Warning: response or response.output is None")
        #     return ""
        
        return response.output.choices[0].message.content[0]["text"]


class QwenSingle:
    """
    DashScope SDK caller with local image base64 input and high-resolution parameters.
    """
    def __init__(self):
        self.api_key = os.getenv("QWEN_API_KEY", "INVALID")
        # FIXME change this to a avaliable model (newer and better) for dashscope no longer provides qwen-vl-max-latest
        # or use a third-party provider to keep this model choice.
        # self.model = "qwen-vl-max-latest"
        self.model = "qwen3.7-plus"
    
    def request_with_image(self, prompt, image_path, max_retries=5):
        base64_image = encode_image(image_path)
        messages = [
            {
                "role": "system",
                "content": [{"text": "You are a helpful assistant."}]
            },
            {
                "role": "user",
                "content": [
                    {"image": f"data:image/png;base64,{base64_image}"},
                    {"text": prompt}
                ]
            }
        ]
        retry = 0
        wait_time = 10  # Initial 10-second wait.
        while retry < max_retries:
            try:
                response = dashscope.MultiModalConversation.call(
                    api_key=self.api_key,
                    model=self.model,
                    messages=messages,
                    vl_high_resolution_images=True
                )
                # Robustness checks.
                if not response or not getattr(response, "output", None):
                    print("Warning: response or response.output is None")
                    raise Exception("Empty response")
                if not response.output.choices or not response.output.choices[0].message.content:
                    print("Warning: response.output.choices[0].message.content is empty")
                    raise Exception("Empty choices")
                return response.output.choices[0].message.content[0]["text"]
            except Exception as e:
                print(f"API call failed: {e}. Retrying in {wait_time} seconds (attempt {retry+1}).")
                time.sleep(wait_time)
                wait_time *= 2  # Exponential backoff.
                retry += 1
        print("API failed after multiple retries; skipping this image.")
        return ""
    
    
class QwenMulti:
    """
    DashScope SDK caller with multi-image base64 input and high-resolution parameters.
    """
    def __init__(self):
        self.api_key = os.getenv("QWEN_API_KEY", "INVALID")
        self.model = "qwen3-vl-flash"

    def request_with_images(self, prompt, image_paths, image_format="jpg"):
        image_contents = []
        for path in image_paths:
            base64_img = encode_image(path)
            image_contents.append({"image": f"data:image/{image_format};base64,{base64_img}"})
            # print(path)
        image_contents.append({"text": prompt})

        messages = [
            {"role": "system", "content": [{"text": "You are a helpful assistant."}]},
            {"role": "user", "content": image_contents}
        ]
        response = MultiModalConversation.call(
            api_key=self.api_key,
            model=self.model,
            messages=messages,
            parameters={"vl_high_resolution_images": True}
        )
        return response.output.choices[0].message.content[0]["text"]


"""
CVPR version: qwen-vl-max / Gemini-2.5-pro
it is highly recommended to use frontier models for better performance.
"""
