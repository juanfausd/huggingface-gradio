# L2: Image captioning app

import os
import io
import IPython.display
from PIL import Image
import base64 
from dotenv import load_dotenv, find_dotenv
import requests, json

# Settings
_ = load_dotenv(find_dotenv()) # read local .env file
hf_api_token = os.environ['HF_API_TOKEN']   # You can set environment variables by executing this command: export HF_API_TOKEN=hf_TKPpXAasdasqwe1233
server_port = 12779
image_captioning_endpoint_url = 'https://api-inference.huggingface.co/models/Salesforce/blip-image-captioning-base'

# Image-to-text endpoint
def get_completion(inputs, parameters=None, ENDPOINT_URL=image_captioning_endpoint_url:
    headers = {
      "Authorization": f"Bearer {hf_api_token}",
      "Content-Type": "application/json"
    }
    data = { "inputs": inputs }
    if parameters is not None:
        data.update({"parameters": parameters})
    response = requests.request("POST",
                                ENDPOINT_URL,
                                headers=headers,
                                data=json.dumps(data))
    return json.loads(response.content.decode("utf-8"))

import gradio as gr 

def image_to_base64_str(pil_image):
    byte_arr = io.BytesIO()
    pil_image.save(byte_arr, format='PNG')
    byte_arr = byte_arr.getvalue()
    return str(base64.b64encode(byte_arr).decode('utf-8'))

def captioner(image):
    base64_image = image_to_base64_str(image)
    result = get_completion(base64_image)
    return result[0]['generated_text']

gr.close_all()
demo = gr.Interface(fn=captioner,
                    inputs=[gr.Image(label="Upload image", type="pil")],
                    outputs=[gr.Textbox(label="Caption")],
                    title="Image Captioning with BLIP",
                    description="Caption any image using the BLIP model",
                    allow_flagging="never",
                    examples=["christmas_dog.jpeg", "bird_flight.jpeg", "cow.jpeg"])

demo.launch(share=True, server_port=server_port)