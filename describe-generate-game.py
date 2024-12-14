# L4: Describe and generate game
import os
import io
from IPython.display import Image, display, HTML
from PIL import Image
import base64 
from dotenv import load_dotenv, find_dotenv
from transformers import pipeline
import gradio as gr
import requests, json

# Settings
_ = load_dotenv(find_dotenv()) # read local .env file
hf_api_token = os.environ['HF_API_TOKEN']   # You can set environment variables by executing this command: export HF_API_TOKEN=hf_TKPpXAasdasqwe1233
server_port = 17778
image_captioning_endpoint_url = 'https://api-inference.huggingface.co/models/Salesforce/blip-image-captioning-base'
image_generation_endpoint_url = 'https://api-inference.huggingface.co/models/John6666/juggernaut-xl-rundiffusion-hyper-sdxl'

def get_completion(inputs, parameters=None, ENDPOINT_URL=""):
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

def get_completion_img(inputs, parameters=None, ENDPOINT_URL=""):
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
    return response.content

def image_to_base64_str(pil_image):
    byte_arr = io.BytesIO()
    pil_image.save(byte_arr, format='PNG')
    byte_arr = byte_arr.getvalue()
    return str(base64.b64encode(byte_arr).decode('utf-8'))

def captioner(image):
    base64_image = image_to_base64_str(image)
    result = get_completion(base64_image, None, image_captioning_endpoint_url)
    return result[0]['generated_text']

def generate(prompt):
    output = get_completion_img(prompt, None, image_generation_endpoint_url)
    result_image = Image.open(io.BytesIO(output))
    return result_image

def caption_and_generate(image):
    caption = captioner(image)
    image = generate(caption)
    return [caption, image]

with gr.Blocks() as demo:
    gr.Markdown("# Generate caption and alternative image from image 🖍️")
    image_upload = gr.Image(label="Your first image",type="pil")
    btn_all = gr.Button("Caption and generate")
    caption = gr.Textbox(label="Generated caption")
    image_output = gr.Image(label="Generated Image")

    btn_all.click(fn=caption_and_generate, inputs=[image_upload], outputs=[caption, image_output])

gr.close_all()

demo.launch(share=True, server_port=server_port)