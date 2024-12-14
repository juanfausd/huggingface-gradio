import os
import io
import IPython.display
from PIL import Image
import base64 
from dotenv import load_dotenv, find_dotenv
import requests, json
import gradio as gr 

# Settings
_ = load_dotenv(find_dotenv()) # read local .env file
hf_api_token = os.environ['HF_API_TOKEN']   # You can set environment variables by executing this command: export HF_API_TOKEN=hf_TKPpXAasdasqwe1233
server_port = 12781
image_generation_endpoint_url = 'https://api-inference.huggingface.co/models/stabilityai/stable-diffusion-xl-base-1.0'

# Text-to-image endpoint
def get_completion(inputs, parameters=None, ENDPOINT_URL=image_generation_endpoint_url):
    headers = {
      "Authorization": f"Bearer {hf_api_token}",
      "Content-Type": "application/json"
    }   
    data = { "inputs": inputs }
    
    if parameters is not None:
        data.update({"parameters": parameters})

    response = requests.post(image_generation_endpoint_url, headers=headers, json=json.dumps(data))
    
    return response.content

def generate(prompt, negative_prompt, steps, guidance, width, height):
    params = {
        "negative_prompt": negative_prompt,
        "num_inference_steps": steps,
        "guidance_scale": guidance,
        "width": width,
        "height": height
    }
    output = get_completion(prompt, params)
    result_image = Image.open(io.BytesIO(output))
    return result_image

# Example using Gradio Interface 
# demo = gr.Interface(fn=generate,
#                     inputs=[
#                         gr.Textbox(label="Your prompt"),
#                         gr.Textbox(label="Negative prompt"),
#                         gr.Slider(label="Inference Steps", minimum=1, maximum=100, value=25,
#                                  info="In how many steps will the denoiser denoise the image?"),
#                         gr.Slider(label="Guidance Scale", minimum=1, maximum=20, value=7, 
#                                   info="Controls how much the text prompt influences the result"),
#                         gr.Slider(label="Width", minimum=64, maximum=512, step=64, value=512),
#                         gr.Slider(label="Height", minimum=64, maximum=512, step=64, value=512),
#                     ],
#                     outputs=[gr.Image(label="Result")],
#                     title="Image Generation with Stable Diffusion",
#                     description="Generate any image with Stable Diffusion",
#                     allow_flagging="never"
#                     )

# Example using Gradio Blocks
with gr.Blocks() as demo:
    gr.Markdown("# Image Generation with Stable Diffusion")
    with gr.Row():
        with gr.Column(scale=4):
            prompt = gr.Textbox(label="Your prompt") #Give prompt some real estate
        with gr.Column(scale=1, min_width=50):
            btn = gr.Button("Submit") #Submit button side by side!
    with gr.Accordion("Advanced options", open=False): #Let's hide the advanced options!
            negative_prompt = gr.Textbox(label="Negative prompt")
            with gr.Row():
                with gr.Column():
                    steps = gr.Slider(label="Inference Steps", minimum=1, maximum=100, value=25,
                      info="In many steps will the denoiser denoise the image?")
                    guidance = gr.Slider(label="Guidance Scale", minimum=1, maximum=20, value=7,
                      info="Controls how much the text prompt influences the result")
                with gr.Column():
                    width = gr.Slider(label="Width", minimum=64, maximum=512, step=64, value=512)
                    height = gr.Slider(label="Height", minimum=64, maximum=512, step=64, value=512)
    output = gr.Image(label="Result") #Move the output up too
            
    btn.click(fn=generate, inputs=[prompt,negative_prompt,steps,guidance,width,height], outputs=[output])

gr.close_all()

demo.launch(share=True, server_port=server_port)