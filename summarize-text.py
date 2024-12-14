# L1: NLP tasks with a simple interface 🗞️

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
hf_api_token = 'hf_TKPpXAasdasqwe1233'   # You can set environment variables by executing this command: export HF_API_TOKEN=hf_TKPpXAasdasqwe1233
server_port = 17724
text_summarization_endpoint_url = 'https://api-inference.huggingface.co/models/sshleifer/distilbart-cnn-12-6'

# Summarization endpoint
def get_completion(inputs, parameters=None,ENDPOINT_URL=text_summarization_endpoint_url): 
    headers = {
      "Authorization": f"Bearer {hf_api_token}",
      "Content-Type": "application/json"
    }
    data = { "inputs": inputs }
    if parameters is not None:
        data.update({"parameters": parameters})
    response = requests.request("POST",
                                ENDPOINT_URL, headers=headers,
                                data=json.dumps(data)
                               )
    return json.loads(response.content.decode("utf-8"))

get_completion = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")

def summarize(input):
    output = get_completion(input)
    return output[0]['summary_text']

gr.close_all()

demo = gr.Interface(fn=summarize, 
                    inputs=[gr.Textbox(label="Text to summarize", lines=6)],
                    outputs=[gr.Textbox(label="Result", lines=3)],
                    title="Text summarization with distilbart-cnn",
                    description="Summarize any text using the `shleifer/distilbart-cnn-12-6` model under the hood!"
                   )
demo.launch(share=True, server_port=server_port)