import os
import io
from IPython.display import Image, display, HTML
from PIL import Image
import base64 
from transformers import pipeline
import gradio as gr
from transformers import pipeline
import requests, json

# Settings
_ = load_dotenv(find_dotenv()) # read local .env file
hf_api_token = os.environ['HF_API_TOKEN']   # You can set environment variables by executing this command: export HF_API_TOKEN=hf_TKPpXAasdasqwe1233
server_port = 12778
named_entity_recognition_endpoint_url = 'https://api-inference.huggingface.co/models/dslim/bert-base-NER'

# Summarization endpoint
def get_completion(inputs, parameters=None, ENDPOINT_URL=named_entity_recognition_endpoint_url): 
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

def merge_tokens(tokens):
    merged_tokens = []
    for token in tokens:
        if merged_tokens and token['entity'].startswith('I-') and merged_tokens[-1]['entity'].endswith(token['entity'][2:]):
            # If current token continues the entity of the last one, merge them
            last_token = merged_tokens[-1]
            last_token['word'] += token['word'].replace('##', '')
            last_token['end'] = token['end']
            last_token['score'] = (last_token['score'] + token['score']) / 2
        else:
            # Otherwise, add the token to the list
            merged_tokens.append(token)

    return merged_tokens

def ner(input):
    output = get_completion(input, parameters=None, ENDPOINT_URL=API_URL)
    merged_tokens = merge_tokens(output)
    return {"text": input, "entities": merged_tokens}

gr.close_all()
demo = gr.Interface(fn=ner,
                    inputs=[gr.Textbox(label="Text to find entities", lines=2)],
                    outputs=[gr.HighlightedText(label="Text with entities")],
                    title="NER with dslim/bert-base-NER",
                    description="Find entities using the `dslim/bert-base-NER` model under the hood!",
                    allow_flagging="never",
                    examples=["My name is Andrew, I'm building DeeplearningAI and I live in California", "My name is Poli, I live in Vienna and work at HuggingFace"])

demo.launch(share=True, server_port=server_port)