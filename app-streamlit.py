from diffusers import StableDiffusionPipeline, EulerDiscreteScheduler, StableDiffusionImg2ImgPipeline
import torch
#import gradio as gr
import streamlit as st
from io import BytesIO
from PIL import Image
import numpy as np
#from trans import LanguageTrans
import requests
import http.client
import random
import json

model_id = "stabilityai/stable-diffusion-2"

# Use the Euler scheduler here instead
scheduler = EulerDiscreteScheduler.from_pretrained(model_id, subfolder="scheduler")

device = "cuda" if torch.cuda.is_available() else "cpu"

def is_contains_chinese(strs):
    for _char in strs:
        if '\u4e00' <= _char <='\u9fa5':
            return True
    return False
def trans_youdao(sentence):
		"""有道翻译"""
		content = sentence
		data = {
			"i": content,
			"from": "AUTO",
			"to": "AUTO",
			"smartresult": "dict",
			"client": "fanyideskweb",
            "doctype": "json",
			"version": "2.1",
			"keyfrom": "fanyi.web",
			"action": "FY_BY_REALTIME",
			"typoResult": "false"
		}
		response = requests.post("http://fanyi.youdao.com/translate?smartresult=dict&smartresult=rule",
		                         data=data).json()
		resp = response["translateResult"][0][0]["tgt"]
        #print('{}'.format(resp))
        #print('{}'.format(resp))
		return(resp)
    #return resp
    

scale=7.5
steps=100
#i=0
#examples = [["An adventurer is approached by a mysterious stranger in the tavern for a new quest."],[]]

def gen(prompt, input_image, strength, scale, steps):
    
    if is_contains_chinese(prompt):
        
        prompt = trans_youdao(prompt)
    
    print(prompt)

    #print(input_image)
    if input_image is None:
        pipe = StableDiffusionPipeline.from_pretrained(model_id, scheduler=scheduler, revision="fp16", torch_dtype=torch.float16)
        pipe = pipe.to(device)
        pipe.enable_attention_slicing()
        image = pipe(prompt=prompt, height=768, width=768,guidance_scale=scale, num_inference_steps=steps).images
    else:
        pipe = StableDiffusionImg2ImgPipeline.from_pretrained(model_id, scheduler=scheduler, revision="fp16", torch_dtype=torch.float16)        
        #input_image=Image.fromarray(np.uint8(input_image))
        input_image = Image.open(BytesIO(input_image)).convert("RGB")
        input_image = input_image.resize((768, 768))
        pipe = pipe.to(device)
        pipe.enable_attention_slicing()
        image = pipe(prompt=prompt, init_image=input_image, strength=strength, guidance_scale=scale, num_inference_steps=steps).images
    
    
    return image[0]
#gr.Interface(fn=gen, inputs=['text', 'image', gr.Slider(0, 1, 0.75),gr.Slider(1, 10, 7.5), gr.Slider(1, maximum=100, value=50, step=1)], outputs='image', title="Bilingual Stable Diffusion 2.0", description="SD 2.0. <b>Welcome:</b> My implementation of SD2 both text and image to image, with both Chinese and English support.", article = "<b>Example</b>:a fabulous mountain view from the window of a luxury hotel with the sun rising in the sky <br> <b>Example</b>: 站在山上看远方的两个人<br> Code Ape: <a href=\"https://huggingface.co/qianli\">千里马</a>").launch(share=True)
st.header('Bilingual Stable Difussion 2 with txt2img and img2img')
txt=st.text_input('prompt(in Chinese or English)')
uploaded_file = st.file_uploader("Choose a file",type=['png', 'jpg'])
sdr_stg = st.slider('Strength:', 0.0, 1.0, 0.75)
sdr_scl=st.slider('Scale:', 1.0, 10.0, 7.5)
sdr_stp=st.slider('Step:', 1, 100, 50)

if uploaded_file is not None:
    # To read file as bytes:
    image = uploaded_file.getvalue()
    st.image(image, caption='Input image(optional)')
else:
    image=None

if st.button('Generate'):
    img=gen(txt, image, sdr_stg, sdr_scl, sdr_stp)

    if img is not None:
        st.image(img, caption='output image')
        imgByte=BytesIO()
        img.save(imgByte,format='PNG')
        byte_res=imgByte.getvalue()
        btn = st.download_button(
            label="Download image",
            data=byte_res,
            file_name="img_"+txt+".png",
            mime="image/png"
          )
        st.success('Done!')
if st.button('Clear and Rerun'):
    if st.image is not None:
        image=None
        st.experimental_rerun()
st.info("It's created by 千里马 and powered by Stability AI")
