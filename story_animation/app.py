import numpy as np
from safetensors import safe_open
from diffusers import StableDiffusionPipeline,DPMSolverMultistepScheduler
import os
import openai
import random
# from functions import generate_story, next_line, prev_line, load_sd
from pathlib import Path
from fastapi import FastAPI, HTTPException
from reportlab.lib.pagesizes import letter, landscape
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader
import cv2
import os
from fastapi import FastAPI
from pydantic import BaseModel
import requests
import cv2
from fastapi.middleware.cors import CORSMiddleware
import torch
from fastapi.staticfiles import StaticFiles
from fpdf import FPDF
from fastapi.responses import FileResponse
import random
from PIL import Image, ImageDraw, ImageFont
import httpx
import json
import time
from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks
from fastapi.responses import JSONResponse
from typing import Any, Dict
from pydantic import BaseModel
from functions import textwitimage, textwitimage_v6, textwitimage_v1, textwitimage_v2, generate_prompt

openai.api_key = "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"
story_type = {
"fantasy": "You are an AI story writer assistant. Your task is to weave enchanting additions into the mystical fabric of the user's tale, breathing life into magical creatures, bewitching locales, and extraordinary adventures. Each sentence of the generated story should have less than 41 words and should end with a full stop.",
"science_fiction": "You are an AI story writer assistant. Your assignment is to expand the universe of the user's story with advanced technologies, alien civilizations, and futuristic dilemmas, propelling the narrative forward with your imaginative extrapolations. Each sentence of the generated story should have less than 41 words and should end with a full stop.",
"mystery": "You are an AI story writer assistant. Your job is to intricately add more suspense, clues, and unexpected turns to the user's gripping whodunit, thereby deepening the enigma and intrigue of the story. Each sentence of the generated story should have less than 41 words and should end with a full stop.",
"romance": "You are an AI story writer assistant. Your role is to instill more tender moments, passionate exchanges, and emotional dilemmas into the user's love story, adding more depth to the romantic dynamics of the characters. Each sentence of the generated story should have less than 41 words and should end with a full stop.",
"historical_fiction": "You are an AI story writer assistant. Your responsibility is to enrich the user's historical narrative by incorporating more vivid details from the era, creating a deeper sense of time and place, and heightening the historical tension. Each sentence of the generated story should have less than 41 words and should end with a full stop.",
"horror": "You are an AI story writer assistant. You are charged with the task of amplifying the eerie atmospheres, terrifying entities, and heart-stopping moments in the user's horror story, ratcheting up the fear factor in an unforgettable manner. Each sentence of the generated story should have less than 41 words and should end with a full stop.",
"adventure": "You are an AI story writer assistant. Your mission is to infuse the user's story with thrilling escapades, perilous quests, and awe-inspiring discoveries, providing an adrenaline rush to the narrative. Each sentence of the generated story should have less than 41 words and should end with a full stop.",
"comedy": "You are an AI story writer assistant. Your duty is to inject more laughter, wit, and comedic situations into the user's story, amplifying its humor quotient and creating unforgettable moments of hilarity. Each sentence of the generated story should have less than 41 words and should end with a full stop.",
"None": ""
}

model_name = "gpt-3.5-turbo"
app = FastAPI()
origins = [
  'http://54.157.42.127:3000',
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE"],
    allow_headers=["*"],
)

# app.mount("/images", StaticFiles(directory="images"), name="images")

class Sd_input(BaseModel):
    prompt: str
    line_box: int
    org_text: str
    style: str
    display: str
    uuid: str
    seed: int

class StoryInput(BaseModel):
    genre: str
    user_input: str

class Sline(BaseModel):
    input_text: str
    count: int
class StoryOutput(BaseModel):
    story: str
from multiprocessing import Process, Manager

class StoryTask:
    def __init__(self, genre, user_input, result_dict):
        self.genre = genre
        self.user_input = user_input
        self.result_dict = result_dict

    def run(self):
        mymessages = [{"role": "system", "content": story_type[self.genre]}]
        mymessages.append({"role": "user", "content": self.user_input})

        response = openai.ChatCompletion.create(
            model=model_name,
            messages=mymessages
        )

        assistant_output = response['choices'][0]['message']['content']
        self.result_dict["story"] = self.user_input + " " + assistant_output

def openai_task(story_task: StoryTask):
    story_task.run()

@app.post("/generate_story", response_model=StoryOutput)
def generate_story(input: StoryInput):
    with Manager() as manager:
        result_dict = manager.dict()
        print(input.genre, input.user_input)
        story_task = StoryTask(input.genre, input.user_input, result_dict)
        p = Process(target=openai_task, args=(story_task,))
        p.start()
        p.join()
        # Convert the Manager's dict to a standard Python dict before returning it
        return dict(result_dict)

@app.post("/next_line")
async def next_line(input: Sline):
    input_txt = input.input_text
    count = input.count
    text_lines = input_txt.split(".")
    line_count = len(text_lines)

    if count >= line_count:
        count = 0

    new_line = text_lines[count].strip().replace("\n", "")
    count += 1

    if len(new_line) != 0:
        return {"line": new_line, "count": count}

    return {"line": "Empty line", "count": count}


@app.post("/prev_line")
async def prev_line(input: Sline):
    input_txt = input.input_text
    count = input.count
    text_lines = input_txt.split(".")
    line_count = len(text_lines)

    if count <= 0:
        count = line_count - 1
    else:
        count -= 1

    new_line = text_lines[count].strip().replace("\n", "")

    if len(new_line) != 0:
        return {"line": new_line, "count": count}

    return {"line": "Empty line", "count": count}


@app.get("/{folder_id}_images/{image_name}")
async def serve_images(folder_id: str, image_name: str):
    image_path = Path(f'./{folder_id}_images/{image_name}')
    
    if not image_path.exists() or not image_path.is_file():
        raise HTTPException(status_code=404, detail="Image not found")

    return FileResponse(image_path)

from multiprocessing import Process, Manager

class SdTask:
    def __init__(self, input: Sd_input, result_dict):
        self.input = input
        self.result_dict = result_dict

    def run(self):
        prompt = self.input.prompt
        line_box = self.input.line_box
        org_text = self.input.org_text
        style_opt = self.input.style
        display_opt = self.input.display
        nwuuid = self.input.uuid
        nwseed = self.input.seed
        if line_box == 0:
            nwseed = random.randint(1555, 3000)

        if org_text != "Empty line":
            if "ogkalu/Comic-Diffusion" in style_opt:
                comic_style = style_opt.split("-")[-1]
                prompt = prompt+" "+comic_style
                style_opt = "ogkalu/Comic-Diffusion"

            pipe = StableDiffusionPipeline.from_pretrained(style_opt, torch_dtype=torch.float16)
            pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
            pipe = pipe.to("cuda")
            generator = torch.Generator("cuda").manual_seed(int(nwseed))
            prompt = generate_prompt(prompt)
            print("prompttttttttttttt", prompt)
           
            image = pipe(prompt, height=512, width=512, generator=generator, num_inference_steps=50).images[0]
            image = np.asarray(image)
            if display_opt == "default-style":
                image = textwitimage(org_text, image)
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            elif display_opt == "Style-1":
                image = textwitimage_v1(org_text, image)
            else:
                image = textwitimage_v6(org_text, image)

            folderpath = "./"+nwuuid+"_images"
            os.makedirs(folderpath, exist_ok=True)

            cv2.imwrite(folderpath+"/" + str(line_box) + ".png", image)

        self.result_dict["image"] = "done"
        self.result_dict["uuid"] = nwuuid
        self.result_dict["seed"] = nwseed


def openai_sd_task(sd_task: SdTask):
    sd_task.run()


@app.post("/load_sd")
def load_sd(input: Sd_input):
    with Manager() as manager:
        result_dict = manager.dict()
        sd_task = SdTask(input, result_dict)
        p = Process(target=openai_sd_task, args=(sd_task,))
        p.start()
        p.join()
        return dict(result_dict)




@app.get("/download_pdf")
async def download_pdf(uuid: str):
    folderpath = "./"+uuid+"_images"
    image_paths = sorted(os.listdir(folderpath))

    pdf_file = "./generated_images.pdf"
    c = canvas.Canvas(pdf_file, pagesize=landscape(letter))
    
    for image_path in image_paths:
        image_full_path = folderpath + "/" + image_path
        img = cv2.imread(image_full_path)
        height, width, _ = img.shape

        # Convert from pixel to point
        width, height = width * 0.75, height * 0.75

        image = ImageReader(image_full_path)
        c.setPageSize((width, height))
        c.drawImage(image, 0, 0, width=width, height=height)
        c.showPage()
        
    c.save()

    return FileResponse(pdf_file, media_type="application/pdf", filename="generated_images.pdf")


