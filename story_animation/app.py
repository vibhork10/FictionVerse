import numpy as np
from safetensors import safe_open
from diffusers import StableDiffusionPipeline,DPMSolverMultistepScheduler
import os
import openai
from functions import generate_story, next_line, prev_line, load_sd
from fastapi import FastAPI
from pydantic import BaseModel
import requests
import cv2
from fastapi.middleware.cors import CORSMiddleware
import torch


openai.api_key = "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"
story_type = {
    "fantasy": "You are an AI story writer assistant. You have to add a few lines to the story which the user has written.",
    "science_fiction": "You are an AI story writer assistant. You have to add a few lines to the science fiction story which the user has written.",
    "mystery": "You are an AI story writer assistant. You have to add a few lines to the mystery story which the user has written.",
    "romance": "You are an AI story writer assistant. You have to add a few lines to the romance story which the user has written.",
    "historical_fiction": "You are an AI story writer assistant. You have to add a few lines to the historical fiction story which the user has written.",
    "horror": "You are an AI story writer assistant. You have to add a few lines to the horror story which the user has written.",
    "adventure": "You are an AI story writer assistant. You have to add a few lines to the adventure story which the user has written.",
    "comedy": "You are an AI story writer assistant. You have to add a few lines to the comedy story which the user has written.",
    "None":"",
}

model_name = "gpt-3.5-turbo"
app = FastAPI()
origins = [
    "http://localhost",
    "http://localhost:3000",
    "http://127.0.0.1",
    "http://127.0.0.1:3000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class Sd_input(BaseModel):
    prompt: str
    seed: int
    line_box: int
    org_text: str
    options: bool

class StoryInput(BaseModel):
    genre: str
    user_input: str

class Sline(BaseModel):
    input_text: str
    count: int

def round_corners(image, radius):
    height, width, channels = image.shape

    # Create a mask with the same dimensions and channels as the image and rounded corners
    #mask = np.zeros((height, width, 4), dtype=np.uint8)
    mask = np.zeros((height, width), dtype=np.uint8)
    cv2.rectangle(mask, (radius, 0), (width - radius, height), 255, -1)
    cv2.rectangle(mask, (0, radius), (width, height - radius), 255, -1)
    cv2.circle(mask, (radius, radius), radius, 255, -1)
    cv2.circle(mask, (width - radius, radius), radius, 255, -1)
    cv2.circle(mask, (radius, height - radius), radius, 255, -1)
    cv2.circle(mask, (width - radius, height - radius), radius, 255, -1)

    if channels == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2BGRA)

    alpha = np.zeros_like(mask, dtype=np.uint8)
    alpha[mask == 255] = 255
    image[:, :, 3] = alpha

    return image
def wrap_text(text, width, font, font_scale,thickness=1):
    words = text.split()
    wrapped_lines = []
    line = []
    for word in words:
        line.append(word)
        (tw, _), _ = cv2.getTextSize(' '.join(line), font, font_scale, thickness)
        if tw > width:
            line.pop()
            wrapped_lines.append(' '.join(line))
            line = [word]
    wrapped_lines.append(' '.join(line))

    return wrapped_lines
def round_corners_image(image, radius):
    height, width, channels = image.shape

    # Create a mask with the same dimensions and channels as the image and rounded corners
    #mask = np.zeros((height, width, 4), dtype=np.uint8)
    mask = np.zeros((height, width), dtype=np.uint8)
    cv2.rectangle(mask, (radius, 0), (width - radius, height), 255, -1)
    cv2.rectangle(mask, (0, radius), (width, height - radius), 255, -1)
    cv2.circle(mask, (radius, radius), radius, 255, -1)
    cv2.circle(mask, (width - radius, radius), radius, 255, -1)
    cv2.circle(mask, (radius, height - radius), radius, 255, -1)
    cv2.circle(mask, (width - radius, height - radius), radius, 255, -1)

    if channels == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2BGRA)

    image[mask == 0] = [0, 0, 0, 255]

    return image


def textwitimage(text,image,font_size=1,font=cv2.FONT_HERSHEY_SIMPLEX,spacing = 1.5,thickness=1):
  
  wrapped_lines = wrap_text(text, 512, font, font_size, thickness)
  text_height = int((len(wrapped_lines) - 1) * font_size * spacing * 20 + font_size * 20)
  margin = int(0.1 * text_height)

  height = text_height + 2 * margin

  blank_image =  np.ones((height, 512, 4), dtype=np.uint8)

  y = margin + int(font_size * 20)
  for line in wrapped_lines:
      (tw, th), _ = cv2.getTextSize(line, font, font_size, thickness)
      x = (512 - tw) // 2
      cv2.putText(blank_image, line, (x, y), font, font_size, (255, 255, 255), thickness, cv2.LINE_AA)
      y += int(th * spacing)

  blank_image = cv2.copyMakeBorder(blank_image, 0, margin, 0, 0, cv2.BORDER_CONSTANT, value=(0, 0, 0))

  img = round_corners_image(image, 50)
  cv2.imwrite("/content/images/"+str(5)+".png", img)
  concatenated_image = cv2.vconcat([blank_image,img])
  new_border = int(10)
  rounded_image = cv2.copyMakeBorder(concatenated_image, 0, new_border, new_border, new_border, cv2.BORDER_CONSTANT, value=(0, 0, 0))
  final = round_corners(rounded_image, 50)
  
 

  return final
# from fastapi.exceptions import RequestValidationError
# from fastapi.responses import JSONResponse
# @app.exception_handler(RequestValidationError)
# async def validation_exception_handler(request, exc):
#     return JSONResponse(status_code=422, content={"detail": str(exc)})

@app.post("/generate_story")
def generate_story(input: StoryInput):
    genre = input.genre
    user_input = input.user_input
    print(user_input, genre)
    if genre != "None":
        mymessages = [{"role": "system", "content": story_type[genre]}]
        mymessages.append({"role": "user", "content": user_input})

        response = openai.ChatCompletion.create(
            model=model_name,
            messages=mymessages
        )

        assistant_output = response['choices'][0]['message']['content']
    else:
        assistant_output = ""

    return {"story": user_input + " " + assistant_output}



    
@app.post("/next_line")
async def next_line(input: Sline):
    print("nexxxxxxxt")
    input_txt = input.input_text
    count = input.count
    text_lines = input_txt.split(".")
    if count < len(text_lines):
        new_line = text_lines[count].strip().replace("\n","")
        count += 1
        if (len(new_line) != 0):
            return {"line": new_line, "count": count}
    return {"line": "Empty line", "count": count+1}

@app.post("/prev_line")
async def prev_line(input: Sline):
    input_txt = input.input_text
    count = input.count
    text_lines = input_txt.split(".")
    if count < len(text_lines) and count > -1:
        new_line = text_lines[count].strip().replace("\n","")
        count -= 1
        if (len(new_line) != 0):
            return {"line": new_line, "count": count}
    return {"line": "Empty line", "count": count-1}

@app.post("/load_sd")
async def load_sd(input: Sd_input):
    print("ssddddddddddd")
    prompt = input.prompt
    seed = input.seed
    line_box = input.line_box
    org_text = input.org_text
    options = input.options
    print("sssssssssssssssssssssssssssss")
    pipe = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16)
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    pipe = pipe.to("cuda")
    generator = torch.Generator("cuda").manual_seed(int(seed))
    image = pipe(prompt, generator=generator, num_inference_steps=50).images[0]
    image = np.asarray(image)
    image = textwitimage(org_text, image)
    os.makedirs("images", exist_ok=True)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    with open("logger.txt", "a") as f:
        f.write("/n image generated")
    # print("/Downloads/story_animation/images/" + str(line_box) + ".png")
    cv2.imwrite("C:/Users/Subhrajit/Downloads/story_animation/images/" + str(line_box) + ".png", image)

    return {"image": "done"}