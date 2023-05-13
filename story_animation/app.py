import numpy as np
from safetensors import safe_open
from diffusers import StableDiffusionPipeline,DPMSolverMultistepScheduler
import os
import openai
# from functions import generate_story, next_line, prev_line, load_sd
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

app.mount("/images", StaticFiles(directory="images"), name="images")

class Sd_input(BaseModel):
    prompt: str
    line_box: int
    org_text: str
    style: str
    display: str

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
def wrap_text_pil(text, width, font):
    words = text.split()
    lines = []
    line = []
    for word in words:
        line.append(word)
        if font.getsize(' '.join(line))[0] > width:
            line.pop()
            lines.append(' '.join(line))
            line = [word]
    lines.append(' '.join(line))
    return lines
def round_corners_w(image, rounding_value):
    img = cv2.cvtColor(image, cv2.COLOR_BGR2BGRA)
    mask = np.zeros_like(img)
    h, w, _ = img.shape
    cv2.rectangle(mask, (rounding_value, rounding_value), (w - rounding_value, h - rounding_value), (255, 255, 255, 255), -1, cv2.LINE_AA)
    result = cv2.addWeighted(img, 1, mask, 0, 0)
    result = cv2.cvtColor(result, cv2.COLOR_BGRA2BGR)
    return result
def draw_thick_polygon(draw, points, outline, fill, thickness):
    # Draw the filled polygon
    draw.polygon(points, fill=fill)

    # Draw the outline by creating lines between each point
    for i in range(len(points)):
        draw.line([points[i-1], points[i]], fill=outline, width=thickness)
def textwitimage_v6(text, image, font_size=15, font_path="comic.ttf", spacing=1.5, thickness=5, border_thickness=5):
    # Convert the image from OpenCV's BGR format to PIL's RGB format
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(image)
    img_width, img_height = image.size

    # Define the dimensions of the text field and the size of the font
    width = img_width
    height = img_height // 8

    # Define the slant factor
    slant = 15

    # Create a white image with the desired dimensions
    dialog_img = Image.new('RGB', (width + slant, height), (255, 255, 255))

    # Draw the parallelogram shape on the image
    draw = ImageDraw.Draw(dialog_img)
    points = [(0, height), (width, height), (width + slant, 0), (slant, 0)]
    draw_thick_polygon(draw, points, outline=(0, 0, 0), fill=(255, 255, 255), thickness=border_thickness)

    # Add the text to the cropped text field
    font = ImageFont.truetype(font_path, font_size)
    wrapped_lines = wrap_text_pil(text, width - 2 * slant, font)
    text_y = (height - font_size * len(wrapped_lines)) // 2
    text_offset_x = slant // 2  # Add an offset to move the text away from the edges
    text_offset_y = 5  # Add an offset to move the text away from the edges
    for line in wrapped_lines:
        text_size = font.getsize(line)
        text_x = (width - text_size[0]) // 2 + text_offset_x
        draw.text((text_x, text_y + text_offset_y), line, font=font, fill=(0, 0, 0))
        text_y += font_size

    # Convert the dialogue image back to OpenCV's BGR format
    dialog_img = np.array(dialog_img)
    dialog_img = cv2.cvtColor(dialog_img, cv2.COLOR_RGB2BGR)

    # Resize the dialog_img to match the width of img
    dialog_img = cv2.resize(dialog_img, (img_width, height))

    # Calculate the overlap
    overlap = int(height * 0.8)
    image = np.array(image)
    image_overlap = image[:overlap, :, :]
    dialog_overlap = dialog_img[-overlap:, :, :]

    # Blend the overlapping region using addWeighted
    blended_overlap = cv2.addWeighted(dialog_overlap, 1.3, image_overlap, 0.0, 0)

    # Replace the overlapping region in the dialog image with the blended region
    dialog_img[-overlap:, :, :] = blended_overlap

    # Combine the dialog image and the original image
    concatenated_image = np.vstack((dialog_img, image[overlap:, :, :]))

    # Add border and round corners
    new_border = int(10)
    rounded_image = cv2.copyMakeBorder(concatenated_image, 0, new_border, new_border, new_border, cv2.BORDER_CONSTANT, value=(255, 255, 255))
    # rounded_image
    return rounded_image
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


@app.post("/load_sd")
async def load_sd(input: Sd_input):
    print("ssddddddddddd")
    prompt = input.prompt
    seed = 1337877655
    line_box = input.line_box
    org_text = input.org_text
    style_opt = input.style
    display_opt = input.display
    print("sssssssssssssssssssssssssssss")
    if "ogkalu/Comic-Diffusion" in style_opt:
        comic_style = style_opt.split("-")[-1]
        prompt = prompt+" "+comic_style
        style_opt = "ogkalu/Comic-Diffusion"
    pipe = StableDiffusionPipeline.from_pretrained(style_opt, torch_dtype=torch.float16)
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    pipe = pipe.to("cuda")
    generator = torch.Generator("cuda").manual_seed(int(seed))
    image = pipe(prompt, height=512, width=512, generator=generator, num_inference_steps=50).images[0]
    image = np.asarray(image)
    if display_opt == "default-style":
        image = textwitimage(org_text, image)
    else:
        image = textwitimage_v6(org_text, image)
    os.makedirs("images", exist_ok=True)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    with open("logger.txt", "a") as f:
        f.write("/n image generated")
    # print("/Downloads/story_animation/images/" + str(line_box) + ".png")
    cv2.imwrite("./images/" + str(line_box) + ".png", image)

    return {"image": "done"}



@app.get("/download_pdf")
async def download_pdf():
    pdf = FPDF(orientation='P', unit='pt', format=(532, 735))
    image_paths = sorted(os.listdir("./images"))

    for image_path in image_paths:
        pdf.add_page()
        pdf.image("./images/" + image_path, 0, 0, 532, 735)

    pdf_file = "images/generated_pdf.pdf"
    pdf.output(pdf_file)

    return FileResponse(pdf_file, media_type="application/pdf", filename="generated_images.pdf")


# prompthero/openjourney-v4
#andite/anything-v4.0
#ogkalu/Comic-Diffusion