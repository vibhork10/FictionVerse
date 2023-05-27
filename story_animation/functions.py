
import numpy as np
from safetensors import safe_open
from diffusers import StableDiffusionPipeline,DPMSolverMultistepScheduler
import os
import openai
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
from transformers import pipeline, set_seed
import random
import re

gpt2_pipe = pipeline('text-generation', model='Gustavosta/MagicPrompt-Stable-Diffusion', tokenizer='gpt2')


def generate_prompt(starting_text):
    seed = random.randint(1000, 10000)
    set_seed(seed)
    response = gpt2_pipe(starting_text, max_length=(len(starting_text) + 23), num_return_sequences=1)
    response_list = []
    for x in response:
        resp = x['generated_text'].strip()
        if resp != starting_text and len(resp) > (len(starting_text) + 4) and resp.endswith((":", "-", "—")) is False:
            response_list.append(resp+'\n')

    response_end = "\n".join(response_list)
    response_end = re.sub('[^ ]+\.[^ ]+','', response_end)
    response_end = response_end.replace("<", "").replace(">", "")

    if response_end != "":
        return response_end

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
def textwitimage_v1(text, image, font_size=15, font_path="comic.ttf", spacing=1.5, thickness=5, border_thickness=5):
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
    dialog_img = Image.new('RGB', (width + slant, height), (0, 0, 0))

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
    rounded_image = cv2.copyMakeBorder(concatenated_image, 0, new_border, new_border, new_border, cv2.BORDER_CONSTANT, value=(0, 0, 0))
    # rounded_image
    return rounded_image
def textwitimage_v2(text, image, font_size=15, font_path="comic.ttf", spacing=1.5, thickness=5, border_thickness=5):
    # Convert the image from OpenCV's BGR format to PIL's RGB format
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(image)
    img_width, img_height = image.size

    # Define the dimensions of the text field and the size of the font
    width_factor = 1.2  # Change this to your needs
    width = int(img_width * width_factor)
    height = img_height // 8


    # Define the slant factor
    slant = 15

    # Create a black image with the desired dimensions
    dialog_img = Image.new('RGB', (width + slant, height), (0, 0, 0))

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

    # Resize the dialog_img to match the new width
    dialog_img = cv2.resize(dialog_img, (width, height))

    # Convert the PIL Image object to a NumPy array and then resize
    image = np.array(image)
    image = cv2.resize(image, (width, img_height))

    # Calculate the overlap
    overlap = int(height * 0.8)
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
    rounded_image = cv2.copyMakeBorder(concatenated_image, 0, new_border, new_border, new_border, cv2.BORDER_CONSTANT, value=(0, 0, 0))
    # rounded_image
    return rounded_image