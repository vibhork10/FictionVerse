
import torch
import cv2
import os
import numpy as np
import cv2
import subprocess
from zipfile import ZipFile 
import os
import zipfile
import os
from diffusers import StableDiffusionPipeline,DPMSolverMultistepScheduler
import re
import os
import openai
import gradio as gr

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

def generate_story(genre, user_input):
    if genre != "None":
      mymessages = [{"role": "system", "content": story_type[genre]}]
      mymessages.append({"role": "user", "content": user_input})

      response = openai.ChatCompletion.create(
          model=model_name,
          messages=mymessages
      )

      assistant_output = response['choices'][0]['message']['content']
      # return user_input + "\n" + assistant_output
    else:
      assistant_output = ""
    return user_input + " " + assistant_output


def change_textbox(choice):
    if choice == "object":
        return "object"
    else:
        return "style"


def preview(files, sd: gr.SelectData):
    print(files)
    return files[sd.index].name
import shutil

def save_files(file_location, foldername, progress=gr.Progress(track_tqdm=True)):  
    file_url = file_location.split("/")[5]
    filefull= "https://docs.google.com/uc?export=download&confirm=t&id={}".format(file_url)
    liste = subprocess.run(["wget",filefull, "-O", foldername])
    with zipfile.ZipFile(foldername, 'r') as zip:
      zip.extractall('/content/')
    temp="Folder saved to /content/"+foldername
    return gr.update(lines=1, visible=True,value=str(temp))

def train_func(input, temp_type, progress=gr.Progress(track_tqdm=True)):
  input_file = input.split(".")[0]
  list_dir = subprocess.run(["lora_pti", "--pretrained_model_name_or_path","runwayml/stable-diffusion-v1-5","--instance_data_dir", "/content/"+input_file, \
                              "--output_dir", "/content/output/", "--train_text_encoder", "--resolution", "512", "--train_batch_size", "1", "--gradient_accumulation_steps", "4", \
                              "--scale_lr", "--learning_rate_unet", "1e-4", "--learning_rate_text", "5e-5", "--learning_rate_ti", "5e-2", "--color_jitter","--lr_scheduler", "linear", \
                              "--lr_warmup_steps", "0", "--placeholder_tokens", "<s1>", "--initializer_tokens", "man", "--use_template", temp_type, "--save_steps", "50", "--max_train_steps_ti", \
                              "1000", "--max_train_steps_tuning", "1000","--perform_inversion", "True", "--clip_ti_decay", "--weight_decay_ti", "0.000", "--weight_decay_lora", "0.001", \
                              "--continue_inversion", \
                              "--continue_inversion_lr", \
                              "--device", "cuda:0", "--lora_rank", "16"], capture_output=True)
  l = input_file+temp_type
  return gr.update(lines=1, visible=True,value=str(l))

def textwitimage(text,image):
  # Load the input image
  h, w, _ = image.shape
  print(h,w)
  # Create a white background image of the same size
  white_bg = np.full((h, w, 3), 255, dtype=np.uint8)

  # Define the font and font parameters
  font = cv2.FONT_HERSHEY_SIMPLEX
  font_thickness = 2
  font_scale = 1

  # Calculate the maximum width and height of the text box
  text_box_width = int(0.9 * w)
  text_box_height = int(0.9 * h)

  # Split the text into words
  words = text.split()

  # Initialize the lines list with the first word
  lines = [words[0]]
  print(lines)
  # Iterate over the remaining words and add them to lines, splitting lines as necessary
  for word in words[1:]:
      # Add the word to the current line
      line = lines[-1] + ' ' + word
      print(line)
      # Get the size of the line
      line_size, _ = cv2.getTextSize(line, font, font_scale, font_thickness)
      print(line_size)
      # If the line is too long, start a new line
      if line_size[0] > text_box_width:
          lines.append(word)
      else:
          lines[-1] = line

  # Calculate the font size based on the height of the text box
  font_size = 1
  # Create a blank image to draw the text on
  text_image = np.full((text_box_height, text_box_width, 3), 255, dtype=np.uint8)

  # Draw the lines of text on the image, starting at the top
  text_y = int(0.3*h)
  for line in lines:
      line_size, _ = cv2.getTextSize(line, font, font_size, font_thickness)
      text_x = int((text_box_width - line_size[0]) / 2)
      cv2.putText(white_bg, line, (text_x, text_y), font, font_size, (0, 0, 0), font_thickness)
      text_y += line_size[1] + 10*font_size

  # Calculate the position to place the text at the center
  text_x = (w - text_box_width) // 2
  text_y = (h - text_box_height) // 2

  # Paste the text image onto the white background
  #white_bg[text_y:text_y+text_box_height, text_x:text_x+text_box_width, :] = text_image

  # Concatenate the input image and the white background image horizontally
  concatenated_image = cv2.vconcat([image, white_bg])
  return concatenated_image
  
def convert_model(artstyle, style, progress=gr.Progress(track_tqdm=True)):
  output = "/content/output/final_lora.safetensors"
  # artsyle_loc = "/content/"+artstyle
  os.makedirs("/content/models_saved", exist_ok=True)
  os.rename(output, "/content/models_saved/"+str(artstyle)+"_"+str(style)+".safetensors")
  return gr.update(lines=1, visible=True,value="Conversion has been completed model can be found at /content/models_saved/")


# def convert_vid(images_list):
#   result = cv2.VideoWriter('/content/newanimation.mp4', 
#                           cv2.VideoWriter_fourcc(*'MP4V'),
#                           2,(512,512))
#   for img in images_list:
#     image = img
#     iamge = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
#     result.write(image)
#   result.release()

def return_value(prompt, image):
  return prompt, image


def load_sd(prompt, seed, line_box, org_text, options):        
  with open("logger.txt", "a") as f:
    f.write("Prompt "+str(prompt)+"Seed "+str(seed)+"line_box "+str(line_box)+"options "+str(options))                                                                                                                                                                                                                                                                                                                                                                         
  if options == "1":
    print("sssssssssssssssssssssssssssss")
    pipe = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16)
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)   
    pipe = pipe.to("cuda")
    generator = torch.Generator("cuda").manual_seed(int(seed))  
    image = pipe(prompt, generator = generator, num_inference_steps=50).images[0] 
    image = np.asarray(image)
    image = textwitimage(org_text,image)
    os.makedirs("images", exist_ok=True)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    cv2.imwrite("/images/"+str(line_box)+".png", image)
  else:
    input("1111111111111111111111111:")
    from lora_diffusion import patch_pipe, tune_lora_scale, image_grid    
    pipe = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16)
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)   
    model_loc= "/models_saved/"+options
    patch_pipe(pipe,model_loc, patch_text=True,patch_ti=True, patch_unet=True)
    pipe = pipe.to("cuda")
    sc = 0.6
    generator = torch.Generator("cuda").manual_seed(int(seed))
    tune_lora_scale(pipe.unet, sc)
    tune_lora_scale(pipe.text_encoder,sc)
    if options.split("_")[1].split(".")[0] == "style":
      image = pipe(prompt+", in style of <s1>", generator = generator, num_inference_steps=50).images[0]
    else:
      name_variable = "<s1>"
      prompt = re.sub(r'<(.+?)>', name_variable, prompt)
      with open("logger.txt", "a") as f:
        f.write("changed"+prompt)
      neg = "double face, hands, wrist, Ugly, Duplicate, Extra fingers, Mutated hands, Poorly drawn face, Mutation, Deformed, Blurry, Bad anatomy, Bad proportions, Extra limbs, cloned face, Disfigured, Missing arms, Missing legs, Extra arms, Extra legs, Fused fingers, Too many fingers, Long neck, writing, letters, Multiple bodies, multiple heads, extra hands, extra fingers, ugly, skinny, extra leg, extra foot, blur, bad anatomy, double body, stacked body, fused hands, fused body, fused heads, fused legs, fused feet, multiple faces, conjoined, siamese twin, double faces, two faces, texts, watermarked, watermark, logo, face out of frame, stacked background, out of frame portrait, bucktoothed, cropped, yellow"
      image = pipe(prompt, negative_prompt=neg, generator = generator, num_inference_steps=50).images[0]
    image = np.asarray(image)
    image = textwitimage(org_text,image)
    os.makedirs("/images/", exist_ok=True)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    cv2.imwrite("/images/"+str(line_box)+".png", image)
  


def change_textbox_style(choice):
    if choice == "object":
        return "object"
    else:
        return "style"
def create_pdf():
  from PIL import Image
  images_list = os.listdir("/content/images/")
  images_list.sort()
  images = []
  img1 = Image.open(os.path.join("/content/images/", images_list[0]))
  img_1 = img1.convert('RGB')
  for filename in images_list[1:]:
      if filename.endswith('.png'):
          # Open the image file using PIL
          img = Image.open(os.path.join("/content/images/", filename))

          # Convert the image to PDF format
          img_nw = img.convert('RGB')
          images.append(img_nw)
  
  img_1.save('/content/convert.pdf', save_all=True, append_images=images)
  return "Converted to pdf"


def next_line(input_text, count):
  count = int(count)
  text_lines = input_text.split(".")
  if count < len(text_lines):
    new_line = text_lines[count].strip().replace("\n","")
    count += 1
    if (len(new_line) != 0):
      return new_line, count
  return "Empty line", count+1

def prev_line(input_text, count):
  count = int(count)
  text_lines = input_text.split(".")
  if count < len(text_lines) and count > -1:
    new_line = text_lines[count].strip().replace("\n","")
    count -= 1
    if (len(new_line) != 0):
      return new_line, count
  return "Empty line", count-1

