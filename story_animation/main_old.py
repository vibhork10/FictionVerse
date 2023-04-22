import numpy as np
from safetensors import safe_open
from diffusers import StableDiffusionPipeline,DPMSolverMultistepScheduler
import os
import openai
from functions import generate_story, next_line, prev_line, load_sd

# openai.api_key = "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"

# story_type = {
#     "fantasy": "You are an AI story writer assistant. You have to add a few lines to the story which the user has written.",
#     "science_fiction": "You are an AI story writer assistant. You have to add a few lines to the science fiction story which the user has written.",
#     "mystery": "You are an AI story writer assistant. You have to add a few lines to the mystery story which the user has written.",
#     "romance": "You are an AI story writer assistant. You have to add a few lines to the romance story which the user has written.",
#     "historical_fiction": "You are an AI story writer assistant. You have to add a few lines to the historical fiction story which the user has written.",
#     "horror": "You are an AI story writer assistant. You have to add a few lines to the horror story which the user has written.",
#     "adventure": "You are an AI story writer assistant. You have to add a few lines to the adventure story which the user has written.",
#     "comedy": "You are an AI story writer assistant. You have to add a few lines to the comedy story which the user has written.",
#     "None":"",
# }

# model_name = "gpt-3.5-turbo"

if __name__ =="__main__":
    os.makedirs("models_saved", exist_ok=True)
    os.makedirs("images", exist_ok=True)
    genre = input("Enter genre:")
    user_input = input("Enter input:")
    Aigenerate = generate_story(genre, user_input) #Takes in the genre and user input to generate ai story
    cos = True
    count = 0
    while(cos):
        line_np = input("Next line or previous (next/prev):")
        if line_np == "next":
            nw_line, count = next_line(Aigenerate, count)
        else:
            nw_line, count = prev_line(Aigenerate, count)
        print("Original Text line: ", nw_line, " Line Count:", count)
        seed = input("Enter seed:")
        prompt = input("Enter prompt:")
        options =  input("Enter options:")
        # def load_sd(prompt, seed, line_box, org_text, options)
        load_sd(prompt, seed, count, nw_line, options)
        cos = input("Enter condition (True/False):")




    print(Aigenerate)
