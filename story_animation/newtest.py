if __name__ == "__main__":
    import requests
    url_genstory = "http://localhost:8000/generate_story"


    genre = input("Enter genre:")
    user_input_text = input("Enter input text:")
    input_data = {
        "genre": genre,
        "user_input": user_input_text
    }
    url_next = "http://localhost:8000/next_line"
    url_prev = "http://localhost:8000/prev_line"
    response_genstory = requests.post(url_genstory, json=input_data)
    # response_next = requests.post(url_next, json=input_data)
    # response_prev = requests.post(url_prev, json=input_data)
    
    cos = True
    count = 0
    generated_story = response_genstory.json()["generated_story"]
    print(generated_story)
    while(cos):
        input_line = {
            "input_text": generated_story,
            "count": count
        }
        line_np = input("Next line or previous (next/prev):")
        if line_np == "next":
            response_line = requests.post(url_next, json=input_line)
        else:
            response_line = requests.post(url_prev, json=input_line)

        print("Original Text line: ", response_line.json()["line"], " Line Count:",  response_line.json()["count"])
        prompt_text = input("Enter prompt text:")
        seed_text = input("Enter seed value:")
        count = response_line.json()["count"]


        url_sd_genstory = "http://localhost:8000/load_sd"

        input_data = {
            "prompt": prompt_text,
            "seed": seed_text, 
            "line_box": count,
            "org_text": response_line.json()["line"],
            "options": "1"
        }
        response_genstory = requests.post(url_sd_genstory, json=input_data)

        cos = input("Enter condition (True/False):")