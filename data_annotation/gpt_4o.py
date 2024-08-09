import base64
import requests
import json
import os
from data_annotation.utilities import *
from tqdm import tqdm, trange

'''
Version gpt-4o-2024-05-13
'''

# Function to encode the image
def encode_image(image_path):
  with open(image_path, "rb") as image_file:
    return base64.b64encode(image_file.read()).decode('utf-8')


def call_api(image_paths: list[str], prompt=None, api_key=None):
  # Getting the base64 string
  print("Images:", image_paths)
  base64_images = [encode_image(image_path) for image_path in image_paths]
  base64_images_objs = [{
            "type": "image_url",
            "image_url": {
              "url": f"data:image/jpeg;base64,{image}",
              "detail": "low" 
            },
          } for image in base64_images]

  content = [{
            "type": "text",
            "text": prompt,
          }] + base64_images_objs

  headers = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {api_key}"
  }

  payload = {
    "model": "gpt-4o",
    "messages": [
      {
        "role": "user",
        "content": content
      }
    ],
    "max_tokens": 4000
  }

  response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
  # handling unsuccessful request
  try: 
     response.json()['choices'][0]['message']['content']
    #  print(response.json()['choices'][0]['message']['content'])
  except:
     print(f"Error: An exception occurred, img_dirs: {image_paths}")
     
  return response


def main(args):
  f = open('./config/openai_api_key.json')
  conf = json.load(f)
  f.close()
  # OpenAI API Key
  api_key = conf["openai_api_key"]
  
  # Get prompts
  prompts = prepare_prompts(dataset = args.dataset,
                            model = args.prompt_type)

  # Call GPT-4o
  responds = []
  print(f"\n\ No. of prompts: {len(prompts)}")
  last_idx = args.start_id - 1

  for idx in tqdm(range(len(prompts)), desc='Processing'):
    prompt = prompts[idx]
    if (idx+1) > args.stop_id and args.stop_id > 0:
        break
    if (idx+1) < args.start_id: 
      continue
    if args.few_shot_with_example_images:
      response = call_api(args.example_images + [prompt['image_dir']], prompt['prompt'], api_key)
    else: 
      response = call_api([prompt['image_dir']], prompt['prompt'], api_key)
    try: 
      response.json()['choices'][0]['message']['content'] # try to get generated text from the message
      responds.append(prompt)
      responds[-1]['respond'] = response.json()['choices'][0]['message']['content']
      
    except:
      print(f"Error: An exception occurred")
    if (idx+1) % args.file_length == 0:
      # Convert and write JSON object to file
      with open(f"img_prompt_respond_{last_idx+1}-{idx+1}.json", "w") as outfile: 
          json.dump(responds, outfile, indent=4)
      last_idx = idx + 1
      responds = []
  
  with open(f"img_prompt_respond_{last_idx+1}-end.json", "w") as outfile: 
      json.dump(responds, outfile, indent=4)
      


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--stop-id", type=int, default=-1)
    parser.add_argument("--start-id", type=int, default=1)
    parser.add_argument("--file-length", type=int, default=500)
    parser.add_argument("--dataset", type=str, default='meme_text_retrieval')
    parser.add_argument("--prompt-type", type=str, default='gpt-4o-all-data')
    parser.add_argument("--few-shot-with-example-images", type=bool, default=False)
    example_image_paths = ['./data/figmemes/images/1483414267738.jpg', 
                          './data/figmemes/images/1484978535453.jpg', 
                          './data/figmemes/images/1483781574074.png']
    parser.add_argument("--example-images", type=list, default=example_image_paths)
    args = parser.parse_args()
    main(args)


