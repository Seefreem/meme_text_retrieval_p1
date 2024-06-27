from llava.serve.cli import *
import argparse
from tqdm import tqdm

# python -m llava.serve.cli --model-path liuhaotian/llava-v1.6-mistral-7b  --image-file ./memes/1a3hfy.jpg  --load-4bit
# parser = argparse.ArgumentParser(description='Process some integers.')
parser = argparse.ArgumentParser()
parser.add_argument("--model-path", type=str, default="liuhaotian/llava-v1.6-mistral-7b")
parser.add_argument("--model-base", type=str, default=None)
parser.add_argument("--image-file", type=str, required=True, default='./memes/1a3hfy.jpg')
parser.add_argument("--device", type=str, default="cuda")
parser.add_argument("--conv-mode", type=str, default=None)
parser.add_argument("--temperature", type=float, default=0.2)
parser.add_argument("--max-new-tokens", type=int, default=512)
parser.add_argument("--load-8bit", action="store_true")
parser.add_argument("--load-4bit", action="store_true")
parser.add_argument("--debug", action="store_true")
args = parser.parse_args('--model-path liuhaotian/llava-v1.6-mistral-7b  --image-file ./memes/1a3hfy.jpg  --load-4bit'.split())


# Model
disable_torch_init()

model_name = get_model_name_from_path(args.model_path)
tokenizer, model, image_processor, context_len = load_pretrained_model(args.model_path, args.model_base, model_name, args.load_8bit, args.load_4bit, device=args.device)

if "llama-2" in model_name.lower():
    conv_mode = "llava_llama_2"
elif "mistral" in model_name.lower():
    conv_mode = "mistral_instruct"
elif "v1.6-34b" in model_name.lower():
    conv_mode = "chatml_direct"
elif "v1" in model_name.lower():
    conv_mode = "llava_v1"
elif "mpt" in model_name.lower():
    conv_mode = "mpt"
else:
    conv_mode = "llava_v0"

if args.conv_mode is not None and conv_mode != args.conv_mode:
    print('[WARNING] the auto inferred conversation mode is {}, while `--conv-mode` is {}, using {}'.format(conv_mode, args.conv_mode, args.conv_mode))
else:
    args.conv_mode = conv_mode



about_section = [r'Batman Slapping Robin, also known as "My Parents Are Dead," is an exploitable image of the DC Comics superhero Batman slapping his protégé Robin in mid-conversation. Taken from a comic book published in 1965, the single-panel illustration has inspired a series of parodies featuring custom-captioned speech bubbles on 4chan, Reddit and elsewhere.',
                 r'You Like Krabby Patties Don’t You, Squidward?, also known by the snowclone "You Like X, Don’t You, Y?", is a reaction image, catchphrase and exploitable taken from an episode of SpongeBob SquarePants showing SpongeBob with a comically knowing grin after he realizes that Squidward likes Krabby Patties. The image has been used in similar contexts online, usually from a person discovering another\'s secret, and sometimes as a phrasal template with the key elements replaced to make a pop culture reference.',
                 r'Philosoraptor is an advice animal image macro series featuring an illustration of a Velociraptor paired with captions depicting the dinosaur as being deeply immersed in metaphysical inquiries or unraveling quirky paradoxes.',
                 r'X, X Everywhere is a phrasal template typically used for a wide range of cultural references, from quotable lines heard in popular films and TV shows to more obscure things found in viral videos and games. Many instances of "X Everywhere" may be seen as an indicator for emerging trends, while some should be read as a grievance or complaint against them, depending on the context.',
                 r'Hide The Pain Harold, also simply referred to as "Harold" or "Maurice" is the nickname given to a senior stock photography model whose facial expression appears to indicate suppressed pain and/or discomfort.']
img_dirs = ["./memes/1a3hfy.jpg", 
            "./memes/1a3k9c.jpg",
            "./memes/1a8wvp.jpg",
            "./memes/1abayq.jpg",
            "./memes/1abovr.jpg"]
inputs_template = []
for i in range(len(about_section)):
    inputs_template.append({'image_dir':img_dirs[i], 'prompt':f'Here is the context of the meme: "{about_section[i]}". First, based the given context, read the text in this image and explain the meme. Then, provide information for the following categories: \n Visual Elaboration (focus on the main content): \n Detected Text: \n Meaning of the Meme (briefly):'})
    inputs_template.append({'image_dir':img_dirs[i], 'prompt':f'Here is the context of the meme: "{about_section[i]}". First, based the given context, read the text in this image and explain the meme. Then, choose the most suitable literary device from the given category words: allegory, alliteration, allusion, amplification, anagram, analogy, anthropomorphism, antithesis, chiasmus, circumlocution, euphemism, hyperbole, imagery, metaphor, onomatopoeia, oxymoron, paradox, personification, portmanteau, pun, satire, simile, and symbolism. If no suitable word, use "None" as the category word. Only reply with the chosen word.'})
    inputs_template.append({'image_dir':img_dirs[i], 'prompt':f'Here is the context of the meme: "{about_section[i]}". First, based the given context, explain the meme. Then, choose the most suitable emotion word from the given category words: fear, anger, joy, sadness, surprise, disgust, guilt, contempt, shame, embarrassment, envy, jealousy, love, hate, and interest. If no suitable word, use "None" as the category word.  Only reply with the chosen word.'})
    
    inputs_template.append({'image_dir':img_dirs[i], 'prompt':'First, read the text in this image and explain the meme. Then, provide information for the following categories: Visual Elaboration (focus on the main content): Detected Text: Meaning of the Meme (briefly):'})
    inputs_template.append({'image_dir':img_dirs[i], 'prompt':'First, read the text in this image and explain the meme. Then, choose the most suitable literary device from the given category words: allegory, alliteration, allusion, amplification, anagram, analogy, anthropomorphism, antithesis, chiasmus, circumlocution, euphemism, hyperbole, imagery, metaphor, onomatopoeia, oxymoron, paradox, personification, portmanteau, pun, satire, simile, and symbolism. If no suitable word, use "None" as the category word. Only reply with the chosen word.'})
    inputs_template.append({'image_dir':img_dirs[i], 'prompt':'First, read the text in this image and explain the meme. Then, choose the most suitable emotion word from the given category words: fear, anger, joy, sadness, surprise, disgust, guilt, contempt, shame, embarrassment, envy, jealousy, love, hate, and interest. If no suitable word, use "None" as the category word.  Only reply with the chosen word.'})
    


responds = []
for i in tqdm(range(len(inputs_template)), desc='Processing'):    
    input = inputs_template[i]
    conv = conv_templates[args.conv_mode].copy()
    if "mpt" in model_name.lower():
        roles = ('user', 'assistant')
    else:
        roles = conv.roles
    # Q: If I do not re-initialize the conv, the program will throw out an error of inex out of range. Why and how to correct it?
    print('\n\n')
    image = load_image(input['image_dir'])
    image_size = image.size
    # Similar operation in model_worker.py
    image_tensor = process_images([image], image_processor, model.config)
    if type(image_tensor) is list:
        image_tensor = [image.to(model.device, dtype=torch.float16) for image in image_tensor]
    else:
        image_tensor = image_tensor.to(model.device, dtype=torch.float16)

    inp = input['prompt']

    if image is not None:
        # first message
        if model.config.mm_use_im_start_end: # False
            inp = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + inp
        else:
            inp = DEFAULT_IMAGE_TOKEN + '\n' + inp
        image = None

    conv.append_message(conv.roles[0], inp)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()

    input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).to(model.device)
    # stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
    # keywords = [stop_str]
    streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

    with torch.inference_mode():
        model
        output_ids = model.generate(
            input_ids,
            images=image_tensor,
            image_sizes=[image_size],
            do_sample=True if args.temperature > 0 else False,
            temperature=args.temperature,
            max_new_tokens=args.max_new_tokens,
            streamer=streamer,
            # use_cache=True)
            use_cache=False)

    outputs = tokenizer.decode(output_ids[0], skip_special_tokens = True).strip()
    conv.messages[-1][-1] = outputs
    responds.append(outputs)
    # if args.debug:
    #     print("\n", {"prompt": prompt, "outputs": outputs}, "\n")


    import json 

logs = []
for i in range(len(inputs_template)):
    logs.append(inputs_template[i])
    logs[-1]['respond'] = responds[i]
   
# Convert and write JSON object to file
with open("img_prompt_respond.json", "w") as outfile: 
    json.dump(logs, outfile, indent=4)
