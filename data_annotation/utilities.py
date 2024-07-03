import argparse
import json
import os

def prepare_prompts(memes_config_file, 
                    meme_template_info_file, 
                    memes_img_dir,
                    model=None):
    '''
    Generate prompts
    '''
    # Load the info of all memes 
    with open(memes_config_file, 'r', encoding='utf-8') as json_file:
        final_processed_config_file = json.load(json_file)

    # Load about info
    with open(meme_template_info_file, 'r', encoding='utf-8') as json_file:
        template_info_50 = json.load(json_file)

    # Prepare img_dir and about_section
    meme_info_all = []
    for i in final_processed_config_file:
        meme_name = i['url'].split("/")[-1].split(".json")[0]
        meme_type = i['type']
        meme_info = {"img_dir": memes_img_dir + meme_name,
                    "about_section":template_info_50[i['type']]['about']}
        meme_info_all.append(meme_info)
    print('Example: ', meme_info_all[0])

    # Generate prompts
    inputs_template = []
    for meme_info in meme_info_all:
        # filter out nonexistent images
        if not os.path.exists(meme_info['img_dir']):
            continue
        about = meme_info['about_section']
        if 'gpt-4o-memecap' == model:
            # Here reading the text and explaining the meme provides a context for 
            # generating meme caption (i.e. the meaning of the meme).
            inputs_template.append({'image_dir':meme_info['img_dir'], 'prompt':f'Here is the context of the meme: "{about}". First, read the text in this image. Then, based the given context, explain the meme. Finally, provide the meaning of the meme, with the format: "Meaning: [xxx]":'})
        elif 'llava-v1.6-34B' == model:
            inputs_template.append({'image_dir':meme_info['img_dir'], 'prompt':f'Here is the context of the meme: "{about}". First, based the given context, read the text in this image and explain the meme. Then, provide information for the following categories: \n Visual Elaboration (focus on the main content): \n Detected Text: \n Meaning of the Meme (briefly):'})
            inputs_template.append({'image_dir':meme_info['img_dir'], 'prompt':f'Here is the context of the meme: "{about}". First, based the given context, read the text in this image and explain the meme. Then, choose the most suitable literary device from the given category words: sarcasm, allegory, alliteration, allusion, amplification, anagram, analogy, anthropomorphism, antithesis, chiasmus, circumlocution, euphemism, hyperbole, imagery, metaphor, onomatopoeia, oxymoron, paradox, personification, portmanteau, pun, satire, simile, and symbolism. If no suitable word, use "None" as the category word. Only reply with the chosen word.'})
            inputs_template.append({'image_dir':meme_info['img_dir'], 'prompt':f'Here is the context of the meme: "{about}". First, based the given context, explain the meme. Then, choose the most suitable emotion word from the given category words: fear, anger, joy, sadness, surprise, disgust, guilt, contempt, shame, embarrassment, envy, jealousy, love, hate, and interest. If no suitable word, use "None" as the category word.  Only reply with the chosen word.'})
        elif 'gpt-4o-all-data' == model:
            inputs_template.append({'image_dir':meme_info['img_dir'], 'prompt':f'Here is the context of the meme: "{about}". First, based the given context, read the text in this image and explain the meme. Then, provide information for the following categories: \n Visual Elaboration (focus on the main content): \n Detected Text: \n Meaning of the Meme (briefly): \n Then, choose the most suitable literary device from the given category words: sarcasm, allegory, alliteration, allusion, amplification, anagram, analogy, anthropomorphism, antithesis, chiasmus, circumlocution, euphemism, hyperbole, imagery, metaphor, onomatopoeia, oxymoron, paradox, personification, portmanteau, pun, satire, simile, and symbolism. If no suitable word, use "None" as the category word. Only reply with the chosen word. \n Finally, choose the most suitable emotion word from the given category words: fear, anger, joy, sadness, surprise, disgust, guilt, contempt, shame, embarrassment, envy, jealousy, love, hate, and interest. If no suitable word, use "None" as the category word. Only reply with the chosen word.'})
            # inputs_template.append({'image_dir':meme_info['img_dir'], 'prompt':f'Here is the context of the meme: "{about}". First, based the given context, explain the meme to me and read the text in this image. Then, provide details for the following categories: \n Visual Elaboration (focus on the main content): \n Detected Text: \n Meaning of the Meme (briefly): \n Literary Device (category words only): \n Emotion (category words only):'})
        else:
            assert False, "The value of the parameter model should be one of the element of (gpt-4o-memecap, llava-v1.6-34B, gpt-4o-all-data) "
    return inputs_template