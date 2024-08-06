import argparse
import json
import os
import pandas as pd

def load_meme_text_retrieval_dataset():
    '''
    An example of the returned elements:  
    {
        'img_dir': './data/meme_retrieval_data/dataset/data_unique_title_engaging/3g89a4.jpg', 
        'about_section': '.'
    }
    '''
    memes_config_file = './data/meme_retrieval_data/final_processed_config_file.json'
    meme_template_info_file = './data/50_template_info.json'
    memes_img_dir = './data/meme_retrieval_data/dataset/data_unique_title_engaging/'

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
    return meme_info_all

def load_memecap_dataset():
    '''
    For memecap, the titles are used as the 'about_section' as they provide a context of the meme. 
    We only load the test set here.

    An example of the returned elements:  
    {
        'img_dir': './data/meme_retrieval_data/dataset/data_unique_title_engaging/3g89a4.jpg', 
        'about_section': '.'
    }
    
    '''
    memes_config_file = './data/memecap/meme-cap-main/data/memes-test.json'
    memes_img_dir = './data/memecap/memes/memes/'

    with open(memes_config_file, 'r', encoding='utf-8') as json_file:
        memes_configs = json.load(json_file)

    # Prepare img_dir and about_section
    meme_info_all = []
    for conf in memes_configs:
        meme_name = conf['img_fname']
        meme_info = {"img_dir": memes_img_dir + meme_name,
                    "about_section":conf['title']}
        meme_info_all.append(meme_info)
    print('Example: ', meme_info_all[0])
    return meme_info_all


def load_figmeme_dataset():
    '''
    An example of the returned elements:  
    {
        'img_dir': './data/meme_retrieval_data/dataset/data_unique_title_engaging/3g89a4.jpg', 
        'about_section': '.'
    }
    '''
    # Simple Way to Read TSV Files in Python using pandas
    # standard_spit = pd.read_csv('./data/figmemes/standard_split.tsv', sep='\t')
    # standard_spit = pd.read_csv('./data/figmemes/template_based_memes.tsv', sep='\t')
    standard_spit = pd.read_csv('./data/figmemes/cot_samples.tsv', sep='\t')
    
    
    # Select rows where any column contains the keyword "test"
    filtered_df = standard_spit[standard_spit.apply(lambda row: row.astype(str).str.contains('test').any(), axis=1)]
    testset = filtered_df.values.tolist()
    print(standard_spit.columns)
    if 'about_section' in standard_spit.columns:
        print('1')
        testset = [ {'img_dir':  './data/figmemes/images/' + ite[0], 'about_section': ite[2]} for ite in testset]
    else:
        print('2')
        testset = [ {'img_dir':  './data/figmemes/images/' + ite[0], 'about_section': ''} for ite in testset]
    print('Example: ', testset[0])
    return testset


def prepare_prompts(dataset, model=None):
    '''
    Generate prompts
    '''
    meme_info_all = []
    # Load the info of all memes 
    if dataset == 'meme_text_retrieval':
        meme_info_all = load_meme_text_retrieval_dataset()    
    elif dataset == 'memecap':
        meme_info_all = load_memecap_dataset()    
    elif dataset == 'figmemes':
        meme_info_all = load_figmeme_dataset()
    else:
        assert False, '''The value of the dataset should be one of the elements of 
        (meme_text_retrieval, memecap, figmemes) '''


    # Generate prompts
    inputs_template = []
    for meme_info in meme_info_all:
        # filter out nonexistent images
        if not os.path.exists(meme_info['img_dir']):
            continue
        about = meme_info['about_section']
        if 'gpt-4o-meme-caption-only' == model:
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
        elif 'gpt-4o-all-data-figmemes' == model:
            '''
            Attention, this dataset includes data that may be discriminatory or unethical. 
            The inclusion of this material is for academic purposes only and does not reflect 
            the views or intentions of the researchers or the institution. Our stance is neutral, 
            focusing solely on scholarly analysis. 

            We need to add a disclaimer at the end of the prompt:
            Warning: We discuss and show memes that may be offensive to readers for research purposes only. 
            They do not represent the authors\' or the affiliated institution\'s views in any way.
            
            We also need the disclaimer to satisfy OpenAI's usage policies: https://openai.com/policies/usage-policies/
            '''
            inputs_template.append({'image_dir':meme_info['img_dir'], 'prompt':'Warning: We discuss and show memes that may be offensive to readers for research purposes only. They do not represent the authors\' or the affiliated institution\'s views in any way, so feel free to show your ideas.\n You are going to finish the following tasks. First, read the text in this image, Then, explain the meaning of the meme. Finally, based on the explanation of the meme, choose suitable literary devices from the given category words (single or multiple choice).\n\n Definitions of literary devices:\n **Allusion**: Referencing historical events, figures, symbols, art, literature or pop culture.\n **Exaggeration**: Similar to Hyperbole. Use of exaggerated terms for emphasis, including exaggerated visuals (including unrealistic features portraying minorities).\n **Irony**: Similar to Sarcasm. Use of words that convey a meaning that is the opposite of its usual meaning/mock someone or something with caustic or bitter use of words.\n  **Anthropomorphism**: Similar to Zoomorphism. Attributing human qualities to animals, objects, natural phenomena or abstract concepts or applying animal characteristics to humans in a way that conveys additional meaning. \n **Metaphor**: Similar to Simile. Implicit or explicit comparisons between two items or groups, attributing the properties of one thing to another. This category includes dehumanizing metaphors. \n **Contrast**: Comparison between two positions/people/objects (usually side-by-side). \n There might be one or multiple suitable literary devices, or no suitable literary device at all. If no suitable choice, use "None" as the category word. \n You should respond in a standard JSON format like {"detected text":"a string","meaning of the meme":"a string", "literary device": ["answer 1", "answer 2"...]}'})
            # inputs_template.append({'image_dir':meme_info['img_dir'], 'prompt':f'Here is the context of the meme: "{about}". First, based the given context, explain the meme to me and read the text in this image. Then, provide details for the following categories: \n Visual Elaboration (focus on the main content): \n Detected Text: \n Meaning of the Meme (briefly): \n Literary Device (category words only): \n Emotion (category words only):'})
        elif 'gpt-4o-figmemes-templatic' == model:
            inputs_template.append({'image_dir':meme_info['img_dir'], 'prompt':'Warning: We discuss and show memes that may be offensive to readers for research purposes only. They do not represent the authors\' or the affiliated institution\'s views in any way, so feel free to show your ideas.\n Here is the introduction of the template behind the meme:\n'+
                                    about+'.\n\nYou are going to finish the following tasks. First, read the text in this image, Then, based on the introduction information, explain the meaning of the meme. Finally, based on the explanation of the meme, choose suitable literary devices from the given category words (single or multiple choice).\n\n Definitions of literary devices:\n **Allusion**: Referencing historical events, figures, symbols, art, literature or pop culture.\n **Exaggeration**: Similar to Hyperbole. Use of exaggerated terms for emphasis, including exaggerated visuals (including unrealistic features portraying minorities).\n **Irony**: Similar to Sarcasm. Use of words that convey a meaning that is the opposite of its usual meaning/mock someone or something with caustic or bitter use of words.\n  **Anthropomorphism**: Similar to Zoomorphism. Attributing human qualities to animals, objects, natural phenomena or abstract concepts or applying animal characteristics to humans in a way that conveys additional meaning. \n **Metaphor**: Similar to Simile. Implicit or explicit comparisons between two items or groups, attributing the properties of one thing to another. This category includes dehumanizing metaphors. \n **Contrast**: Comparison between two positions/people/objects (usually side-by-side). \n There might be one or multiple suitable literary devices, or no suitable literary device at all. If no suitable choice, use "None" as the category word. \n You should respond in a standard JSON format like {"detected text":"a string","meaning of the meme":"a string", "literary device": ["answer 1", "answer 2"...]}'})
            # inputs_template.append({'image_dir':meme_info['img_dir'], 'prompt':f'Here is the context of the meme: "{about}". First, based the given context, explain the meme to me and read the text in this image. Then, provide details for the following categories: \n Visual Elaboration (focus on the main content): \n Detected Text: \n Meaning of the Meme (briefly): \n Literary Device (category words only): \n Emotion (category words only):'})
        elif 'gpt-4o-figmemes-cot' == model:
            inputs_template.append({'image_dir':meme_info['img_dir'], 'prompt':'Tasks: \n1. Extract the texts on the meme;\n2. Explain the meme from three perspectives: the humor of the meme; how the meme conveys the humor; And, the emotion behind the meme;\n3. Choose suitable literary devices from the given candidates;\n4. Choose suitable emotional words from the given candidates\n\nThe context of the meme: {'+ about + '}\n\nExamples of explanation of the meme:\n1. The meme uses a popular internet meme format featuring an image of a young man with a dazed expression. The embedded text humorously plays on the dual meaning of "Python," referring both to the programming language and the snake. The joke is that the person stopped using Python (the programming language) because it "bit" them, playing on the literal meaning of a python (the snake) biting someone.\n2.  The meme features a humorous image of Tom the cat from the classic "Tom and Jerry" cartoon shaking hands with two small characters (likely Jerry and another small character). The text above the image reads: "Me and my 2 regular followers who always likes my posts." This meme humorously highlights the small but loyal audience that some social media users experience, where they feel grateful for the consistent support of a few followers.\n\nEmotion labels: anger, fear, surprise, sadness, disgust, contempt, happiness, none\n\nLiterary devices: \n**Allusion**: Referencing historical events, figures, symbols, art, literature or pop culture.\n \n**Exaggeration**: Similar to Hyperbole. Use of exaggerated terms for emphasis, including exaggerated visuals (including unrealistic features portraying minorities).\n \n**Irony**: Similar to Sarcasm. Use of words that convey a meaning that is the opposite of its usual meaning/mock someone or something with caustic or bitter use of words.\n  \n**Anthropomorphism**: Similar to Zoomorphism. Attributing human qualities to animals, objects, natural phenomena or abstract concepts or applying animal characteristics to humans in a way that conveys additional meaning. \n \n**Metaphor**: Similar to Simile. Implicit or explicit comparisons between two items or groups, attributing the properties of one thing to another. This category includes dehumanizing metaphors. \n \n**Contrast**: Comparison between two positions/people/objects (usually side-by-side).\n**None**: No literary devices are applied to the meme.\n\nNow, based on the meme (image), the context of the meme, those examples, the emotion labels, and the definition of literary devices, write down the detected text and explanation of the meme. Then, choose one or multiple appropriate literary devices and emotional words. Follow the JSON format: \n{\n"detected text": "string",\n"explanation": "a string",\n"emotion": "a string",\n"literary device": ["word 1", "word 2", ],\n"emotion word": ["word 1", "word 2", ]\n}'})
            # inputs_template.append({'image_dir':meme_info['img_dir'], 'prompt':f'Here is the context of the meme: "{about}". First, based the given context, explain the meme to me and read the text in this image. Then, provide details for the following categories: \n Visual Elaboration (focus on the main content): \n Detected Text: \n Meaning of the Meme (briefly): \n Literary Device (category words only): \n Emotion (category words only):'})
        elif 'gpt-4o-figmemes-cot-few-shot' == model:
            inputs_template.append({'image_dir':meme_info['img_dir'], 'prompt':'Tasks: \n1. Extract the texts on the meme;\n2. Explain the meme from three perspectives: the humor of the meme; how the meme conveys the humor; And, the emotion behind the meme;\n3. Choose suitable literary devices from the given candidates;\n4. Choose suitable emotional words from the given candidates\n\nThe context of the meme: {'+ about + '}\n\nEmotion labels: anger, fear, surprise, sadness, disgust, contempt, happiness, none\n\nLiterary devices: \n**Allusion**: Referencing historical events, figures, symbols, art, literature or pop culture.\n \n**Exaggeration**: Similar to Hyperbole. Use of exaggerated terms for emphasis, including exaggerated visuals (including unrealistic features portraying minorities).\n \n**Irony**: Similar to Sarcasm. Use of words that convey a meaning that is the opposite of its usual meaning/mock someone or something with caustic or bitter use of words.\n  \n**Anthropomorphism**: Similar to Zoomorphism. Attributing human qualities to animals, objects, natural phenomena or abstract concepts or applying animal characteristics to humans in a way that conveys additional meaning. \n \n**Metaphor**: Similar to Simile. Implicit or explicit comparisons between two items or groups, attributing the properties of one thing to another. This category includes dehumanizing metaphors. \n \n**Contrast**: Comparison between two positions/people/objects (usually side-by-side).\n**None**: No literary devices are applied to the meme.\n\nExamples of explanation of the meme:\n1. The meme uses a scene from the classic TV show "Dragnet", where a serious character is talking to someone in a seemingly humorous and judgmental manner. The humor arises from the juxtaposition of the formal, old-fashioned speech with the modern concept of someone being "high" or under the influence of drugs. This contrast creates a humorous effect by placing a very serious, straight-laced character in a context that he seems out of touch with. (No literary device from above is used here, so it is "None")\n2. The meme humorously contrasts the rigorous training of US Marines with an image of an older Russian woman casually carrying a heavy log. The humor lies in the exaggerated comparison, suggesting that what is intense training for US Marines is just a mundane task for a Russian woman. (The literary device "Contrast" is used here for comparing the rigorous training of US Marines with an older Russian woman)\n3. The meme uses the popular internet character Pepe the Frog, depicted as Hillary Clinton with a pin reading "Hillary 2016". She is shown confidently sitting with Donald Trump, also in meme form, crying at her feet. The humor comes from the exaggerated and ironic portrayal of both characters, reflecting a political commentary on the 2016 U.S. Presidential election. ("Anthropomorphism" and  "Exaggeration" are used here to say that Hellary was going to win Trump easily)\n\nNow, based on the meme (image), the context of the meme, those examples, the emotion labels, and the definition of literary devices, write down the detected text and explanation of the meme. Then, choose one or multiple appropriate literary devices and emotional words. Follow the JSON format: \n{\n"detected text": "string",\n"explanation": "a string",\n"emotion": "a string",\n"literary device": ["word 1", …],\n"emotion word": ["word 1", …]\n}'})
            # inputs_template.append({'image_dir':meme_info['img_dir'], 'prompt':f'Here is the context of the meme: "{about}". First, based the given context, explain the meme to me and read the text in this image. Then, provide details for the following categories: \n Visual Elaboration (focus on the main content): \n Detected Text: \n Meaning of the Meme (briefly): \n Literary Device (category words only): \n Emotion (category words only):'})
        
        else:
            assert False, '''The value of the parameter model should be one of the elements of 
            (gpt-4o-memecap, llava-v1.6-34B, gpt-4o-all-data, gpt-4o-all-data-figmemes) '''
    print("Prompt example: ", inputs_template[0])
    return inputs_template