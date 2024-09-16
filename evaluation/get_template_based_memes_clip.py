import argparse
import torch
import clip
from PIL import Image

from memetils import load_dataset, str2bool
from tlc import *
from template_instance_pair_visualization import create_html
from LPIPS import lpips_filtering

parser = argparse.ArgumentParser(description='Template Label Counter')
parser.add_argument('--template_path', action='store', type=str, dest='path', default='../data/meme_retrieval_data/template_info.json')
parser.add_argument('--dataset', action="store", type=str, dest='dataset', default='memecap')
parser.add_argument('--data_root', action="store", type=str, dest='data_root', default='../data/memecap')

args = parser.parse_args()
dataset = load_dataset(args)
dataset[1]

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-L/14", device=device)

images = []
template_info = []
with open(args.path, 'r') as f:
    for line in tqdm(f):
        temp_info = dict(json.loads(line))
        # self.info.append(temp_info)
        for template in temp_info.keys():
            template = temp_info[template]
            # file_path = template["out_paths"][0]
            file_path = '../data/meme_retrieval_data/templates/' + template["out_paths"][0].split('/')[-1]
            template["out_paths"][0] = file_path
            if os.path.exists(file_path):
                # This process is the pre-process of CLIP model.
                # im = preprocess(Image.open(file_path)).unsqueeze(0).to(device)
                template_info.append(temp_info)
                images.append(file_path)
            else:
                print('Error, file not exist:', file_path)

# images = torch.cat(images, dim=0)
text = clip.tokenize(dataset[1]['test']['ocr_text']).to(device)


logits_per_text = []

with torch.no_grad():
    # logits_per_text: text2image retrieval scores
    text_features = model.encode_text(text)
    # normalized features
    text_features = (text_features / text_features.norm(p=2, dim=-1, keepdim=True)).cpu()
    print('text_features.shape:', text_features.shape) 

    # Calculate the image embeddings one by one, due to the limited GPU memory.
    image_embeds = []
    for img in tqdm(images): 
        image_embedding = model.encode_image(preprocess(Image.open(img)).unsqueeze(0).to(device))
        # normalized features
        image_embeds.append(image_embedding.cpu())
        
    # Convert list of tensors to a single tensor
    image_embeds = torch.stack(image_embeds).squeeze(1)
    image_embeds = image_embeds / image_embeds.norm(p=2, dim=-1, keepdim=True)
    print("image_embeds.shape:", image_embeds.shape)

    # cosine similarity as logits
    # text2image retrieval similarities
    # logit_scale = model.logit_scale.exp().cpu().detach().numpy()
    logits_per_text = torch.matmul(text_features.type(torch.float32), image_embeds.t().type(torch.float32))
    

# image indexes
m_shape = logits_per_text.shape
# top 1 image indexes
similarity, img_idx = logits_per_text.topk(1, dim=1)

# Find the nearest template and save the results in a JSON file
# template_dict = dict()
meme2template_json = []
counter = 0
for idx, idx_row in tqdm(enumerate(img_idx)):
    template_names = [template_info[idx] for idx in idx_row]
    template_file_names = []
    
    # By constraining the distance to filter out those meme that are similar to templates
    # minima similarity here
    if similarity[idx] < 0.1:
        continue
    about = ''
    for template_idx in idx_row:
        temp_info = template_info[template_idx]
        for template in temp_info.keys():
            template = temp_info[template]
            template_file_names.append(template["out_paths"][0])
            about = template["original_info"][0]["about"]
            break
    counter += 1
    meme2template_json.append(dict({'meme_name':dataset[1]['test']['img_path'][idx], 
                                    'similarity':similarity[idx].tolist(), 
                                    'templates':template_names,
                                    'template_file_name': template_file_names,
                                    'about_section': about}))

with open(f"meme2template.json", "w") as outfile: 
    json.dump(meme2template_json, outfile, indent=4)   
print(f"In total, there are {counter} instances.")

output_file_name = "meme2template.json" 
lpips_filtered_meme_paris = lpips_filtering(meme2template_json,
                                            output_filename_suffix=output_file_name)
create_html(lpips_filtered_meme_paris)