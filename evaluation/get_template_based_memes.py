import argparse
import datasets

from memetils import load_dataset, str2bool
from tlc import *
from template_instance_pair_visualization import create_html
from LPIPS import lpips_filtering

parser = argparse.ArgumentParser(description='Template Label Counter')
parser.add_argument('--template_path', action='store', type=str, dest='path', default='../data/meme_retrieval_data/template_info.json')
parser.add_argument('--dataset', action="store", type=str, dest='dataset', default='figmemes')
parser.add_argument('--data_root', action="store", type=str, dest='data_root', default='../data/figmemes')
parser.add_argument('--num_neigh', action="store", type=int, dest='num_neigh', default=1)
parser.add_argument('--vote_type', action="store", type=str, dest='vote_type', default='template')
parser.add_argument('--split', action="store", type=str, dest='split', default='standard')
parser.add_argument('--all_feature_type', action="store", type=str, dest='all_feature_type', default='')
parser.add_argument('--include_examples', action="store", type=str2bool, dest='examples', default=False)
# Jupyter has this command parameter
parser.add_argument('--feature_extraction', action="store", type=str, dest='feature', default='pixel') 
parser.add_argument('--meme_size', action="store", type=int, dest='meme_size', default=64) 
parser.add_argument('--task', action="store", type=int, dest='task', default=1)
# No combination, only meme embeddings.
# 'fancy', 'fusion', 'concatenate', 'ablation', 'None'
parser.add_argument('--combine', action="store", type=str, dest='combine', default='concatenate') 
parser.add_argument('--just_text', action="store", type=str2bool, dest='just_text', default='False')
parser.add_argument('--need_to_read', action="store", type=bool, dest='need_to_read', default=False)
parser.add_argument('--save_embeddings', action="store", type=bool, dest='save_embeddings', default=True)
parser.add_argument('--text_features', action="store", default='img_captions') # 'ocr_text', 'img_captions'

args = parser.parse_args()
dataset = load_dataset(args)
dataset[1]['train'] = datasets.Dataset.from_dict({'img_path': [dataset[1]['train'][0]['img_path']], 
                                                  'ocr_text': [dataset[1]['train'][0]['ocr_text']], 
                                                  'img_captions': [dataset[1]['train'][0]['img_captions']]})
dataset[1]['validation'] = datasets.Dataset.from_dict({'img_path': [dataset[1]['validation'][0]['img_path']], 
                                                  'ocr_text': [dataset[1]['validation'][0]['ocr_text']],
                                                  'img_captions': [dataset[1]['validation'][0]['img_captions']]})

print(dataset[1])

# for ite in dataset[1]['test']['img_captions']:
#     print(ite)

print("dataset[1]['test'][0]", dataset[1]['test'][:3])


args.feature = 'pixel'
if not args.need_to_read:
    model, preprocess = clip.load('../model/ViT-L-14-336px.pt')
    model.cuda().eval()
    # print(model)
    print(preprocess)
    # The __init__ function will calculate the embeddings 
    # for templates and the target dataset, if 'need_to_read' is 'True'
    tlc = TemplateLabelCounter(args=args, dataset=dataset, model=model, 
                               preprocess=preprocess, need_to_read=False)
else:
    tlc = TemplateLabelCounter(args=args, dataset=dataset, need_to_read=True)



# tlc.get_template_embeddings()
# tlc.get_meme_embeddings()
output_file_name = "meme2template.json"
meme2template_json = tlc.meme2template(max_distance = 30, output_file_name=output_file_name)
lpips_filtered_meme_paris = lpips_filtering(meme2template_json,
                                            output_filename_suffix=output_file_name,
                                            max_distance=0.29)
create_html(lpips_filtered_meme_paris)

