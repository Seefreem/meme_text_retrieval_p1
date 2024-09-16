import inspect
import json
from tqdm import tqdm
import lpips
import argparse
import torch
import PIL
from PIL import Image
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor
# import cv2
from template_instance_pair_visualization import create_html

BICUBIC = Image.BICUBIC


def convert_image_to_rgb(image):
    return image.convert("RGB")

def transform(n_px):
    return Compose([
        Resize(n_px, interpolation=BICUBIC),
        CenterCrop(n_px),
        convert_image_to_rgb,
        ToTensor()
    ])

def lpips_filtering(test_memes_configs, output_filename_suffix, max_distance=0.3):
    print(inspect.currentframe().f_code.co_name)
    instances = []
    templates = []

    for ite in tqdm(test_memes_configs):
        template_file_name = ite["template_file_name"][0]
        meme_name = ite["meme_name"]
        # Transform the image from [h, w, c] to [c, h, w]
        # image = np.array(PIL.Image.open(template_file_name))
        # transformed_image = pre_process_img(image)
        preprocessor = transform(64)
        transformed_image =  preprocessor(PIL.Image.open(template_file_name))
        transformed_image = torch.unsqueeze(transformed_image, dim=0)
        # print(transformed_image.shape)
        templates.append(transformed_image)

        transformed_image =  preprocessor(PIL.Image.open(meme_name))
        transformed_image = torch.unsqueeze(transformed_image, dim=0)
        # print(transformed_image.shape)
        instances.append(transformed_image)

    instances = torch.concat(instances, dim=0)
    print(instances.shape)
    templates = torch.concat(templates, dim=0)
    print(templates.shape)

    # Load lpips pipeline and calculate distances
    loss_fn_alex = lpips.LPIPS(net='alex') # best forward scores
    # loss_fn_vgg = lpips.LPIPS(net='vgg') # closer to "traditional" perceptual loss, when used for optimization

    d = loss_fn_alex(instances, templates)
    # print('The distance between the two images is', d.detach().squeeze())

    # Filter memes according to distances
    lpips_filtered_meme_paris = []
    lpips_filtered_dis = []
    for idx, dis in enumerate(d):
        if dis < max_distance:
            lpips_filtered_meme_paris. append(test_memes_configs[idx])
            lpips_filtered_dis.append(dis.detach().squeeze().item())

    print('The distance between the image pairs are', lpips_filtered_dis)
    print(f'After lpips filtering, there are {len(lpips_filtered_meme_paris)} paris of memes')
    with open(f"lpips_filtered_" + output_filename_suffix, "w") as outfile: 
        json.dump(lpips_filtered_meme_paris, outfile, indent=4)  
    return lpips_filtered_meme_paris

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='lpips filtering')
    parser.add_argument('--filename', action='store', type=str, dest='filename', default='meme2template.json')
    args = parser.parse_args()

    test_memes_configs = []
    with open(args.filename, 'r', encoding='utf-8') as json_file:
            test_memes_configs = json.load(json_file)

    lpips_filtered_meme_paris = lpips_filtering(test_memes_configs, args.filename)
    create_html(lpips_filtered_meme_paris)
