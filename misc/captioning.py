import torch
from PIL import Image
from utils.geo_utils import *
from transformers import BlipProcessor, Blip2ForConditionalGeneration
from einops import rearrange
import torchvision
import shutil
import os
import pickle
from tqdm import tqdm
import argparse



parser = argparse.ArgumentParser()

parser.add_argument("--image_dir",
                    type=str,
                    default=None,
                    required=True)

parser.add_argument("--out_dir",
                    type=str,
                    default=None,
                    required=True)

parser.add_argument("--name",
                    type=str,
                    default="caption",
                    required=False)

parser.add_argument("--stride",
                    type=int,
                    default=30,)

parser.add_argument("--cache_dir",
                    type=str,
                    default=None,)
 
parser.add_argument("--device",
                    type=str,
                    default="cuda:0")

parser.add_argument("--augment", 
                    action="store_true")

args = parser.parse_args()
torch.backends.cudnn.enabled = False
cache_dir = args.cache_dir
blip_processor = None
blip_model = None
device = args.device

@torch.no_grad()
def get_labels(images):
    # image = invTrans(image)
    global blip_processor
    global blip_model

    if blip_processor is None:
        blip_processor = BlipProcessor.from_pretrained("Salesforce/blip2-flan-t5-xxl", cache_dir=cache_dir)
    if blip_model is None:
        blip_model = Blip2ForConditionalGeneration.from_pretrained("Salesforce/blip2-flan-t5-xxl",  cache_dir=cache_dir, torch_dtype=torch.float16, device_map="auto", load_in_8bit=True, )

    captions = list()
    if isinstance(images, torch.Tensor):
        if len(images.shape) == 3:
            images = rearrange(images, 'c h w -> 1 c h w')

        pil_images = list()
        for image in images:
            # image in [-1, 1]
            image = (image.numpy() + 1) / 2
            image = (image * 255).astype(np.uint8)
            image = np.transpose(image, (1, 2, 0))
            image = Image.fromarray(image)
            pil_images.append(image)
        images = pil_images

    for image in images:
        # inputs = blip_processor(image,  return_tensors="pt")
        inputs = blip_processor(image,  return_tensors="pt").to("cuda", torch.float16)
        outputs = blip_model.generate(**inputs)
        caption = blip_processor.decode(outputs[0], skip_special_tokens=True) # logits = outputs.logits predicted_class_idx = logits.argmax(-1).item()
        # return model.config.id2label[predicted_class_idx]
        captions.append(caption)
    return captions




def extract_captions():

    image_dir = args.image_dir
    dest = args.out_dir
    dataset_name = args.name 
    stride = args.stride
    
    invalid_images = list()
    captions_dict = dict()
    for filename in tqdm(os.listdir(image_dir)):
        image_path = os.path.join(image_dir, filename)
        try:
            image = Image.open(image_path)
            captions = get_labels([image])
            captions_dict[filename] = captions
        except:
            invalid_images.append(filename)
            print(f'adding f{filename} to invalid image')
            continue


    with open(os.path.join(dest, dataset_name + '.pkl'), 'wb') as f:
        pickle.dump(captions_dict, f) 

    for filename in invalid_images:
        shutil.move(os.path.join(image_dir, filename), os.path.join("/hdd/invalid", filename))


    if args.augment:
        print('extracting augmenting caption')
        aug_captions_dict = dict() 
        for filename in tqdm(os.listdir(image_dir)):
            image_path = os.path.join(image_dir, filename)
            image = Image.open(image_path) 
            image = torchvision.transforms.functional.pil_to_tensor(image) 
            nfovs = list() 

            # extract augmented caption along the x axis
            for i in range(360//stride):
                nfov, _ = get_nfov(image, i, 0,)
                nfov = torchvision.transforms.functional.to_pil_image(nfov)
                nfovs.append(nfov)
            captions = get_labels(nfovs) 
            aug_captions_dict[filename] = captions 



        with open(os.path.join(dest, dataset_name + '_aug.pkl'), 'wb') as f:
            pickle.dump(aug_captions_dict, f) 



if __name__ == '__main__':
    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)

    extract_captions()
    
       




