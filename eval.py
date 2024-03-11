
import torch
import equilib 
from PIL import Image
import os
from pathlib import Path
import numpy as np
import shutil
from rich.progress import track
from cleanfid import fid
from einops import rearrange
from torchmetrics.image.inception import InceptionScore
# from torchmetrics.image import StructuralSimilarityIndexMeasure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from torchmetrics.multimodal import CLIPScore
# from ignite.metrics import InceptionScore
import pickle
from misc.captioning import get_labels

from sentence_transformers import SentenceTransformer, util

# from ignite.metrics import InceptionScore
def compute_sentence_similarity(sen1, sen2):
    # model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    model = SentenceTransformer('all-mpnet-base-v2')
    scores = dict()
    sum = 0.
    for s1, s2 in zip(sen1, sen2):
        embedding_1= model.encode(s1, convert_to_tensor=True)
        embedding_2 = model.encode(s2, convert_to_tensor=True)
        score = util.pytorch_cos_sim(embedding_1, embedding_2)
        
        print(score)
        # scores[s1] = score.detach().cpu().squeeze().numpy()
        sum += score 

    sim = sum / min(len(sen1), len(sen2))
    return sim, scores


    



def read_equi(image_path, ):

    equi_img = Image.open(image_path)#  resize((224, 224))
    equi_img = np.asarray(equi_img)
    equi_img = np.transpose(equi_img, (2, 0, 1))

    return equi_img

def to_cubemaps(image_path):

    equi_img = Image.open(image_path)
    equi_img = np.asarray(equi_img)
    equi_img = np.transpose(equi_img, (2, 0, 1))

    rots = {
            'roll': 0.,
            'pitch': 0.,  # rotate vertical
            'yaw': 0.,  # rotate horizontal
        }
    cubes = equilib.equi2cube(equi_img, w_face=299, cube_format='dict', rots=rots)
    faces = list()
    for k, v in cubes.items():
        if k == 'U' or k == 'D':
            continue 
        v = np.transpose(v, (1, 2, 0))
        v = Image.fromarray(v)
        faces.append(v)

    return faces 






if __name__ == '__main__':

    src = ''
    caption_info = ''
    dest = './fid_folder/'




    # shutil.rmtree(dest)
    dest_path = Path(dest)
    if dest_path.exists():
        # os.rmdir(dest)
        shutil.rmtree(dest)

    load_images = True
    os.makedirs(dest, exist_ok=True) 
    counter = 0
    # gen_dir 
    gen_dir = os.path.join(dest, 'gen')
    gt_dir = os.path.join(dest, 'gt')

    
     
    
    with open(caption_info, 'rb') as f:
        captions = pickle.load(f)
        print(f"loading {len(captions)} captions")
    

    equis = list() 
    texts = list()
    gen_caption = list() 
    os.makedirs(gen_dir, exist_ok=True)
    os.makedirs(gt_dir, exist_ok=True)

    for entry in track(os.listdir(src)):
        gen = os.path.join(src, entry, 'result.png')
        gt = os.path.join(src, entry, 'gt.png')

        filename = None
        try:
            print(src, entry)
            with open(os.path.join(src, entry, 'path.text'), 'r') as f:
                filename = f.readline() 
            filename = filename.split('/')[-1]
        except:
            raise

        if filename is not None:
            texts.append(captions[filename])

        equis.append(read_equi(gen,))

        equi = read_equi(gen)
        equi = equilib.equi2equi(equi, rots={'roll':0, 'pitch':100, 'yaw':0})
        equi = rearrange(equi, 'c h w -> h w c')
        equi = Image.fromarray(equi)
        gen_caption.append(get_labels([equi]))
        

        # we divide gt and result to cubemap for evaluation
        if load_images:
            gen = os.path.join(src, entry, 'result.png')
            gt = os.path.join(src, entry, 'gt.png')
            gen_faces = to_cubemaps(gen) 
            gt_faces = to_cubemaps(gt) 
            for gen_face, gt_face in zip(gen_faces, gt_faces):
                gen_face.save(os.path.join(gen_dir, f"{str(counter)}.png"))
                gt_face.save(os.path.join(gt_dir, f"{str(counter)}.png"))
                counter += 1

    # sen_sim = 0
    if len(texts) == len(gen_caption):
       sen_sim,  scores = compute_sentence_similarity(texts, gen_caption)
       print("SS: ",scores, sen_sim)
        



    # fid score
    fid_score = fid.compute_fid(gen_dir, gt_dir,  mode="clean", model_name="clip_vit_b_32")
    print('fid : ',fid_score)
    # score = fid.compute_fid(gen_dir, gt_dir,  dataset_res=299)
    # print(score)


    clip_score = list() 
    metric = CLIPScore(model_name_or_path="openai/clip-vit-base-patch16")
    for img, txt in zip(equis, texts):
        score = metric(torch.from_numpy(img), txt)
        clip_score.append(score)


    # compute IS 
    inception = InceptionScore(split=10)
    imgs = list() 
    target_dir = gen_dir
    for image_path in os.listdir(gt_dir):
        img = Image.open(os.path.join(target_dir, image_path))
        img = np.asarray(img)
        img = np.transpose(img, (2, 0, 1))
        img = torch.from_numpy(img)
        imgs.append(img)
        # img = rearrange(img, 'c h w -> b c h w')
    imgs = torch.stack(imgs, dim=0)
    imgs1 = imgs
    inception.update(imgs)
    print("IS: " , inception.compute())

    imgs2 = list()
    target_dir = gt_dir
    for image_path in os.listdir(target_dir):
        img = Image.open(os.path.join(target_dir, image_path))
        img = np.asarray(img)
        img = np.transpose(img, (2, 0, 1))
        img = torch.from_numpy(img)
        imgs2.append(img)
        # img = rearrange(img, 'c h w -> b c h w')
    imgs2 = torch.stack(imgs2, dim=0)

    imgs1 = (imgs1.to(torch.float32) / 255 ) * 2 - 1
    imgs2 = (imgs2.to(torch.float32) / 255 ) * 2 - 1



    mean_lpips = 0.
    lpips_list = list()
    lpips = LearnedPerceptualImagePatchSimilarity(net_type='squeeze')
    for i1, i2 in zip(imgs1, imgs2):
        lpips_score = lpips(i1.unsqueeze(dim=0), i2.unsqueeze(dim=0)).detach().cpu()
        # mean_lpips += lpips(i1.unsqueeze(dim=0), i2.unsqueeze(dim=0)).detach().cpu()
        lpips_list.append(lpips_score)


    for l in lpips_list:
        mean_lpips += l 

    mean_lpips /= len(lpips_list) 

    lpips_np = np.asarray(lpips_list) 
    print("mean lpips:" , mean_lpips)
    # np.save('./lpips', lpips_np)






    # 
    


    
