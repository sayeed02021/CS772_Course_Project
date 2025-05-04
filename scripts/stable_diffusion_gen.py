import torch
from diffusers import StableDiffusionPipeline, DDIMScheduler
from resent_model import ResNet_Model
import torch.nn.functional as F

import numpy as np
import os 
import glob
from PIL import Image


# Load stable diffusion model first
model_id = "CompVis/stable-diffusion-v1-4"
if torch.cuda.is_available():
    device = torch.device('cuda')
elif torch.backends.mps.is_available():
    device = torch.device('mps')
else:
    device = torch.device('mps')
generator = torch.Generator(device=device).manual_seed(42)
pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float32)
pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
pipe.scheduler.eta = 0.0
pipe = pipe.to(device)
tokenizer = pipe.tokenizer
text_encoder = pipe.text_encoder
## Args ##
folder_path = '/kaggle/input/gen-data'
emb_path = f'{folder_path}/Class_embeds_gen_OOD'

class_tokens = torch.tensor(np.load('/kaggle/input/class-tokens-c100/token_embed_c100.npy'))

class_labels = np.array([
    'apples',  # id 0
    'aquarium fish',
    'baby',
    'bear',
    'beaver',
    'bed',
    'bee',
    'beetle',
    'bicycle',
    'bottles',
    'bowls',
    'boy',
    'bridge',
    'bus',
    'butterfly',
    'camel',
    'cans',
    'castle',
    'caterpillar',
    'cattle',
    'chair',
    'chimpanzee',
    'clock',
    'cloud',
    'cockroach',
    'couch',
    'crab',
    'crocodile',
    'cups',
    'dinosaur',
    'dolphin',
    'elephant',
    'flatfish',
    'forest',
    'fox',
    'girl',
    'hamster',
    'house',
    'kangaroo',
    'computer keyboard',
    'lamp',
    'lawn-mower',
    'leopard',
    'lion',
    'lizard',
    'lobster',
    'man',
    'maple',
    'motorcycle',
    'mountain',
    'mouse',
    'mushrooms',
    'oak',
    'oranges',
    'orchids',
    'otter',
    'palm',
    'pears',
    'pickup truck',
    'pine',
    'plain',
    'plates',
    'poppies',
    'porcupine',
    'possum',
    'rabbit',
    'raccoon',
    'ray',
    'road',
    'rocket',
    'roses',
    'sea',
    'seal',
    'shark',
    'shrew',
    'skunk',
    'skyscraper',
    'snail',
    'snake',
    'spider',
    'squirrel',
    'streetcar',
    'sunflowers',
    'sweet peppers',
    'table',
    'tank',
    'telephone',
    'television',
    'tiger',
    'tractor',
    'train',
    'trout',
    'tulips',
    'turtle',
    'wardrobe',
    'whale',
    'willow',
    'wolf',
    'woman',
    'worm'])

labels_to_not_generate = ['aquarium fish',
'chimpanzee',
'cockroach',
'flatfish',
'computer keyboard',
'lawn-mower',
'pickup truck',
'porcupine',
'shrew',
'skyscraper',
'sweet peppers']


indices = [50, 100, 199]
for label in class_labels:
        
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()

    if label in labels_to_not_generate:
        continue
    
    prompt = f'{label}'

    text_input = tokenizer(
    prompt,
    padding="max_length",
    max_length=tokenizer.model_max_length,
    truncation=True,
    return_tensors="pt"
)
    embedding_layer = text_encoder.get_input_embeddings()
    original_weights = embedding_layer.weight.data.clone()

    input_ids = text_input.input_ids.to(device)
    tokens = tokenizer.convert_ids_to_tokens(input_ids[0])
    target_token = label+"</w>"
    target_idx = tokens.index(target_token)


    label_idx = np.where(class_labels==label)[0].item()
    generated_embeddings = torch.tensor(np.load(f'{emb_path}/Class_{label_idx}_train_embeds.npy')) # nx768
    # generated_embeddings = torch.tensor(gen_samples)

    
    # indices = np.arange(0,len(generated_embeddings))[-3:]
    # indices = [150]
    final_text_embeddings = torch.zeros(len(indices),77,768).to(device)
    for i,idx in enumerate(indices):
        custom_token = generated_embeddings[idx, :].unsqueeze(0)
        custom_token = F.normalize(custom_token)*class_tokens[label_idx,:].norm()
        
        
        target_token_id = input_ids[0, target_idx].item()
        embedding_layer.weight.data[target_token_id] = custom_token
        text_embeddings = text_encoder(input_ids)[0]
        final_text_embeddings[i,:,:] = text_embeddings[0,:,:]
        embedding_layer.weight.data = original_weights
    b = pipe(prompt_embeds=final_text_embeddings,
            height=512, width=512,
            num_inference_steps=50, 
             generator=generator).images
    
    save_path = f'Class_{label}_images'
    os.makedirs(save_path, exist_ok=True)
    for b_idx, img in enumerate(b):
        img.save(f'{save_path}/{b_idx}.png')
    print('Saved')
   