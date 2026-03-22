#!/usr/bin/env python
# coding=utf-8
# Copyright 2023 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
import os 
import numpy as np  
import torch 
import torch.nn.functional as F 
import torch.utils.checkpoint 
import transformers  
from PIL import Image  
from torch.utils.data import Dataset, DataLoader  
from torchvision import transforms  
from tqdm.auto import tqdm  
from transformers import AutoTokenizer, PretrainedConfig, ViTFeatureExtractor, ViTModel  

import lpips 
import json  
from PIL import Image
import requests  
from transformers import AutoProcessor, AutoTokenizer, CLIPModel  
import torchvision.transforms.functional as TF 
from torchvision.transforms import Compose, ToTensor, Normalize, Resize, ToPILImage 
from torch.nn.functional import cosine_similarity 
import re

class PromptDatasetCLIP(Dataset):
    def __init__(self, image_dir, tokenizer, processor):
        self.image_dir = image_dir

        self.image_lst = [os.path.join(self.image_dir, f) for f in os.listdir(self.image_dir) if f.endswith(".png")]
        self.prompt_lst = [os.path.splitext(f)[0].replace('_qwe', '') for f in os.listdir(self.image_dir) if f.endswith(".png")]
        
        self.tokenizer = tokenizer
        self.processor = processor
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def __len__(self):
        return len(self.image_lst)

    def __getitem__(self, idx):
        image_path = self.image_lst[idx]
        image = Image.open(image_path)
        prompt = self.prompt_lst[idx]

        extrema = image.getextrema()
        if all(min_val == max_val == 0 for min_val, max_val in extrema):
            return None, None
        else:
            prompt_inputs = self.tokenizer([prompt], padding=True, return_tensors="pt")
            image_inputs = self.processor(images=image, return_tensors="pt")

            return image_inputs, prompt_inputs


class PairwiseImageDatasetCLIP(Dataset):
    def __init__(self, subject, data_dir_A, data_dir_B, processor):
        self.data_dir_A = data_dir_A
        self.data_dir_B = data_dir_B
        
        self.data_dir_A = os.path.join(self.data_dir_A, subject) 
        self.image_files_A = [os.path.join(self.data_dir_A, f) for f in os.listdir(self.data_dir_A) if f.endswith(".jpg")] 
        self.image_files_B = [os.path.join(self.data_dir_B, f) for f in os.listdir(self.data_dir_B) if f.endswith(".png")]

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.processor = processor

    def __len__(self):
        return len(self.image_files_A) * len(self.image_files_B)

    def __getitem__(self, index):
        index_A = index // len(self.image_files_B)
        index_B = index % len(self.image_files_B)
        
        image_A = Image.open(self.image_files_A[index_A]) # .convert("RGB")
        image_B = Image.open(self.image_files_B[index_B]) # .convert("RGB")

        extrema_A = image_A.getextrema()
        extrema_B = image_B.getextrema()
        if all(min_val == max_val == 0 for min_val, max_val in extrema_A) or all(min_val == max_val == 0 for min_val, max_val in extrema_B):
            return None, None
        else:
            inputs_A = self.processor(images=image_A, return_tensors="pt")
            inputs_B = self.processor(images=image_B, return_tensors="pt")

            return inputs_A, inputs_B


class PairwiseImageDatasetDINO(Dataset):
    def __init__(self, subject, data_dir_A, data_dir_B, feature_extractor):
        self.data_dir_A = data_dir_A
        self.data_dir_B = data_dir_B
        
        self.data_dir_A = os.path.join(self.data_dir_A, subject) 
        self.image_files_A = [os.path.join(self.data_dir_A, f) for f in os.listdir(self.data_dir_A) if f.endswith(".jpg")] 
        self.image_files_B = [os.path.join(self.data_dir_B, f) for f in os.listdir(self.data_dir_B) if f.endswith(".png")]
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.feature_extractor = feature_extractor

    def __len__(self):
        return len(self.image_files_A) * len(self.image_files_B)

    def __getitem__(self, index):
        index_A = index // len(self.image_files_B)
        index_B = index % len(self.image_files_B)
        
        image_A = Image.open(self.image_files_A[index_A]) # .convert("RGB")
        image_B = Image.open(self.image_files_B[index_B]) # .convert("RGB")

        extrema_A = image_A.getextrema()
        extrema_B = image_B.getextrema()
        if all(min_val == max_val == 0 for min_val, max_val in extrema_A) or all(min_val == max_val == 0 for min_val, max_val in extrema_B):
            return None, None
        else:
            inputs_A = self.feature_extractor(images=image_A, return_tensors="pt")
            inputs_B = self.feature_extractor(images=image_B, return_tensors="pt")

            return inputs_A, inputs_B

class SelfPairwiseImageDatasetLPIPS(Dataset):
    def __init__(self, data_dir):
        self.data_dir_A = data_dir
        self.data_dir_B = data_dir

        self.image_files_A = [os.path.join(self.data_dir_A, f) for f in os.listdir(self.data_dir_A) if f.endswith(".png")]
        self.image_files_B = [os.path.join(self.data_dir_B, f) for f in os.listdir(self.data_dir_B) if f.endswith(".png")]

        self.transform = Compose([
            Resize((512, 512)),
            ToTensor(),
            Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def __len__(self):
        return len(self.image_files_A) * (len(self.image_files_B) - 1)

    def __getitem__(self, index):
        index_A = index // (len(self.image_files_B) - 1)
        index_B = index % (len(self.image_files_B) - 1)

        if index_B >= index_A:
            index_B += 1
        
        image_A = Image.open(self.image_files_A[index_A]) # .convert("RGB")
        image_B = Image.open(self.image_files_B[index_B]) # .convert("RGB")

        extrema_A = image_A.getextrema()
        extrema_B = image_B.getextrema()
        if all(min_val == max_val == 0 for min_val, max_val in extrema_A) or all(min_val == max_val == 0 for min_val, max_val in extrema_B):
            return None, None
        else:
            if self.transform:
                image_A = self.transform(image_A)
                image_B = self.transform(image_B)

            return image_A, image_B

def clip_text(image_dir):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to(device)
    tokenizer = AutoTokenizer.from_pretrained("openai/clip-vit-large-patch14")
    processor = AutoProcessor.from_pretrained("openai/clip-vit-large-patch14")
    
    dataset = PromptDatasetCLIP(image_dir, tokenizer, processor)

    clip_texts = []
    for i in tqdm(range(len(dataset))):
        image_inputs, prompt_inputs = dataset[i]
        if image_inputs is not None and prompt_inputs is not None:
            image_inputs['pixel_values'] = image_inputs['pixel_values'].to(device)
            prompt_inputs['input_ids'] = prompt_inputs['input_ids'].to(device)
            prompt_inputs['attention_mask'] = prompt_inputs['attention_mask'].to(device)

            image_features = model.get_image_features(**image_inputs)
            text_features = model.get_text_features(**prompt_inputs)

            sim = cosine_similarity(image_features, text_features)

            clip_texts.append(sim.item())

    clip_text_mean = torch.tensor(clip_texts).mean().item()
    print('clip-text mean: ', clip_text_mean)

    return clip_text_mean


def clip_image(subject, image_dir):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to(device)
    processor = AutoProcessor.from_pretrained("openai/clip-vit-large-patch14")

    clips = []
    dataset = PairwiseImageDatasetCLIP(subject, './data/dreambooth/dataset', image_dir, processor)

    for i in tqdm(range(len(dataset))):
        inputs_A, inputs_B = dataset[i]
        if inputs_A is not None and inputs_B is not None:
            inputs_A['pixel_values'] = inputs_A['pixel_values'].to(device)
            inputs_B['pixel_values'] = inputs_B['pixel_values'].to(device) 

            image_A_features = model.get_image_features(**inputs_A)
            image_B_features = model.get_image_features(**inputs_B)

            image_A_features = image_A_features / image_A_features.norm(p=2, dim=-1, keepdim=True)
            image_B_features = image_B_features / image_B_features.norm(p=2, dim=-1, keepdim=True)
        
            logit_scale = model.logit_scale.exp()
            sim = torch.matmul(image_A_features, image_B_features.t()) # * logit_scale
            clips.append(sim.item())

    clip_mean = torch.tensor(clips).mean().item()
    print('clip mean: ', clip_mean)

    return clip_mean


def dino(subject, image_dir):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ViTModel.from_pretrained('facebook/dino-vits16').to(device)
    feature_extractor = ViTFeatureExtractor.from_pretrained('facebook/dino-vits16')

    dinos = []
    dataset = PairwiseImageDatasetDINO(subject, './data/dreambooth/dataset', image_dir, feature_extractor)

    for i in tqdm(range(len(dataset))):
        inputs_A, inputs_B = dataset[i]
        if inputs_A is not None and inputs_B is not None:
            inputs_A['pixel_values'] = inputs_A['pixel_values'].to(device)
            inputs_B['pixel_values'] = inputs_B['pixel_values'].to(device) 

            outputs_A = model(**inputs_A)
            image_A_features = outputs_A.last_hidden_state[:, 0, :]

            outputs_B = model(**inputs_B)
            image_B_features = outputs_B.last_hidden_state[:, 0, :]

            image_A_features = image_A_features / image_A_features.norm(p=2, dim=-1, keepdim=True)
            image_B_features = image_B_features / image_B_features.norm(p=2, dim=-1, keepdim=True)

            sim = torch.matmul(image_A_features, image_B_features.t()) # * logit_scale
            dinos.append(sim.item())

    dino_mean = torch.tensor(dinos).mean().item()
    print('dino mean: ', dino_mean)

    return dino_mean


def lpips_image(image_dir):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    loss_fn = lpips.LPIPS(net='vgg').to(device)

    lpips_lst = []
    dataset = SelfPairwiseImageDatasetLPIPS(image_dir)
    
    for i in tqdm(range(len(dataset))):
        image_A, image_B = dataset[i]
        if image_A is not None and image_B is not None:
            image_A = image_A.to(device)
            image_B = image_B.to(device)

            distance = loss_fn(image_A, image_B)

            lpips_lst.append(distance.item())

    lpips_mean = torch.tensor(lpips_lst).mean().item()
    print('LPIPS mean', lpips_mean)

    return lpips_mean,

if __name__ == "__main__":
    output_dir = './data/output/boft/validation'
    
    epochs = list(range(201, 1601, 200)) 
    models = [401, 801, 802, 804, 1601]

    for epoch in epochs:
        for model in models:
            image_dir_path = os.path.join(output_dir, str(epoch))
            pattern = rf'_boft_{model}$'
            image_dirs = [os.path.join(image_dir_path, f) for f in os.listdir(image_dir_path) 
                          if os.path.isdir(os.path.join(image_dir_path, f)) and re.search(pattern, f)]
            clip_text_values = []
            dino_values = []
            clip_values = []
            lpips_values = []
            for image_dir in image_dirs:
                subject = re.sub(rf'_boft_{model}$', '', os.path.basename(image_dir))
                clip_text_values.append(clip_text(image_dir))
                dino_values.append(dino(subject, image_dir))
                clip_values.append(clip_image(subject, image_dir))
                lpips_values.append(lpips_image(image_dir))
            
            clip_text_result = torch.tensor(clip_text_values).mean().item()
            dino_result = torch.tensor(dino_values).mean().item()
            clip_result = torch.tensor(clip_values).mean().item()
            lpips_result = torch.tensor(lpips_values).mean().item()

            filename = f"results.txt"
            file_exists = os.path.isfile(filename)

            with open(filename, "a" if file_exists else "w") as file:
                if file_exists:
                    file.write("\n")
                file.write(f"model{model}-top10, epoch{epoch}: clip_text {clip_text_result}, dino {dino_result}, clip {clip_result}, lpips {lpips_result}")
