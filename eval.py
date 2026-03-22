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
    """
    使用 CLIP 进行「文本-图像」相似度评估的数据集。

    **输入数据（初始化参数）**
    - `image_dir`：生成图像的根目录。
    - `tokenizer`：CLIP 文本 tokenizer，用于把 prompt 转成 token id。
    - `processor`：CLIP 图像处理器，用于把 PIL Image 转成模型输入（pixel_values）。

    **getitem 输出数据**
    - 返回 `(image_inputs, prompt_inputs)`：
        - `image_inputs`：`processor` 返回的字典，主要包含：
            - `pixel_values`: 形状约为 `(1, 3, H, W)` 的张量；
        - `prompt_inputs`：`tokenizer` 返回的字典，包含：
            - `input_ids`: 形状约为 `(1, seq_len)` 的 token ids；
            - `attention_mask`: 同形状的 mask；
    - 如果图像全黑（像素全为 0），返回 `(None, None)`，在后续计算中会跳过。
    """
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

        # 如果图像是纯黑图（所有像素值为 0），则认为是无效样本，返回 None
        extrema = image.getextrema()
        if all(min_val == max_val == 0 for min_val, max_val in extrema):
            return None, None
        else:
            # 文本提示 -> token ids
            prompt_inputs = self.tokenizer([prompt], padding=True, return_tensors="pt")
            # 图像 -> CLIP 所需的 pixel_values
            image_inputs = self.processor(images=image, return_tensors="pt")

            return image_inputs, prompt_inputs


class PairwiseImageDatasetCLIP(Dataset):
    """
    用于 CLIP image encoder 的「成对图像相似度」评估的数据集。

    **输入数据（初始化参数）**
    - `subject`：评估的类别名（如 "backpack"），对应 ./data/<subject> 目录。
    - `data_dir_A`：真实图像（参考图像）根目录。
    - `data_dir_B`：生成图像（待评估）根目录。
    - `processor`：CLIP 图像处理器（同上）。

    **getitem 输出数据**
    - 返回 `(inputs_A, inputs_B)`：
        - `inputs_A`：由 `processor` 处理的真实图像 A；
        - `inputs_B`：由 `processor` 处理的生成图像 B；
      二者均为字典形式，至少包含 `pixel_values`。
    - 若任意一张图为纯黑图，返回 `(None, None)`，在上层会被跳过。
    """
    def __init__(self, subject, data_dir_A, data_dir_B, processor):
        self.data_dir_A = data_dir_A
        self.data_dir_B = data_dir_B
        
        self.data_dir_A = os.path.join(self.data_dir_A, subject) 
        # A端：真实图像
        self.image_files_A = [os.path.join(self.data_dir_A, f) for f in os.listdir(self.data_dir_A) if f.endswith(".jpg")] 

        # B端：生成图像
        self.image_files_B = [os.path.join(self.data_dir_B, f) for f in os.listdir(self.data_dir_B) if f.endswith(".png")]

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.processor = processor

    def __len__(self):
        return len(self.image_files_A) * len(self.image_files_B)

    def __getitem__(self, index):
        # 将单一 index 映射为 (index_A, index_B)，实现两两配对
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
    """
    用于 DINO ViT 特征的「成对图像相似度」评估的数据集。

    初始化参数与 `PairwiseImageDatasetCLIP` 类似，只是图像处理器换成了
    `ViTFeatureExtractor`，输出用于 ViTModel。

    **getitem 输出数据**
    - 返回 `(inputs_A, inputs_B)`：
        - `inputs_A` / `inputs_B`：由 `feature_extractor` 处理后的字典，
          通常包含 `pixel_values`，供 `ViTModel` 前向使用。
    """
    def __init__(self, subject, data_dir_A, data_dir_B, feature_extractor):
        self.data_dir_A = data_dir_A
        self.data_dir_B = data_dir_B
        
        self.data_dir_A = os.path.join(self.data_dir_A, subject) 
        # A端：真实图像
        self.image_files_A = [os.path.join(self.data_dir_A, f) for f in os.listdir(self.data_dir_A) if f.endswith(".jpg")] 

        # B端：生成图像
        self.image_files_B = [os.path.join(self.data_dir_B, f) for f in os.listdir(self.data_dir_B) if f.endswith(".png")]
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.feature_extractor = feature_extractor

    def __len__(self):
        return len(self.image_files_A) * len(self.image_files_B)

    def __getitem__(self, index):
        # 同样将线性 index 拆分为 (index_A, index_B) 组合
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
    """
    在同一 subject 目录内部做「自对比」的 LPIPS 距离评估。

    **输入数据**
    - `data_dir`：根目录，内部为 `./<subject>/*.jpg`；

    **getitem 输出数据**
    - 返回 `(image_A, image_B)`：
        - `image_A` / `image_B`：经过 `Resize(512, 512)`、`ToTensor()` 和
          `Normalize(mean=0.5, std=0.5)` 预处理的张量，形状约为 `(3, 512, 512)`；
      这些张量直接可以送入 LPIPS 网络计算感知距离。
    """
    def __init__(self, data_dir):
        self.data_dir_A = data_dir
        self.data_dir_B = data_dir

        self.image_files_A = [os.path.join(self.data_dir_A, f) for f in os.listdir(self.data_dir_A) if f.endswith(".png")]
        self.image_files_B = [os.path.join(self.data_dir_B, f) for f in os.listdir(self.data_dir_B) if f.endswith(".png")]

        # LPIPS 官方推荐的输入预处理：缩放到 512x512，并做 [-1, 1] 归一化
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

        # Ensure we don't have the same index for A and B
        if index_B >= index_A:
            # 跳过 (A, A) 的对比，只比较不同图像对
            index_B += 1
        
        image_A = Image.open(self.image_files_A[index_A]) # .convert("RGB")
        image_B = Image.open(self.image_files_B[index_B]) # .convert("RGB")

        # 过滤掉全黑图（有时表示生成失败或空白）
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
    """
    使用 CLIP 的文本-图像相似度作为评估指标。

    **输入**
    - `image_dir`：生成图像的日志路径根目录；

    **主要流程**
    1. 通过 `PromptDatasetCLIP`，获得一组 (image_inputs, prompt_inputs)；
    2. 使用 CLIP 文本编码器和图像编码器分别得到 text_features 和 image_features；
    3. 对每一对 (图像, 文本) 计算余弦相似度；
    4. 对所有样本的相似度取平均，得到 `mean_similarity`。

    **输出**
    - 返回 `(mean_similarity)`：
        - `mean_similarity`：所有图像-文本对的平均余弦相似度（值越大越好，表明生成图像更符合文本描述）；
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 加载预训练 CLIP 模型与对应 tokenizer / processor
    model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to(device)
    # 文本特征
    tokenizer = AutoTokenizer.from_pretrained("openai/clip-vit-large-patch14")
    # 图像特征
    processor = AutoProcessor.from_pretrained("openai/clip-vit-large-patch14")
    
    dataset = PromptDatasetCLIP(image_dir, tokenizer, processor)
    # dataloader = DataLoader(dataset, batch_size=32)  # 目前代码中实际上逐样本使用 dataset[i]，未直接用 dataloader

    clip_texts = []
    for i in tqdm(range(len(dataset))):
        image_inputs, prompt_inputs = dataset[i]
        if image_inputs is not None and prompt_inputs is not None:
            image_inputs['pixel_values'] = image_inputs['pixel_values'].to(device)
            prompt_inputs['input_ids'] = prompt_inputs['input_ids'].to(device)
            prompt_inputs['attention_mask'] = prompt_inputs['attention_mask'].to(device)
            # print(prompt_inputs)
            # 获得图像特征与文本特征
            image_features = model.get_image_features(**image_inputs)
            text_features = model.get_text_features(**prompt_inputs)

            # 逐样本计算余弦相似度（CLIP 常用指标）
            sim = cosine_similarity(image_features, text_features)

            #image_features = image_features / image_features.norm(p=2, dim=-1, keepdim=True)
            #text_features = text_features / text_features.norm(p=2, dim=-1, keepdim=True)
            #logit_scale = model.logit_scale.exp()
            #sim = torch.matmul(text_features, image_features.t()) * logit_scale
            clip_texts.append(sim.item())

    clip_text_mean = torch.tensor(clip_texts).mean().item()
    print('clip-text mean: ', clip_text_mean)

    return clip_text_mean


def clip_image(subject, image_dir):
    """
    使用 CLIP 图像编码器对「真实图像 vs 生成图像」做图像-图像相似度评估。

    **输入**
    - `image_dir`：生成图像根目录；

    **流程**
    1. 对于每个 `subject`：
        - 从 `./data/<subject>/*.jpg` 读取真实图像；
        - 从 `image_dir` 内对应带有该 subject 与 prompt_token 的子目录读取生成图像；
    2. 构造所有 (A, B) 成对组合；
    3. 经 CLIP 图像编码器得到 `image_A_features`、`image_B_features`；
    4. 将特征归一化后，计算内积（相当于余弦相似度）；
    5. 对所有成对样本的相似度取均值，得到 `mean_similarity`。

    **输出**
    - 返回 `(mean_similarity)`：数值越大表示生成图在 CLIP 空间中越接近真实图。
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to(device)
    # Get the image features
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

            # 特征 L2 归一化，使得内积约等于余弦相似度
            image_A_features = image_A_features / image_A_features.norm(p=2, dim=-1, keepdim=True)
            image_B_features = image_B_features / image_B_features.norm(p=2, dim=-1, keepdim=True)
        
            logit_scale = model.logit_scale.exp()
            sim = torch.matmul(image_A_features, image_B_features.t()) # * logit_scale
            clips.append(sim.item())

    clip_mean = torch.tensor(clips).mean().item()
    print('clip mean: ', clip_mean)

    return clip_mean


def dino(subject, image_dir):
    """
    使用自监督视觉模型 DINO (ViT-S/16) 的 [CLS] 特征，评估真实图像与生成图像的相似度。

    **输入**
    - `subject`：类别名；
    - `image_dir`：生成图像根目录；

    **流程**
    1. 对每个 `subject`，构建 `PairwiseImageDatasetDINO` 数据集；
    2. 使用 `ViTFeatureExtractor` 将图像处理为 ViT 输入；
    3. 前向 DINO ViT 模型，取 `last_hidden_state[:, 0, :]` 作为 [CLS] 全局特征；
    4. 对特征做 L2 归一化，然后计算内积作为相似度；
    5. 汇总所有 (A, B) 对的相似度，取平均值。

    **输出**
    - 返回 `(mean_similarity)`：数值越大表示在 DINO 特征空间中越接近。
    """
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

            # A 图像的 ViT 输出
            outputs_A = model(**inputs_A)
            image_A_features = outputs_A.last_hidden_state[:, 0, :]

            # B 图像的 ViT 输出
            outputs_B = model(**inputs_B)
            image_B_features = outputs_B.last_hidden_state[:, 0, :]

            # L2 归一化便于内积代表余弦相似度
            image_A_features = image_A_features / image_A_features.norm(p=2, dim=-1, keepdim=True)
            image_B_features = image_B_features / image_B_features.norm(p=2, dim=-1, keepdim=True)

            sim = torch.matmul(image_A_features, image_B_features.t()) # * logit_scale
            dinos.append(sim.item())

    dino_mean = torch.tensor(dinos).mean().item()
    print('dino mean: ', dino_mean)

    return dino_mean


def lpips_image(image_dir):
    """
    使用 LPIPS（Learned Perceptual Image Patch Similarity）。

    **输入**
    - `image_dir`：生成图像根目录；

    **流程**
    1. 对每个 `subject` 构造 `PairwiseImageDatasetLPIPS`，内部返回两张经过统一预处理的图像张量；
    2. 使用 `lpips.LPIPS(net='vgg')` 模型计算 (image_A, image_B) 之间的 LPIPS 距离；
    3. 对所有样本对的距离取平均，得到 `mean_similarity`；

    **输出**
    - 返回 `(mean_similarity, 'lpips_image')`：
        - 这里的 mean_similarity 是「平均 LPIPS 距离」，**数值越小越好**，
          表示生成图与真实图在感知上越接近。
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Set up the LPIPS model (vgg=True uses the VGG-based model from the paper)
    loss_fn = lpips.LPIPS(net='vgg').to(device)

    lpips_lst = []
    dataset = SelfPairwiseImageDatasetLPIPS(image_dir)
    
    for i in tqdm(range(len(dataset))):
        image_A, image_B = dataset[i]
        if image_A is not None and image_B is not None:
            image_A = image_A.to(device)
            image_B = image_B.to(device)

            # Calculate LPIPS between the two images
            distance = loss_fn(image_A, image_B)

            lpips_lst.append(distance.item())

    lpips_mean = torch.tensor(lpips_lst).mean().item()
    print('LPIPS mean', lpips_mean)

    return lpips_mean,

if __name__ == "__main__":
    """
    脚本入口：对指定目录下的生成结果进行评估，并把结果追加写入 `results.txt`。

    **输入配置**
    - `image_dir`：生成图像日志目录；

    **评估指标**
    - `clip_text(image_dir, epoch)`：文本-图像相似度，值越大越好；
    - `clip_image(image_dir, epoch)`：真实图 vs 生成图的 CLIP 图像特征相似度，值越大越好；
    - `dino(image_dir, epoch)`：使用 DINO ViT 特征的图像-图像相似度，值越大越好；
    - `lpips_image(image_dir, epoch)`：LPIPS 感知距离，值越小越好；

    """
    output_dir = './data/output/boft/validation'
    
    epochs = list(range(201, 1601, 200))  # [201, 401, 601, 801, 1001, 1201]
    # models = [401, 801, 802, 804, 1601]
    models = [3201]

    for epoch in epochs:
        for model in models:
            image_dir_path = os.path.join(output_dir, str(epoch))
            # 匹配 {subject_name}_boft_{model} 格式的文件夹
            pattern = rf'_boft_{model}$'
            image_dirs = [os.path.join(image_dir_path, f) for f in os.listdir(image_dir_path) 
                          if os.path.isdir(os.path.join(image_dir_path, f)) and re.search(pattern, f)]
            clip_text_values = []
            dino_values = []
            clip_values = []
            lpips_values = []
            count = 0
            for image_dir in image_dirs:
                count += 1
                if count <= 10:
                    # 从文件夹名中提取 subject name（例如：backpack_boft_401 -> backpack）
                    subject = re.sub(rf'_boft_{model}$', '', os.path.basename(image_dir))
                    clip_text_values.append(clip_text(image_dir))
                    dino_values.append(dino(subject, image_dir))
                    clip_values.append(clip_image(subject, image_dir))
                    lpips_values.append(lpips_image(image_dir))
            
            clip_text_result = torch.tensor(clip_text_values).mean().item()
            dino_result = torch.tensor(dino_values).mean().item()
            clip_result = torch.tensor(clip_values).mean().item()
            lpips_result = torch.tensor(lpips_values).mean().item()

            filename = f"results.txt"  # 保存评估结果的文件名
            # 检查文件是否存在，用于决定是覆盖写还是追加写
            file_exists = os.path.isfile(filename)

            # 若已存在则追加，否则创建新文件
            with open(filename, "a" if file_exists else "w") as file:
                # 如果是追加模式，在末尾先换一行
                if file_exists:
                    file.write("\n")
                file.write(f"model{model}-top10, epoch{epoch}: clip_text {clip_text_result}, dino {dino_result}, clip {clip_result}, lpips {lpips_result}")
