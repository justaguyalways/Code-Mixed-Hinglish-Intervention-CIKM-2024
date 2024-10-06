import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import pandas as pd
import random
import torchaudio
import pytorchvideo.data
from .utils import return_audio_tensor, return_image_tensor, return_video_tensor
from ast import literal_eval
from pytorchvideo.transforms import (
    ApplyTransformToKey,
    Normalize,
    RandomShortSideScale,
    RemoveKey,
    ShortSideScale,
    UniformTemporalSubsample,
)
from torchvision.transforms import (
    Compose,
    Lambda,
    RandomCrop,
    RandomHorizontalFlip,
    Resize,
    CenterCrop,
    Normalize,
    RandomRotation,
)

def generate_prompt(text, keywords, intervention , train):
    # print(data_point)

    if train:
        return f"""
            OFFENSIVE INPUT: {text}
            OFFENSIVE KEYWORDS AND INTERVENTION: {keywords} ; {intervention}
            """
    else:
        return f"""
            OFFENSIVE INPUT: {text}
            OFFENSIVE KEYWORDS AND INTERVENTION: 
            """
            

class CustomDataset(Dataset):
    def __init__(self, dataframe, train, tokenizer=None):
        self.dataframe = dataframe
        self.train = train
        self.tokenizer = tokenizer
        
        if self.train:
            self.video_transform = Compose(
                [
                    Lambda(lambda x: x / 255.0),
                    Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                    RandomShortSideScale(min_size=256, max_size=320),
                    RandomCrop((224,224)),
                    RandomHorizontalFlip(p=0.5),
                ]
            )
        else:
            self.video_transform = Compose(
                [
                    Lambda(lambda x: x / 255.0),
                    Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                ]
            )
            

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        
        text = self.dataframe["text"].iloc[idx] #change column name in dataframe
        keywords = self.dataframe["keywords_formatted"].iloc[idx]
        intervention = self.dataframe["hinglish_intervention_formatted"].iloc[idx]
        
        prompt = generate_prompt(text, keywords, intervention, self.train)

        
        video = self.dataframe['video_path'].iloc[idx]
        audio = self.dataframe['audio_path'].iloc[idx]
        
        audio = return_audio_tensor(audio)
        video = return_video_tensor(video)
                
        video = self.video_transform(video)

        sample = {
            'prompt': prompt,
            'video': video,
            'audio': audio,
        }
        
        return sample