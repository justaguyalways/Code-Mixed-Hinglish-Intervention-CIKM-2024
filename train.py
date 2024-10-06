'''
python -W ignore run.py
python block_core.py
find -type d -name 'pymp*' -exec rm -r {} \;
'''

import torch
import torch.nn as nn
from tokenizers import AddedToken
from transformers import CLIPModel, VideoMAEModel, Wav2Vec2Model, VideoMAEConfig, CLIPConfig, Wav2Vec2Config, XLMRobertaConfig
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, AutoModelForSeq2SeqLM
from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model
from model.additional_modules import LSTM_fc, FC_head, Gate_Attention
from argparse import Namespace 
from model.model import Multimodal_LLM
from data.dataset import CustomDataset
from iteration import train
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
from copy import deepcopy
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer, AlbertTokenizer, XLMRobertaTokenizerFast, PreTrainedTokenizerFast #only for gpt2 and assign values
from transformers import GPT2Model, BertModel, AlbertModel, XLMRobertaModel
import pickle
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["WORLD_SIZE"] = "1"
os.environ["HF_TOKEN"] = ""

name = "zephyr_vidmae_whisper_"
run = 0

config = Namespace(
    file_name= name + str(run),
    device=torch.device("cuda"),
    tokenizer_path="ckpts",
    video_encoder="MCG-NJU/videomae-base",
    audio_encoder="openai/whisper-small",
    lstm_or_conv = False,
    video_conv_kernel=36,
    video_conv_stride=24,
    video_conv_padding=0,
    audio_conv_kernel=50,
    audio_conv_stride=23,
    audio_conv_padding=1,
    llm_embed_dim=4096,
    llm_output_dim=3072,
    attn_dropout=0.1,
    is_add_bias_kv=True,
    is_add_zero_attn=True,
    attention_heads=8,
    video_dim=768,
    audio_dim=768,
    image_seq_len=197,
    video_seq_len=1568,
    audio_seq_len=1500,
    multimodal_len=64,
    lstm_num_layers=1,
    tokenizer_max_len=512,
    add_pooling = False,
    train=True,
    directory = "checkpoints/",
    results_directory = "results/",
    learning_rate= 2e-5, 
)

df = pd.read_csv("final_data/hinglish_interventions_final.csv")
df_train_val, df_test = train_test_split(df, test_size=0.1, random_state=28703)
df_train, df_val = train_test_split(df_train_val, test_size=0.1, random_state=28703)

num_epochs = 30
patience = 5
batch_size = 1

#for openhathi
llm_name = 'HuggingFaceH4/zephyr-7b-beta'
loraconfig = LoraConfig(
        r=16, 
        lora_alpha=32, 
        #target_modules=["query_key_value"],
        target_modules = ["q_proj", "v_proj", "k_proj", "embed_tokens", "gate_proj", "up_proj", "down_proj", "lm_head"],
        lora_dropout=0.05, 
        bias="none", 
        task_type="CAUSAL_LM"
    )


#same for all models
tokenizer = AutoTokenizer.from_pretrained(llm_name)
model = AutoModelForCausalLM.from_pretrained(llm_name, torch_dtype=torch.bfloat16)

for name, param in model.named_parameters():
        param.requires_grad = False

model.config.use_cache = False

kbit_model = prepare_model_for_kbit_training(model)

adapter_model = get_peft_model(kbit_model, loraconfig)


model = Multimodal_LLM(batch_size=batch_size, config=config, tokenizer=tokenizer, adapter_llm=adapter_model)
seed = 42
torch.manual_seed(seed)


train_ds = CustomDataset(dataframe=df_train, train=True, tokenizer=tokenizer)
val_ds = CustomDataset(df_test, train=True, tokenizer=tokenizer)
test_ds = CustomDataset(df_test, train=False, tokenizer=tokenizer)

train_dataloader = DataLoader(train_ds, batch_size=batch_size, num_workers=16, shuffle=True)
val_dataloader = DataLoader(val_ds, batch_size=batch_size, num_workers=16)
test_dataloader = DataLoader(test_ds, batch_size=batch_size, num_workers=16)


train(model, train_dataloader, val_dataloader, test_dataloader, config, num_epochs, patience, devices=None)
