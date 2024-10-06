import torch
import torch.nn as nn
from tokenizers import AddedToken
from transformers import CLIPModel, VideoMAEModel,  WhisperModel, VideoMAEConfig, CLIPConfig, Wav2Vec2Config
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, AutoModelForSeq2SeqLM
from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model
from .additional_modules import LSTM_fc, Conv1d_fc, Gate_Attention
from argparse import Namespace 

class Multimodal_LLM(nn.Module):
    
    def __init__(self, batch_size, config, tokenizer, adapter_llm):
        super(Multimodal_LLM, self).__init__()
        
        self.config = config
        
        self.batch_size = batch_size
        
        self.tokenizer = tokenizer
        
        # self.custom_special_ids_start = len(self.tokenizer.get_vocab())-len(self.config.special_tokens)

        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        self.tokenizer.padding_side = "right"


        self.video_encoder = VideoMAEModel.from_pretrained(config.video_encoder)
        self.audio_encoder =  WhisperModel.from_pretrained(config.audio_encoder).encoder
        
        
        self.adapter_llm = adapter_llm
        

        self.output_seq_len = min(self.config.audio_seq_len, self.config.video_seq_len, self.config.multimodal_len)
        
        
        if self.config.lstm_or_conv:
            self.transform_audio_to_hidden = LSTM_fc(input_size=self.config.audio_dim, hidden_size=self.config.audio_dim,
                                                    num_layers=self.config.lstm_num_layers, output_seq_len=self.output_seq_len, output_size=self.config.llm_embed_dim)
            self.transform_video_to_hidden = LSTM_fc(input_size=self.config.video_dim, hidden_size=self.config.video_dim,
                                                    num_layers=self.config.lstm_num_layers, output_seq_len=self.output_seq_len, output_size=self.config.llm_embed_dim)
        else:
            self.transform_audio_to_hidden = Conv1d_fc(encoder_embed_dim=self.config.audio_dim, llm_embed_dim=self.config.llm_embed_dim,
                                                       kernel_size=self.config.audio_conv_kernel, stride=self.config.audio_conv_stride, padding=self.config.audio_conv_padding)
            self.transform_video_to_hidden = Conv1d_fc(encoder_embed_dim=self.config.video_dim, llm_embed_dim=self.config.llm_embed_dim,
                                                       kernel_size=self.config.video_conv_kernel, stride=self.config.video_conv_stride, padding=self.config.video_conv_padding)
        
        self.video_align_attention = nn.MultiheadAttention(self.config.llm_embed_dim, 
                                                             self.config.attention_heads * 2,
                                                             dropout=self.config.attn_dropout,
                                                             add_bias_kv=self.config.is_add_bias_kv,
                                                             add_zero_attn=self.config.is_add_zero_attn)
        self.audio_align_attention = nn.MultiheadAttention(self.config.llm_embed_dim, 
                                                             self.config.attention_heads * 2,
                                                             dropout=self.config.attn_dropout,
                                                             add_bias_kv=self.config.is_add_bias_kv,
                                                             add_zero_attn=self.config.is_add_zero_attn)

        self.gate_fusion = Gate_Attention(num_hidden_a = self.config.llm_embed_dim, num_hidden_b = self.config.llm_embed_dim, num_hidden = self.config.llm_embed_dim)
        
        
    def forward(self, inputs):
        
        batch_size = inputs["video"].shape[0]
        
        #ids
        bos = torch.ones([batch_size, 1], dtype=torch.int64, device=self.config.device) * self.tokenizer.bos_token_id
        sep = torch.ones([batch_size, 1], dtype=torch.int64, device=self.config.device) * self.tokenizer.eos_token_id
        eos = torch.ones([batch_size, 1], dtype=torch.int64, device=self.config.device) * self.tokenizer.eos_token_id
    
    
        #mask
        attention_mask_bos = torch.ones([batch_size, 1], dtype=torch.int64, device=self.config.device)
        attention_mask_multimodal = torch.ones([batch_size, self.output_seq_len+1], dtype=torch.int64, device=self.config.device)
        attention_mask_eos = torch.zeros([batch_size, 1], dtype=torch.int64, device=self.config.device)
        
        
        #config.tokenizer_max_len = 256
        text = inputs["prompt"]
        tokenized_text = self.tokenizer(text, return_tensors="pt", padding="max_length", truncation=True, max_length=self.config.tokenizer_max_len-3, add_special_tokens=False)
        
        # print("tokenization success\n")
        embed_tokens = self.adapter_llm.model.model.embed_tokens.to(self.config.device)
        token_embeddings = embed_tokens.weight.unsqueeze(0).repeat(batch_size, 1, 1).transpose(0, 1).contiguous().to(self.config.device)
        text_embeds = self.adapter_llm.model.model.embed_tokens(tokenized_text.input_ids.to(self.config.device))
        attention_mask = tokenized_text.attention_mask.to(self.config.device)
        
        # print(f"Text Embeddings shape: {text_embeds.shape}")

        #multimodal processing
        video_encoder_out = self.video_encoder(inputs["video"])
        audio_encoder_out = self.audio_encoder(inputs["audio"])
        
        # print(f"Video Encoder Output shape: {video_encoder_out.last_hidden_state.shape}")
        # print(f"Audio Encoder Output shape: {audio_encoder_out.last_hidden_state.shape}")

        video_encoder_out = self.transform_video_to_hidden(video_encoder_out.last_hidden_state)
        audio_encoder_out = self.transform_audio_to_hidden(audio_encoder_out.last_hidden_state)

        # print(f"Transformed Video Encoder Output shape: {video_encoder_out.shape}")
        # print(f"Transformed Audio Encoder Output shape: {audio_encoder_out.shape}")

        
        video_encoder_out = self.video_align_attention(video_encoder_out.transpose(0, 1).contiguous(), token_embeddings,
                                                    token_embeddings)[0].transpose(0, 1).contiguous()
        audio_encoder_out = self.audio_align_attention(audio_encoder_out.transpose(0, 1).contiguous(), token_embeddings,
                                                    token_embeddings)[0].transpose(0, 1).contiguous()
        
        # print(f"Video Encoder Output after attention shape: {video_encoder_out.shape}")
        # print(f"Audio Encoder Output after attention shape: {audio_encoder_out.shape}")

        level_2 = self.gate_fusion(video_encoder_out, audio_encoder_out)
        # print(f"Level 2 Output shape: {level_2.shape}")
    
        #input prompt embed final
        #eos and bos embeds with [bs,1]
        bos_embeds = self.adapter_llm.model.model.embed_tokens(bos)
        sep_embeds = self.adapter_llm.model.model.embed_tokens(sep)
        eos_embeds = self.adapter_llm.model.model.embed_tokens(eos)

        # bos_embeds_shape = bos_embeds.shape
        # level_2_shape = level_2.shape
        # sep_embeds_shape = sep_embeds.shape
        # text_embeds_shape = text_embeds.shape
        # eos_embeds_shape = eos_embeds.shape

        # print(f"BOS Embeddings shape: {bos_embeds_shape}")
        # print(f"Level 2 Output shape: {level_2_shape}")
        # print(f"SEP Embeddings shape: {sep_embeds_shape}")
        # print(f"Text Embeddings shape: {text_embeds_shape}")
        # print(f"EOS Embeddings shape: {eos_embeds_shape}")

        text_embeds = torch.cat([bos_embeds, level_2, sep_embeds, text_embeds, eos_embeds], dim=1)

        # print(f"Concatenated Text Embeddings shape: {text_embeds.shape}")

        # attention_mask_bos_shape = attention_mask_bos.shape
        # attention_mask_multimodal_shape = attention_mask_multimodal.shape
        # attention_mask_shape = attention_mask.shape
        # attention_mask_eos_shape = attention_mask_eos.shape

        # print(f"Attention Mask (BOS) shape: {attention_mask_bos_shape}")
        # print(f"Attention Mask (Multimodal) shape: {attention_mask_multimodal_shape}")
        # print(f"Attention Mask shape: {attention_mask_shape}")
        # print(f"Attention Mask (EOS) shape: {attention_mask_eos_shape}")

        attention_mask = torch.cat([attention_mask_bos, attention_mask_multimodal, attention_mask, attention_mask_eos], dim=1)

        # print(f"Attention Mask after concatenation shape: {attention_mask.shape}")

        #targets
        #4 for encoder as -100 for bos, the multimodal inputs, sep and at last eos
        empty_multimodal_targets = (
            torch.ones([batch_size, level_2.shape[1] + 2], dtype=torch.long).fill_(-100)
        ).to(self.config.device)
        
        targets = tokenized_text.input_ids.masked_fill(
            tokenized_text.input_ids == self.tokenizer.pad_token_id, -100
        ).to(self.config.device)

        eos_targets = (
            torch.ones([batch_size, 1], dtype=torch.long).fill_(-100)
        ).to(self.config.device)
        
        targets = torch.cat([empty_multimodal_targets, targets, eos_targets], dim=1)
        
        # print(f"Targets shape: {targets.shape}")
        
        outputs = []
        
        if self.config.train:
            outputs = self.adapter_llm(inputs_embeds=text_embeds, attention_mask=attention_mask, return_dict=True, labels=targets)
        else:
            outputs = self.adapter_llm.generate(inputs_embeds=text_embeds,  max_new_tokens=self.config.generate_text_max_len)
        return outputs
