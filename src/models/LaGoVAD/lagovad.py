import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import time
import transformers
from transformers import AutoTokenizer
from torch import Tensor
from lightning.pytorch.utilities import rank_zero_only
from einops.layers.torch import Rearrange
from einops import einsum, rearrange
from pathlib import Path

import torchmetrics
import numpy as np
import math
import random
from collections import OrderedDict, defaultdict
from typing import Optional, Any, List, Union, Dict
from tqdm import tqdm

from ..base import VADCBase
from .configs import LaGoVADTrainingConfig, LaGoVADModelConfig
from .modeling_clip import SoftPromptCLIPTextModel
from .temporal_encoders import VanillaTransformer, LGTAdapter, RoFormerEncoder
from .fusion_encoders import FusionV1, VadCLIPFusion
from .heads import BinaryHead, MultiClassHead
from .utils import save_pred_result, get_attention_mask
from .losses import (
    mil_loss,
    multi_class_mil_loss,
    multi_class_mil_loss_v2,
    supervised_loss,
    pseudo_sup_mil_loss,
    CapContrastLoss,
)
from .verbalizer import RandomHardPromptVerbalizer, DatasetSpecVerbalizer


class LaGoVADLightModel(VADCBase):
    def __init__(self,
                 model_config: LaGoVADModelConfig,
                 training_config: LaGoVADTrainingConfig
                 ):
        super().__init__()
        self.model_config = model_config
        self.training_config = training_config

        # 1 feature = [?] frame
        self.frame_time = self.training_config.frame_time

        # build temporal encoder
        if self.model_config.temp_enc_type == 'Vanilla':
            self.temporal_encoder = VanillaTransformer(model_config.bert_config)
        elif self.model_config.temp_enc_type == 'RoFormer':
            self.temporal_encoder = RoFormerEncoder(model_config.bert_config)
        elif self.model_config.temp_enc_type == 'LGTAdapter':
            self.temporal_encoder = LGTAdapter(model_config.bert_config,
                                               attn_window=model_config.temp_window_size)
        else:
            raise NotImplementedError
        if self.model_config.temp_gate:
            self.temp_gate_weight = self.model_config.temp_gate_weight
            self.gate_alpha = nn.Parameter(
                torch.zeros(1, dtype=torch.float32).fill_(self.training_config.temp_gate_init))

        # build fusion
        if self.model_config.fusion_type in ['co_attn', 'yw', 'asym']:
            self.fusion = FusionV1(
                fusion_type=self.model_config.fusion_type,
                d_model=model_config.hidden_size,
                num_layers=model_config.fusion_num_layers
            )
        elif self.model_config.fusion_type in ['vadclip']:
            self.fusion = VadCLIPFusion(
                d_model=model_config.hidden_size
            )
        elif self.model_config.fusion_type == '' or self.model_config.fusion_type is None:
            self.fusion = None

        # load clip text model & tokenizer
        self.clip_text_model = SoftPromptCLIPTextModel(
            model_config.clip_name,
            model_config.num_soft_prompts
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_config.clip_name)
        for p in self.clip_text_model.model.parameters():
            p.requires_grad = False

        # load verbalizer
        if self.model_config.verbalizer_type is None:
            self.verbalizer = None
        elif self.model_config.verbalizer_type == 'hard':
            self.verbalizer = RandomHardPromptVerbalizer()
        elif self.model_config.verbalizer_type == 'dataset-specific':
            self.verbalizer = DatasetSpecVerbalizer()
        else:
            raise NotImplementedError

        # =================== Multi class branch ===================
        self.sim_head = MultiClassHead(
            self.model_config.multi_head_type,
            d_model=model_config.hidden_size,
            d_proj=model_config.hidden_size // 2,
            temperature_init=0.2
        )


        # =================== Binary class branch ===================
        self.bin_head = BinaryHead(
            model_config.bin_head_type,
            d_model=model_config.hidden_size,
            num_layers=self.model_config.head_num_layers,
            activation=self.model_config.head_activation,
            kernel_size=self.model_config.head_kernel_size,
            norm=self.model_config.head_norm,
            adp_alpha_init=0.25,
        )

        self.metrics = None
        self.pred_results = []
        self.train_class_names = None
        self.test_class_names = None
        self.predict_class_names = None
        self.log_dir = None
        self.log_img_dir = None

        self.save_hyperparameters()

        if model_config.pretrained_weight_path is not None:
            print(f'Loading pretrained weight from `{model_config.pretrained_weight_path}`')
            state_dict = torch.load(model_config.pretrained_weight_path, weights_only=True, map_location='cpu')['state_dict']
            self.load_state_dict(state_dict, strict=False)

    def _init_log_dir(self):
        if self.log_dir is None:
            self.log_dir = Path(
                self.trainer.logger.save_dir,
                self.trainer.logger.name,
                self.trainer.logger.version,
            )
            self.log_img_dir = self.log_dir / 'result_images'
            self.log_dir.mkdir(exist_ok=True, parents=True)
            self.log_img_dir.mkdir(exist_ok=True)
            if self.trainer.local_rank == 0:
                print(f'=======> Setting log directory to: {self.log_dir}')

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.training_config.lr,
                                      weight_decay=self.training_config.weight_decay)
        # scheduler = transformers.get_cosine_schedule_with_warmup(optimizer, **self.lr_scheduler_cfg)
        scheduler = transformers.get_scheduler(self.training_config.lr_scheduler_name,
                                               **self.training_config.lr_scheduler_kwargs,
                                               num_training_steps=self.trainer.estimated_stepping_batches,
                                               optimizer=optimizer)
        return {
            'optimizer': optimizer,
            'lr_scheduler': {'scheduler': scheduler, 'interval': 'step'},
        }

    def on_train_start(self) -> None:
        self.train_class_names = self.trainer.train_dataloader.dataset.class_names
        # check if class names are in self.verbalizer
        if self.verbalizer is not None:
            if self.model_config.verbalizer_type == 'dataset-specific':
                self.verbalizer.set_dataset(self.trainer.train_dataloader.dataset.abbr)
            for class_name in self.train_class_names:
                if class_name not in self.verbalizer.cls2text:
                    raise ValueError(f'Class name {class_name} not in the verbalizer')

        self._init_log_dir()

    def on_validation_start(self) -> None:
        self.test_class_names = []
        if type(self.trainer.val_dataloaders) is list:
            dataloaders = self.trainer.val_dataloaders
        else:
            dataloaders = [self.trainer.val_dataloaders]
        for i, dl in enumerate(dataloaders):
            self.test_class_names.append(dl.dataset.class_names)
            # check if class names are in self.verbalizer
            if self.verbalizer is not None:
                if self.model_config.verbalizer_type == 'dataset-specific':
                    self.verbalizer.set_dataset(dl.dataset.abbr)
                if self.test_class_names[-1][-1] == 'Abnormal':  # 2 class
                    continue
                for class_name in self.test_class_names[-1]:  # 目前的最后一个
                    if class_name not in self.verbalizer.cls2text:
                        raise ValueError(f'Class name {class_name} not in the verbalizer')

        self._init_log_dir()

    def on_test_start(self) -> None:
        self.test_class_names = []
        if type(self.trainer.test_dataloaders) is list:
            dataloaders = self.trainer.test_dataloaders
        else:
            dataloaders = [self.trainer.test_dataloaders]
        for i, dl in enumerate(dataloaders):
            self.test_class_names.append(dl.dataset.class_names)
            # check if class names are in self.verbalizer
            if self.verbalizer is not None:
                if self.model_config.verbalizer_type == 'dataset-specific':
                    self.verbalizer.set_dataset(dl.dataset.abbr)
                if self.test_class_names[-1][-1] == 'Abnormal':  # 2 class
                    continue
                for class_name in self.test_class_names[-1]:  # 目前的最后一个
                    if class_name not in self.verbalizer.cls2text:
                        raise ValueError(f'Class name {class_name} not in the verbalizer')

    def on_predict_start(self) -> None:
        self.predict_class_names = []
        if type(self.trainer.predict_dataloaders) is list:
            dataloaders = self.trainer.predict_dataloaders
        else:
            dataloaders = [self.trainer.predict_dataloaders]
        for i, dl in enumerate(dataloaders):
            self.predict_class_names.append(dl.dataset.class_names)
            # check if class names are in self.verbalizer
            if self.verbalizer is not None:
                if self.model_config.verbalizer_type == 'dataset-specific':
                    self.verbalizer.set_dataset(dl.dataset.abbr)
                if self.predict_class_names[-1][-1] == 'Abnormal':  # 2 class
                    continue
                for class_name in self.predict_class_names[-1]:  # 目前的最后一个
                    if class_name not in self.verbalizer.cls2text:
                        raise ValueError(f'Class name {class_name} not in the verbalizer')

    def on_save_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        # don't save CLIP
        for k in list(checkpoint['state_dict'].keys()):
            if k.startswith('clip_text_model.model'):
                del checkpoint['state_dict'][k]

    def on_load_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        self.strict_loading = False

    def get_extended_attention_mask(
            self, attention_mask: Tensor, input_shape: tuple[int] = None,
            device: torch.device = None, dtype = None
    ) -> Tensor:
        """
        Makes broadcastable attention and causal masks so that future and masked tokens are ignored.

        Arguments:
            attention_mask (`torch.Tensor`):
                Mask with ones indicating tokens to attend to, zeros for tokens to ignore.
            input_shape (`Tuple[int]`):
                The shape of the input to the model.

        Returns:
            `torch.Tensor` The extended attention mask, with a the same dtype as `attention_mask.dtype`.
        """
        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        if attention_mask.dim() == 3:
            extended_attention_mask = attention_mask[:, None, :, :]
        elif attention_mask.dim() == 2:
            # Provided a padding mask of dimensions [batch_size, seq_length]
            extended_attention_mask = attention_mask[:, None, None, :]
        else:
            raise ValueError(
                f"Wrong shape for input_ids or attention_mask (shape {attention_mask.shape})"
            )

        # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
        # masked positions, this operation will create a tensor which is 0.0 for
        # positions we want to attend and the dtype's smallest value for masked positions.
        # Since we are adding it to the raw scores before the softmax, this is
        # effectively the same as removing these entirely.
        extended_attention_mask = extended_attention_mask.to(dtype=dtype)  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * torch.finfo(dtype).min
        return extended_attention_mask

    @staticmethod
    def get_extended_local_attention_mask(
            attention_mask: Tensor,
            local_window_size: int = 15,
            dtype=torch.float,
    ):
        """
        Args:
            attention_mask: Tensor [B,L]
            local_window_size:
            dtype:
        Returns:
        """
        device = attention_mask.device
        length = attention_mask.shape[-1]
        ws = (local_window_size - 1) // 2
        bs = attention_mask.shape[0]
        local_mask_upper = torch.triu(torch.ones(length, length), -ws)
        local_mask_lower = torch.tril(torch.ones(length, length), ws)
        local_mask = (local_mask_upper * local_mask_lower).unsqueeze(0).repeat(bs, 1, 1)  # [B,L,L]
        # extended_attention_mask = self.get_extended_attention_mask(attention_mask, dtype=dtype)  # [B,1,1,L]
        extended_attention_mask = attention_mask[:, None, None, :].to(dtype=dtype)  # [B,1,1,L]
        extended_local_attention_mask = extended_attention_mask * local_mask[:, None, :, :].to(device)  # [B,1,1,L]
        extended_local_attention_mask = (1.0 - extended_local_attention_mask) * torch.finfo(dtype).min

        return extended_local_attention_mask

    def post_process(self, cls_label_idx, fd_dict):
        # ================================
        # Post-Process
        # ================================
        bs = cls_label_idx.shape[0]
        device = fd_dict['cls_sim_mat'].device

        if self.training_config.mul_score_type == 'vanilla':
            mul_logits = fd_dict['cls_sim_mat'].topk(dim=1, k=5)[0].mean(1)  # B,C
        elif self.training_config.mul_score_type == 'new1':
            # B,T,C -> B,T -> B,1
            mul_logits_normal = fd_dict['cls_sim_mat'][:, :, 0].topk(dim=1, k=1, largest=False)[0]
            # B,T,C -> B,T,C-1 -> B,C-1
            mul_logits_abnormal =  fd_dict['cls_sim_mat'][:, :, 1:].mean(1)
            mul_logits = torch.cat([mul_logits_normal, mul_logits_abnormal], dim=1)
        else:
            raise NotImplementedError
        mul_softmax_probs = mul_logits.softmax(dim=-1)

        # select only anomaly items, obtain their anomaly distribution and apply softmax
        mul_softmax_probs_only_ano = mul_logits[cls_label_idx != 0][:, 1:].softmax(dim=-1)
        mul_sigmoid_probs = fd_dict['cls_sim_mat'].sigmoid()
        # generate scores
        if self.training_config.score_type == 'binary':
            scores = torch.sigmoid(fd_dict['cls_bin_logits'])
        elif self.training_config.score_type == 'mul_max':
            tgt_logits = mul_sigmoid_probs.max(-1)[0]
            scores = torch.sigmoid(tgt_logits)
        else:
            raise NotImplementedError

        return scores, mul_softmax_probs, mul_softmax_probs_only_ano

    def submit_metrics(self, batch, scores, mul_softmax_probs, mul_softmax_probs_only_ano, dataloader_idx: int):
        # ================================
        # Metrics
        # ================================
        bs = batch['v_feat'].shape[0]
        for bi in range(bs):
            tgt_len = batch['target_length'][bi]
            if len(scores[bi]) >= tgt_len:  # pad
                real_score = scores[bi][:tgt_len]
            else:  # resample  B,C,T
                real_score = F.interpolate(scores[bi][None, None, :], tgt_len, mode='linear')[0, 0]  # L'
            real_score = torch.repeat_interleave(real_score, self.frame_time[dataloader_idx], dim=0)  # L

            if 'temp_span' in batch:
                gt = torch.zeros_like(real_score)
                # 1,t,2
                for i in range(batch['temp_span'].shape[1]):
                    anno = batch['temp_span'][bi, i]
                    if anno[0] == anno[1]:
                        continue
                    anno *= self.frame_time[dataloader_idx]
                    gt[anno[0]:anno[1]] = 1.0

                self.metrics[dataloader_idx]['auroc'].update(real_score, gt)
                self.metrics[dataloader_idx]['confusion_mat'].update(real_score, gt)
                self.metrics[dataloader_idx]['avg_precision'].update(real_score, gt.long())
                self.pred_results.append({
                    'video_path': batch['video_path'][bi],
                    'score': real_score.cpu().numpy(),
                    'temp_span': batch['temp_span'][bi].cpu().numpy(),
                    'dataloader_idx': dataloader_idx
                })
            else:
                self.pred_results.append({
                    'video_path': batch['video_path'][bi],
                    'score': real_score.cpu().numpy(),
                    'dataloader_idx': dataloader_idx
                })

        if 'top1_acc' in self.metrics[dataloader_idx]:
            self.metrics[dataloader_idx]['top1_acc'].update(mul_softmax_probs, batch['cls_label_idx'])
            self.metrics[dataloader_idx]['top2_acc'].update(mul_softmax_probs, batch['cls_label_idx'])
            self.metrics[dataloader_idx]['top3_acc'].update(mul_softmax_probs, batch['cls_label_idx'])
            if len(mul_softmax_probs_only_ano) > 0:
                self.metrics[dataloader_idx]['A_top1_acc'].update(
                    mul_softmax_probs_only_ano,
                    batch['cls_label_idx'][batch['cls_label_idx'] != 0] - 1
                )

    def _temporal_encoding(self, v_feat, v_feat_l):
        device = self.device

        # encoder vision  ==>  BxTxD
        vis_attn_mask = get_attention_mask(v_feat_l, v_feat.shape[1])
        if self.model_config.temp_window_size and self.model_config.temp_window_size > 0:
            vis_attn_mask = self.get_extended_local_attention_mask(
                vis_attn_mask,
                local_window_size=self.model_config.temp_window_size,
                dtype=v_feat.dtype
            )
        else:
            vis_attn_mask = self.get_extended_attention_mask(
                vis_attn_mask, (0,),  #
                device=device, dtype=v_feat.dtype
            )

        if self.model_config.temp_enc_type == 'LGTAdapter':
            vis_outputs = self.temporal_encoder(v_feat, v_feat_l)
        else:
            vis_outputs = self.temporal_encoder(v_feat, vis_attn_mask).last_hidden_state
        vis_feats = v_feat
        if self.model_config.temp_gate:
            vis_feats = vis_feats + vis_outputs * torch.tanh(self.gate_alpha) * self.temp_gate_weight
        else:
            vis_feats = vis_feats + vis_outputs

        normed_vis_feats = vis_feats / vis_feats.norm(dim=-1, keepdim=True)
        return vis_feats, normed_vis_feats

    def _head(self, vis_feats, text_outputs, v_fused, t_fused):
        # === HEAD ===
        bs = vis_feats.shape[0]

        # binary-class head
        if self.model_config.bin_head_type == 'vanilla':
            bin_logits = self.bin_head(before_fused=vis_feats)
        elif self.model_config.bin_head_type == 'fused_vanilla':
            bin_logits = self.bin_head(after_fused=v_fused)
        elif self.model_config.bin_head_type == 'adaptive':
            bin_logits = self.bin_head(before_fused=vis_feats, after_fused=v_fused)
        else:
            raise NotImplementedError

        # multi-class head
        if self.model_config.multi_head_type in ['sim', 'proj_sim']:
            sim_mat = self.sim_head(v_fused, t_fused)  # B,T,C
        else:
            sim_mat = None

        return bin_logits, sim_mat

    def _fuse_and_head(self, vis_feats, txt_feats, v_feat_l):
        """
        Args:
            vis_feats: [B,T,E]
            txt_feats: [C,E]
            v_feat_l: [B]
        """
        bs = vis_feats.shape[0]

        expanded_txt_feat = txt_feats[None, :, :].expand(bs, -1, -1)
        if self.model_config.fusion_type == 'vadclip':
            # binary-class head
            assert self.model_config.bin_head_type == 'vanilla'
            bin_logits = self.bin_head(before_fused=vis_feats)
            # fusion
            t_fused = self.fusion(vis_feats, expanded_txt_feat, v_feat_l, bin_logits)
            v_fused = vis_feats
            # multi-class head
            if self.model_config.multi_head_type in ['sim', 'proj_sim']:
                sim_mat = self.sim_head(v_fused, t_fused)  # B,T,C
            else:
                sim_mat = None
        else:
            # fusion
            if self.model_config.fusion_type in ['co_attn', 'yw', 'asym']:
                v_fused, t_fused = self.fusion(vis_feats, expanded_txt_feat, v_feat_l)
            else:
                v_fused, t_fused = vis_feats, expanded_txt_feat
            # binary-class head
            if self.model_config.bin_head_type == 'vanilla':
                bin_logits = self.bin_head(before_fused=vis_feats)
            elif self.model_config.bin_head_type == 'fused_vanilla':
                bin_logits = self.bin_head(after_fused=v_fused)
            elif self.model_config.bin_head_type == 'adaptive':
                bin_logits = self.bin_head(before_fused=vis_feats, after_fused=v_fused)
            else:
                raise NotImplementedError

            # multi-class head
            if self.model_config.multi_head_type in ['sim', 'proj_sim']:
                sim_mat = self.sim_head(v_fused, t_fused)  # B,T,C
            else:
                sim_mat = None

        return bin_logits, sim_mat

    def forward(self, batch, class_names=None, query_captions=None):
        """
        Args:
            batch:
                v_feat [B,L,E], v_feat_l[B]
            class_names: List of str, encoded with soft prompts
            query_captions: List of str, encoded w/o soft_prompts
        Returns:
            vis_feats: [B,T,E] vision features after temporal encoding
            class_feats: [C,E] CLIP encoded class features
            cls_bin_logits: [B,T] binary logits (w/ class_names)
            cls_sim_mat: [B,T,C] similarity matrix (w/ class_names)
            query_caption_feats: [S,E] CLIP encoded query caption features
            cap_bin_logits: [B,T] binary logits (w/ query_captions)
            cap_sim_mat: [B,T,S] similarity matrix (w/ query_captions)
        """
        bs = batch['v_feat'].shape[0]
        device = self.device
        outputs_dict = {}

        # =======================================================
        # Temporal Encoding
        # =======================================================
        vis_feats, normed_vis_feats = self._temporal_encoding(batch['v_feat'], batch['v_feat_l'])
        outputs_dict['vis_feats'] = vis_feats

        # =======================================================
        # Class Name Forward
        # =======================================================
        if class_names is not None:
            # 1. encode text     ==>  CxD
            labels = class_names
            if self.verbalizer is not None:
                labels = self.verbalizer(labels)
            text_inputs = self.tokenizer(labels, padding=True, truncation=True,
                                         max_length=77 - self.clip_text_model.num_soft_prompts,
                                         return_tensors='pt').to(device)
            text_outputs = self.clip_text_model(
                input_ids=text_inputs.input_ids,
                attention_mask=text_inputs.attention_mask,
            ).pooler_output
            outputs_dict['class_feats'] = text_outputs
            # 2. fuse vision&class_feats and predict logits
            outputs_dict['cls_bin_logits'], outputs_dict['cls_sim_mat'] = self._fuse_and_head(
                vis_feats, text_outputs, batch['v_feat_l']
            )

        # =======================================================
        # Class Name Forward
        # =======================================================
        if query_captions is not None:
            # 1. encode caption text ==> SxD
            ano_caps_inputs = self.tokenizer(query_captions, padding=True, truncation=True,
                                             return_tensors='pt', max_length=77).to(device)
            ano_caps_outputs = self.clip_text_model(
                input_ids=ano_caps_inputs.input_ids,
                attention_mask=ano_caps_inputs.attention_mask,
                use_soft_prompt=False,
            ).pooler_output  # S,D
            outputs_dict['query_caption_feats'] = ano_caps_outputs
            # 2. fuse vision&class_feats and predict logits
            outputs_dict['cap_bin_logits'], outputs_dict['cap_sim_mat'] = self._fuse_and_head(
                vis_feats, ano_caps_outputs, batch['v_feat_l']
            )

        return outputs_dict

    def on_train_epoch_start(self) -> None:
        if self.verbalizer is not None and self.model_config.verbalizer_type == 'dataset-specific':
            self.verbalizer.set_dataset(self.trainer.train_dataloader.dataset.abbr)

    def training_step(self, batch: dict, batch_idx):
        """
        Args:
            batch_idx: int
            batch:
                v_feat [B,L,C], v_feat_l[B]
                cls_label [B] list[str]
                cls_label_idx [B] list[str]
                desc_label [B] list[str]
                pseudo_frame_label [B,L]
                pseudo_span [B,n,2]
        """
        bs = batch['v_feat'].shape[0]
        device = self.device
        loss = 0
        return_loss_dict = {}

        # ================================
        # FORWARD
        # ================================
        if all([i == '' for i in batch['desc_label']]):
            query_captions = None
        else:
            query_captions = ['Normal behavior, views or surveillance content captured by camera that not involve any unexpected or unusual events.']
            query_captions += [i for i in batch['desc_label'] if i != '']
        fd_dict = self(batch, self.train_class_names, query_captions)

        # ================================
        # LOSSES
        # ================================

        # binary MIL
        loss_bin = 0
        if 'cls_bin_logits' in fd_dict:
            loss_bin += mil_loss(
                logits=fd_dict['cls_bin_logits'],
                labels=(batch['cls_label_idx'] != 0).float(),
                lengths=batch['target_length'],
                topk_pct=self.training_config.mil_topk_pct
            )
        if 'cap_bin_logits' in fd_dict:
            loss_bin += mil_loss(
                logits=fd_dict['cap_bin_logits'],
                labels=(batch['cls_label_idx'] != 0).float(),
                lengths=batch['target_length'],
                topk_pct=self.training_config.mil_topk_pct
            )
        loss = loss + loss_bin
        self.log('loss_bin', loss_bin, prog_bar=True)
        return_loss_dict['loss_bin'] = loss_bin

        # binary pseudo supervised loss
        if self.training_config.pseudo_sup_weight > 0.0:
            loss_sup = 0
            if 'cls_bin_logits' in fd_dict:
                loss_sup += supervised_loss(fd_dict['cls_bin_logits'], batch['pseudo_frame_label'],
                                            batch['target_length'])
            if 'cap_bin_logits' in fd_dict:
                loss_sup += supervised_loss(fd_dict['cap_bin_logits'], batch['pseudo_frame_label'],
                                            batch['target_length'])
            loss = loss + loss_sup * self.training_config.pseudo_sup_weight
            self.log('loss_sup', loss_sup, prog_bar=True)
            return_loss_dict['loss_sup'] = loss_sup

        # binary pseudo supervised mil loss
        if self.training_config.pseudo_sup_mil_weight > 0.0:
            loss_sup_mil = 0
            if 'cls_bin_logits' in fd_dict:
                loss_sup_mil += pseudo_sup_mil_loss(
                    fd_dict['cls_bin_logits'], batch['pseudo_frame_label'],
                    batch['target_length'],
                    topk_pct=self.training_config.sup_mil_topk_pct)
            if 'cap_bin_logits' in fd_dict:
                loss_sup_mil += pseudo_sup_mil_loss(
                    fd_dict['cap_bin_logits'], batch['pseudo_frame_label'],
                    batch['target_length'],
                    topk_pct=self.training_config.sup_mil_topk_pct)
            loss = loss + loss_sup_mil * self.training_config.pseudo_sup_mil_weight
            self.log('loss_sup_mil', loss_sup_mil, prog_bar=True)
            return_loss_dict['loss_sup_mil'] = loss_sup_mil

        # multi-class MIL
        loss_mul_mil = 0
        if self.training_config.mul_mil_type == 'v1':
            mul_loss_func = multi_class_mil_loss
        elif self.training_config.mul_mil_type == 'v2':
            mul_loss_func = multi_class_mil_loss_v2
        else:
            raise NotImplementedError
        if 'cls_sim_mat' in fd_dict:
            loss_mul_mil += mul_loss_func(
                fd_dict['cls_sim_mat'], batch['cls_label_idx'], batch['v_feat_l'],
                topk_pct=self.training_config.mul_mil_topk_pct
            )
        if 'cap_sim_mat' in fd_dict:
            cap_labels = []
            ano_cnt = 1
            for i in range(bs):
                if batch['cls_label_idx'][i] == 0:
                    cap_labels.append(0)  # 0 is normal
                else:
                    cap_labels.append(ano_cnt)  # add corresponding caption index
                    ano_cnt += 1
            cap_labels = torch.tensor(cap_labels, device=device)
            loss_mul_mil += mul_loss_func(
                fd_dict['cap_sim_mat'], cap_labels, batch['v_feat_l'],
                topk_pct=self.training_config.mul_mil_topk_pct
            )
        loss = loss + loss_mul_mil * self.training_config.mul_weight
        self.log('loss_mul_mil', loss_mul_mil, prog_bar=True)
        return_loss_dict['loss_mul_mil'] = loss_mul_mil

        # asymmetric_infonce_loss
        if self.training_config.anomaly_cap_contrastive_weight > 0.0:
            cap_contrast_loss = CapContrastLoss(
                self.training_config.contrastive_neg_mining,
                self.training_config.contrastive_temp
            )
            loss_anomaly_cap_contrastive = cap_contrast_loss(
                fd_dict['cls_bin_logits'],
                batch['v_feat_l'],
                fd_dict['vis_feats'],
                fd_dict['query_caption_feats'][1:],
                batch['cls_label_idx'],
                batch['pseudo_frame_label'],
            )

            loss = loss + loss_anomaly_cap_contrastive * self.training_config.anomaly_cap_contrastive_weight
            self.log('loss_anomaly_cap_contrastive', loss_anomaly_cap_contrastive, prog_bar=False)
            return_loss_dict['loss_anomaly_cap_contrastive'] = loss_anomaly_cap_contrastive

        self.log('loss', loss, prog_bar=True)
        if hasattr(self, 'gate_alpha'):
            self.log('gate_alpha', self.gate_alpha, prog_bar=False)
        if hasattr(self, 'adp_alpha'):
            self.log('adp_alpha', self.adp_alpha, prog_bar=False)
        return_loss_dict['loss'] = loss

        return return_loss_dict

    def on_validation_epoch_start(self) -> None:
        if self.metrics is None:
            self.metrics = []
            num_val_dataloaders = 1 if type(self.trainer.val_dataloaders) is not list else len(
                self.trainer.val_dataloaders)
            for i in range(num_val_dataloaders):
                metrics = {
                    'auroc': torchmetrics.AUROC(task='binary').to(self.device),
                    'avg_precision': torchmetrics.AveragePrecision(task='binary').to(self.device),
                    'confusion_mat': torchmetrics.ConfusionMatrix(task='binary', num_classes=2).to(self.device),
                }
                if self.model_config.multi_head_type is not None and len(self.test_class_names[i]) > 2:
                    metrics.update({
                        'top1_acc': torchmetrics.Accuracy(task='multiclass', num_classes=len(self.test_class_names[i]),
                                                          top_k=1).to(self.device),
                        'top2_acc': torchmetrics.Accuracy(task='multiclass', num_classes=len(self.test_class_names[i]),
                                                          top_k=2).to(self.device),
                        'top3_acc': torchmetrics.Accuracy(task='multiclass', num_classes=len(self.test_class_names[i]),
                                                          top_k=3).to(self.device),
                        'A_top1_acc': torchmetrics.Accuracy(task='multiclass',
                                                            num_classes=len(self.test_class_names[i]) - 1,
                                                            top_k=1).to(self.device),
                    })
                self.metrics.append(metrics)
        else:
            for metrics in self.metrics:
                for met in metrics.values():
                    met.reset()

        self.pred_results = []

    def on_validation_batch_start(self, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> None:
        if self.verbalizer is not None and self.model_config.verbalizer_type == 'dataset-specific':
            if type(self.trainer.val_dataloaders) is list:
                dl = self.trainer.val_dataloaders[dataloader_idx]
            else:
                dl = self.trainer.val_dataloaders
            self.verbalizer.set_dataset(dl.dataset.abbr)

    def validation_step(self, batch: dict, batch_idx, dataloader_idx=0):
        """
        Args:
            dataloader_idx:
            batch_idx:
            batch:
                v_feat numpy[B,L,C], v_feat_l[B]
                labels [B,C] list[list[str]]
                label: [B]
                temp_anno: 1,4
                target_length: [B]
        """
        bs = batch['v_feat'].shape[0]
        device = self.device

        if self.model_config.tta is False:
            fd_dict = self(batch, self.test_class_names[dataloader_idx])
        else:
            fd_dict_list = []
            for _ in range(self.model_config.tta_num_forwards):
                fd_dict_list.append(self(batch, self.test_class_names[dataloader_idx]))
            fd_dict = {}
            for k in fd_dict_list[0].keys():
                fd_dict[k] = torch.stack([fd[k] for fd in fd_dict_list], dim=0).mean(0)

        # ================================
        # Post-Process
        # ================================
        scores, mul_softmax_probs, mul_softmax_probs_only_ano = self.post_process(batch['cls_label_idx'], fd_dict)

        # ================================
        # Metrics
        # ================================
        self.submit_metrics(
            batch,
            scores, mul_softmax_probs, mul_softmax_probs_only_ano,
            dataloader_idx=dataloader_idx
        )

    def on_validation_epoch_end(self) -> None:
        for i in range(len(self.metrics)):
            auroc = self.metrics[i]['auroc'].compute()
            bcm = self.metrics[i]['confusion_mat'].compute()
            avg_precision = self.metrics[i]['avg_precision'].compute()
            far = bcm[0, 1] / (bcm[0, 1] + bcm[0, 0])
            suffix = '' if i == 0 else f'_{i}'
            prog_bar = True if i == 0 else False
            self.log(f'Valid/AUROC{suffix}', auroc, prog_bar=prog_bar, sync_dist=True)
            self.log(f'Valid/AP{suffix}', avg_precision, prog_bar=prog_bar, sync_dist=True)
            self.log(f'Valid/FAR{suffix}', far, prog_bar=prog_bar, sync_dist=True)
            if 'top1_acc' in self.metrics[i]:
                top1_acc = self.metrics[i]['top1_acc'].compute()
                top2_acc = self.metrics[i]['top2_acc'].compute()
                top3_acc = self.metrics[i]['top3_acc'].compute()
                self.log(f'Valid/top1_acc{suffix}', top1_acc, prog_bar=False, sync_dist=True)
                self.log(f'Valid/top2_acc{suffix}', top2_acc, prog_bar=False, sync_dist=True)
                self.log(f'Valid/top3_acc{suffix}', top3_acc, prog_bar=False, sync_dist=True)
                A_top1_acc = self.metrics[i]['A_top1_acc'].compute()
                self.log(f'Valid/A_top1_acc{suffix}', A_top1_acc, prog_bar=False, sync_dist=True)

        # below is for visualization
        if self.trainer.global_rank == 0:
            if self.training_config.save_res_mode == 'random':
                pred_results_by_dataset = defaultdict(list)
                log_img_paths, log_img_captions = [], []
                for pred_res in self.pred_results:
                    pred_results_by_dataset[pred_res['dataloader_idx']].append(pred_res)
                for d_idx, d_list in pred_results_by_dataset.items():
                    pred_res = random.choice(d_list)  # one sample to upload per dataset
                    video_name = pred_res['video_path'].stem
                    save_path, vid = save_pred_result(
                        pred_res,
                        f"{video_name}_e{self.current_epoch}_{d_idx}",
                        save_dir=self.log_img_dir,
                    )
                    log_img_paths.append(str(save_path))
                    log_img_captions.append(vid)
                self.trainer.logger.log_image(
                    key="prediction", images=log_img_paths, caption=log_img_captions
                )
            elif self.training_config.save_res_mode == 'fix':
                pred_results_by_dataset = defaultdict(list)
                log_img_paths, log_img_captions = [], []
                for pred_res in self.pred_results:
                    pred_results_by_dataset[pred_res['dataloader_idx']].append(pred_res)
                for d_idx, d_list in pred_results_by_dataset.items():
                    pred_res = d_list[-1]  # one sample to upload per dataset
                    video_name = pred_res['video_path'].stem
                    save_path, vid = save_pred_result(
                        pred_res,
                        f"{video_name}_e{self.current_epoch}_{d_idx}",
                        save_dir=self.log_img_dir,
                    )
                    log_img_paths.append(str(save_path))
                    log_img_captions.append(vid)
                self.trainer.logger.log_image(
                    key="prediction", images=log_img_paths, caption=log_img_captions
                )
            elif self.training_config.save_res_mode == 'fix10':
                pred_results_by_dataset = defaultdict(list)
                log_img_paths, log_img_captions = [], []
                for pred_res in self.pred_results:
                    pred_results_by_dataset[pred_res['dataloader_idx']].append(pred_res)
                for d_idx, d_list in pred_results_by_dataset.items():
                    _d_list = [
                        d_list[i]
                        for i in np.linspace(0, len(d_list) - 1, 10).astype(int).tolist()
                    ]
                    for i, pred_res in enumerate(tqdm(_d_list)):  # all save to local
                        video_name = Path(pred_res['video_path']).stem
                        save_path, vid = save_pred_result(
                            pred_res,
                            f"{video_name}_{i}_e{self.current_epoch}_{d_idx}",
                            save_dir=self.log_img_dir,
                        )
                        if i == 9:
                            log_img_paths.append(str(save_path))
                            log_img_captions.append(vid)
                self.trainer.logger.log_image(
                    key="prediction", images=log_img_paths, caption=log_img_captions
                )
            elif self.training_config.save_res_mode == 'all':
                pred_results_by_dataset = defaultdict(list)
                log_img_paths, log_img_captions = [], []
                for pred_res in self.pred_results:
                    pred_results_by_dataset[pred_res['dataloader_idx']].append(pred_res)
                for d_idx, d_list in pred_results_by_dataset.items():
                    log_idx = random.randrange(len(d_list))  # one sample to upload per dataset
                    for i, pred_res in enumerate(tqdm(d_list)):  # all save to local
                        # print(pred_res['score'].shape, pred_res.get('temp_anno', None))
                        save_path, vid = save_pred_result(pred_res, save_dir=self.log_img_dir)
                        if i == log_idx:
                            log_img_paths.append(str(save_path))
                            log_img_captions.append(vid)
                self.trainer.logger.log_image(
                    key="prediction", images=log_img_paths, caption=log_img_captions
                )
        else:
            if self.training_config.save_res_mode == 'fix10':
                pred_results_by_dataset = defaultdict(list)
                for pred_res in self.pred_results:
                    pred_results_by_dataset[pred_res['dataloader_idx']].append(pred_res)
                for d_idx, d_list in pred_results_by_dataset.items():
                    _d_list = d_list[-10:]
                    for i, pred_res in enumerate(_d_list):  # all save to local
                        video_name = Path(pred_res['video_path']).stem
                        save_path, vid = save_pred_result(
                            pred_res,
                            f"{video_name}_{i}_e{self.current_epoch}_{d_idx}",
                            save_dir=self.log_img_dir,
                        )

    def on_test_epoch_start(self) -> None:
        if self.metrics is None:
            self.metrics = []
            num_test_dataloaders = 1 if type(self.trainer.test_dataloaders) is not list else len(
                self.trainer.test_dataloaders)
            for i in range(num_test_dataloaders):
                metrics = {
                    'auroc': torchmetrics.AUROC(task='binary').to(self.device),
                    'avg_precision': torchmetrics.AveragePrecision(task='binary').to(self.device),
                    'confusion_mat': torchmetrics.ConfusionMatrix(task='binary', num_classes=2).to(self.device),
                }
                if self.model_config.multi_head_type is not None and len(self.test_class_names[i]) > 2:
                    metrics.update({
                        'top1_acc': torchmetrics.Accuracy(task='multiclass', num_classes=len(self.test_class_names[i]),
                                                          top_k=1).to(self.device),
                        'top2_acc': torchmetrics.Accuracy(task='multiclass', num_classes=len(self.test_class_names[i]),
                                                          top_k=2).to(self.device),
                        'top3_acc': torchmetrics.Accuracy(task='multiclass', num_classes=len(self.test_class_names[i]),
                                                          top_k=3).to(self.device),
                        'A_top1_acc': torchmetrics.Accuracy(task='multiclass',
                                                            num_classes=len(self.test_class_names[i]) - 1,
                                                            top_k=1).to(self.device),
                    })
                self.metrics.append(metrics)
        else:
            for metrics in self.metrics:
                for met in metrics.values():
                    met.reset()

        self.pred_results = []

    def on_test_batch_start(self, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> None:
        if self.verbalizer is not None and self.model_config.verbalizer_type == 'dataset-specific':
            if type(self.trainer.test_dataloaders) is list:
                dl = self.trainer.test_dataloaders[dataloader_idx]
            else:
                dl = self.trainer.test_dataloaders
            self.verbalizer.set_dataset(dl.dataset.abbr)

    def test_step(self, batch, batch_idx, dataloader_idx=0):
        """
        same as validation step
        use tta by default
        """
        fd_dict_list = []
        for _ in range(self.model_config.tta_num_forwards):
            fd_dict_list.append(self(batch, self.test_class_names[dataloader_idx]))
        fd_dict = {}
        for k in fd_dict_list[0].keys():
            fd_dict[k] = torch.stack([fd[k] for fd in fd_dict_list], dim=0).mean(0)

        # ================================
        # Post-Process
        # ================================
        scores, mul_softmax_probs, mul_softmax_probs_only_ano = self.post_process(batch['cls_label_idx'], fd_dict)

        # ================================
        # Metrics
        # ================================
        self.submit_metrics(
            batch,
            scores, mul_softmax_probs, mul_softmax_probs_only_ano,
            dataloader_idx=dataloader_idx
        )

    def on_test_epoch_end(self) -> None:
        """same as validation"""
        for i in range(len(self.metrics)):
            auroc = self.metrics[i]['auroc'].compute()
            bcm = self.metrics[i]['confusion_mat'].compute()
            avg_precision = self.metrics[i]['avg_precision'].compute()
            far = bcm[0, 1] / (bcm[0, 1] + bcm[0, 0])
            suffix = '' if i == 0 else f'_{i}'
            prog_bar = True if i == 0 else False
            self.log(f'Test/AUROC{suffix}', auroc, prog_bar=prog_bar, sync_dist=True)
            self.log(f'Test/AP{suffix}', avg_precision, prog_bar=prog_bar, sync_dist=True)
            self.log(f'Test/FAR{suffix}', far, prog_bar=prog_bar, sync_dist=True)
            if 'top1_acc' in self.metrics[i]:
                top1_acc = self.metrics[i]['top1_acc'].compute()
                top2_acc = self.metrics[i]['top2_acc'].compute()
                top3_acc = self.metrics[i]['top3_acc'].compute()
                self.log(f'Test/top1_acc{suffix}', top1_acc, prog_bar=False, sync_dist=True)
                self.log(f'Test/top2_acc{suffix}', top2_acc, prog_bar=False, sync_dist=True)
                self.log(f'Test/top3_acc{suffix}', top3_acc, prog_bar=False, sync_dist=True)
                A_top1_acc = self.metrics[i]['A_top1_acc'].compute()
                self.log(f'Test/A_top1_acc{suffix}', A_top1_acc, prog_bar=False, sync_dist=True)

    def on_predict_batch_start(self, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> None:
        if self.verbalizer is not None and self.model_config.verbalizer_type == 'dataset-specific':
            if type(self.trainer.predict_dataloaders) is list:
                dl = self.trainer.predict_dataloaders[dataloader_idx]
            else:
                dl = self.trainer.predict_dataloaders
            self.verbalizer.set_dataset(dl.dataset.abbr)

    def predict_step(self, batch, batch_idx, dataloader_idx=0) -> Any:
        """
        same as validation step
        use tta by default
        result are saved
        """
        fd_dict_list = []
        for _ in range(self.model_config.tta_num_forwards):
            fd_dict_list.append(self(batch, self.predict_class_names[dataloader_idx]))
        fd_dict = {}
        for k in fd_dict_list[0].keys():
            fd_dict[k] = torch.stack([fd[k] for fd in fd_dict_list], dim=0).mean(0)

        # ================================
        # Post-Process
        # ================================
        scores, mul_softmax_probs, mul_softmax_probs_only_ano = self.post_process(batch['cls_label_idx'], fd_dict)
        mul_scores = fd_dict['cls_sim_mat'].permute(0, 2, 1)  # [B,C,T]

        # ================================
        # Change format and save
        # ================================
        bs = batch['v_feat'].shape[0]
        for bi in range(bs):
            # remove pad or interpolate to full len
            tgt_len = batch['target_length'][bi]
            if len(scores[bi]) >= tgt_len:  # pad
                real_score = scores[bi][:tgt_len]
                real_mul_scores = mul_scores[bi][:, :tgt_len]
            else:  # resample  B,C,T
                real_score = F.interpolate(scores[bi][None, None, :], tgt_len, mode='linear')[0, 0]  # L'
                real_mul_scores = F.interpolate(mul_scores[bi][None, :], tgt_len, mode='linear')[0]  # C,L'
            # clip-level to frame-level score
            real_score = torch.repeat_interleave(real_score, self.frame_time[dataloader_idx], dim=0)  # L
            real_mul_scores = torch.repeat_interleave(real_mul_scores, self.frame_time[dataloader_idx], dim=1)  # C,L
            # save prediction result
            self.pred_results.append({
                'video_path': batch['video_path'][bi],
                'score': real_score.cpu().numpy(),
                'real_mul_scores': real_mul_scores.cpu().numpy(),
                'dataloader_idx': dataloader_idx,
                'cls_bin_logits': fd_dict['cls_bin_logits'],
                'cls_sim_mat': fd_dict['cls_sim_mat'],
            })
