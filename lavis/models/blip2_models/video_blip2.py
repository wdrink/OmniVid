"""
 Copyright (c) 2023, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""
import logging
import einops
from addict import Dict
import math
import numpy as np

import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F

from torch.cuda.amp import autocast as autocast

from od_util.box_ops import (
    bbox_xyxy_to_cxcyah,
    bbox_cxcyah_to_xyxy,
    generalized_box_iou,
)
from lavis.common.registry import registry
from lavis.models.blip2_models.blip2 import Blip2Base, disabled_train
from lavis.models.blip2_models.temporal_transformer import VisionTransformer
from lavis.models.blip2_models.Qformer import BertConfig, BertLMHeadModel
from lavis.models.fusion_models.transformer_encoder import (
    build_transformer_encoder,
)
from lavis.models.fusion_models.vlfusion import VLFuse
from lavis.models.fusion_models.position_encoding import build_position_encoding
from lavis.models.blip2_models.ms_transformer import TransformerBlock
from lavis.models.actionformer_models.models import (
    make_neck,
    make_generator,
)
from lavis.models.actionformer_models.meta_archs import (
    PtTransformerClsHead,
    PtTransformerRegHead,
)
from lavis.models.actionformer_models.losses import (
    ctr_diou_loss_1d,
    sigmoid_focal_loss,
)

from lavis.models.blip2_models.vggish import VGGish
from timm.models.layers import trunc_normal_
from timm.loss import LabelSmoothingCrossEntropy

from transformers import (
    BartConfig,
    T5Config,
    T5ForConditionalGeneration,
)

from .modeling_bart import BartForConditionalGeneration


@registry.register_model("video_blip2")
class VideoBlip2(Blip2Base):
    PRETRAINED_MODEL_CONFIG_DICT = {
        "omnicaption_flant5xl": "configs/models/videoblip2/videoblip2_caption_flant5xl.yaml",
        "omnicaption_bartbase": "configs/models/videoblip2/videoblip2_caption_bartbase.yaml",
    }

    def __init__(
        self,
        vit_model="eva_clip_g",
        img_size=224,
        num_frms=100,
        temporal_downrate=1,
        temporal_depth=12,
        drop_path_rate=0,
        use_grad_checkpoint=False,
        vit_precision="fp16",
        freeze_vit=True,
        freeze_qformer=False,
        freeze_llm=False,
        tune_llm_fc=False,
        freeze_audio=False,
        num_query_token=32,
        llm_model="facebook/bart-base",
        prompt="",
        max_txt_len=32,
        apply_lemmatizer=False,
        vocab_size=50265,
        roi_size=7,
        cross_frame_fusion="ealy_full",
        weight_token_loss=False,
        use_iou_loss=False,
        iou_loss_weight=1.0,
        box_in_prompt=True,
        language_in_prompt=False,
        box_in_query=False,
        language_in_query=False,
        score_threshold=0.0,
        box2query="add",
        template_size=128,
        template_boundary=0,
        use_template_feat=False,
        use_frame_mask=False,
        use_audio=False,
        fuse_audio="concat",
        init_3d_patch_embed=True,
        init_2d_patch_embed=False,
        video_backbone_pretrained="k400",
        nollminit=False,
        vl_fuse=False,
        vl_fuse_layer=1,
    ):
        """
        apply_lemmatizer: when set to True, postprocess predict_answers() result with lemmas.
        """
        super().__init__()

        self.visual_encoder, self.ln_vision = self.init_vision_encoder(
            vit_model,
            img_size,
            drop_path_rate,
            use_grad_checkpoint,
            vit_precision,
            init_3d_patch_embed=init_3d_patch_embed,
            init_2d_patch_embed=init_2d_patch_embed,
            pretrained=video_backbone_pretrained,
        )

        # print(self.visual_encoder)
        if freeze_vit:
            for name, param in self.visual_encoder.named_parameters():
                param.requires_grad = False
            self.visual_encoder.eval()
            self.visual_encoder.train = disabled_train
            logging.info("freeze vision encoder")
            print("freeze vision encoder")

        else:
            print("finetune vision encoder")

        self.Qformer, self.query_tokens = self.init_Qformer(
            num_query_token, self.visual_encoder.num_features
        )
        self.num_query_token = num_query_token
        self.Qformer.cls = None
        self.Qformer.bert.embeddings.word_embeddings = None
        self.Qformer.bert.embeddings.position_embeddings = None
        for layer in self.Qformer.bert.encoder.layer:
            layer.output = None
            layer.intermediate = None

        if freeze_qformer:
            for name, param in self.Qformer.named_parameters():
                param.requires_grad = False

            self.Qformer = self.Qformer.eval()
            self.Qformer.train = disabled_train
            logging.info("freeze Qformer")
            print("freeze Qformer")

        else:
            print("finetune Qformer")

        if "t5" in llm_model:
            model_config = T5Config.from_pretrained(llm_model)
            model_config.dense_act_fn = "gelu"
            self.llm = T5ForConditionalGeneration.from_pretrained(
                llm_model, config=model_config
            )

        elif "bart" in llm_model:
            model_config = BartConfig.from_pretrained(llm_model)
            if not nollminit:
                self.llm = BartForConditionalGeneration.from_pretrained(
                    llm_model,
                    config=model_config,
                    use_iou_loss=use_iou_loss,
                    iou_loss_weight=iou_loss_weight,
                )
            else:
                self.llm = BartForConditionalGeneration(
                    model_config,
                    use_iou_loss=use_iou_loss,
                    iou_loss_weight=iou_loss_weight,
                )

        self.use_iou_loss = use_iou_loss
        self.llm.resize_token_embeddings(vocab_size)
        self.roi_size = roi_size

        if freeze_llm:
            for name, param in self.llm.named_parameters():
                if tune_llm_fc and "attn" not in name:
                    param.requires_grad = True

                else:
                    param.requires_grad = False
                    if "t5" in llm_model:
                        param.data = param.data.bfloat16()
                    else:
                        param.data = param.data.half()

            logging.info("freeze llm")
            print("freeze llm")

        else:
            print("finetune llm")

        self.llm_proj = nn.Linear(
            self.Qformer.config.hidden_size, self.llm.config.hidden_size
        )

        if cross_frame_fusion == "early_full":
            self.bottleneck = nn.Sequential(
                nn.Conv2d(
                    self.visual_encoder.num_features,
                    self.visual_encoder.num_features,
                    kernel_size=1,
                ),
                nn.GroupNorm(32, self.visual_encoder.num_features),
            )
            self.fusion_transformer = build_transformer_encoder(
                d_model=self.visual_encoder.num_features
            )
            self.fusion_pos_embed = build_position_encoding(
                hidden_dim=self.visual_encoder.num_features
            )

        elif cross_frame_fusion == "late":
            num_clips = num_frms
            assert (
                num_clips % temporal_downrate == 0
            ), "num_clips %d must be divisible by temporal_downrate %d" % (
                num_clips,
                temporal_downrate,
            )

            if "videoswin" in self.vit_name:
                num_clips //= self.visual_encoder.patch_embed.patch_size[0]

            num_clips //= temporal_downrate
            self.temporal_encoder = VisionTransformer(
                num_frms=num_clips,
                temporal_downrate=temporal_downrate,
                num_queries=num_query_token
                if box2query == "add"
                else num_query_token + 25,
                num_heads=32,
                embed_dim=self.llm.config.hidden_size,
                depth=temporal_depth,
            )

        else:
            assert (
                cross_frame_fusion == "none"
            ), "we only support early_full, late, and none for frame fusion"
            num_clips = num_frms
            assert (
                num_clips % temporal_downrate == 0
            ), "num_clips %d must be divisible by temporal_downrate %d" % (
                num_clips,
                temporal_downrate,
            )

            if "videoswin" in self.vit_name:
                num_clips //= self.visual_encoder.patch_embed.patch_size[0]

            num_clips //= temporal_downrate

            self.clip_pos_embed = nn.Parameter(
                torch.randn(1, num_clips, self.llm.config.hidden_size) * 0.02
            )
            trunc_normal_(self.clip_pos_embed, std=0.02)

        self.cross_frame_fusion = cross_frame_fusion
        self.num_frms = num_frms

        if vl_fuse:
            t2v_encoder = VLFuse(
                self.llm.config.hidden_size,
                self.llm.config.hidden_size,
                n_layers=vl_fuse_layer,
                embed_dim=2048,
                n_head=8,
            )
            self.t2v_encoder = t2v_encoder

        else:
            self.t2v_encoder = None

        self.vl_fuse = vl_fuse

        # inference config
        self.max_txt_len = max_txt_len
        self.prompt = prompt

        self._apply_lemmatizer = apply_lemmatizer
        self._lemmatizer = None
        self.use_frame_mask = use_frame_mask

        if use_audio:
            audio_vggish = VGGish()
            audio_vggish.load_state_dict(torch.load("./pytorch_vggish.pth"))
            self.audio_encoder = audio_vggish

            if freeze_audio:
                for name, param in self.audio_encoder.named_parameters():
                    param.requires_grad = False

            self.audio_encoder = self.audio_encoder.eval()
            self.proj_audio = nn.Linear(
                128, self.llm.config.hidden_size, bias=False
            )
            self.proj_audio_norm = nn.LayerNorm(self.llm.config.hidden_size)

            # init the proj_audio and proj_audio_norm
            trunc_normal_(self.proj_audio.weight, std=0.02)
            nn.init.constant_(self.proj_audio_norm.bias, 0)
            nn.init.constant_(self.proj_audio_norm.weight, 1.0)

        self.use_audio = use_audio
        self.fuse_audio = fuse_audio
        self.weight_token_loss = weight_token_loss

        self.box_in_prompt = box_in_prompt
        self.language_in_prompt = language_in_prompt

        if box_in_query:
            self.pos_trans = nn.Linear(
                self.query_tokens.shape[-1], self.query_tokens.shape[-1]
            )
            self.pos_trans_norm = nn.LayerNorm(self.query_tokens.shape[-1])

            # init the pos_trans and pos_trans_norm
            trunc_normal_(self.pos_trans.weight, std=0.02)
            nn.init.constant_(self.pos_trans.bias, 0)

            nn.init.constant_(self.pos_trans_norm.bias, 0)
            nn.init.constant_(self.pos_trans_norm.weight, 1.0)

        if language_in_query:
            self.text2query = nn.Linear(
                self.llm.config.hidden_size, self.query_tokens.shape[-1]
            )
            self.text2query_norm = nn.LayerNorm(self.query_tokens.shape[-1])

            # init the text2query and text2query_norm
            trunc_normal_(self.text2query.weight, std=0.02)
            nn.init.constant_(self.text2query.bias, 0)

            nn.init.constant_(self.text2query_norm.bias, 0)
            nn.init.constant_(self.text2query_norm.weight, 1.0)

        self.box_in_query = box_in_query
        self.language_in_query = language_in_query
        self.score_threshold = score_threshold
        self.box2query = box2query
        self.template_size = template_size
        self.template_boundary = template_boundary
        self.use_template_feat = use_template_feat

    @classmethod
    def init_video_Qformer(
        cls, num_query_token, vision_width, num_hidden_layers=2
    ):
        encoder_config = BertConfig.from_pretrained("bert-base-uncased")
        encoder_config.num_hidden_layers = num_hidden_layers
        encoder_config.encoder_width = vision_width
        # insert cross-attention layer every other block
        encoder_config.add_cross_attention = True
        encoder_config.cross_attention_freq = 1
        encoder_config.query_length = num_query_token
        Qformer = BertLMHeadModel(config=encoder_config)
        query_tokens = nn.Parameter(
            torch.zeros(1, num_query_token, encoder_config.hidden_size)
        )
        query_tokens.data.normal_(
            mean=0.0, std=encoder_config.initializer_range
        )
        return Qformer, query_tokens

    def get_proposal_pos_embed2(
        self, proposals, query_hidden_size, height, width
    ):
        num_pos_feats = query_hidden_size / 4
        temperature = 10000
        scale = 2 * math.pi

        dim_t = torch.arange(
            num_pos_feats, dtype=torch.float32, device=proposals.device
        )
        dim_t = temperature ** (
            2 * (torch.div(dim_t, 2, rounding_mode="trunc")) / num_pos_feats
        )

        proposals = proposals.unsqueeze(1).repeat(1, 25, 1)

        # shift the proposals (1, L, 4) to different directions
        shift = torch.tensor(
            [
                [0, 0, 0, 0],
                [0, 0, 0, 1],
                [0, 0, 0, 2],
                [0, 0, 0, 3],
                [0, 0, 1, 0],
                [0, 0, 2, 0],
                [0, 0, 3, 0],
                [0, 0, -1, 0],
                [0, 0, -2, 0],
                [0, 0, -3, 0],
                [0, 0, 0, -1],
                [0, 0, 0, -2],
                [0, 0, 0, -3],
                [0, 0, 1, 1],
                [0, 0, 2, 2],
                [0, 0, 3, 3],
                [0, 0, -1, -1],
                [0, 0, -2, -2],
                [0, 0, -3, -3],
                [0, 0, 1, -1],
                [0, 0, 2, -2],
                [0, 0, 3, -3],
                [0, 0, -1, 1],
                [0, 0, -2, 2],
                [0, 0, -3, 3],
            ],
            dtype=torch.float32,
            device=proposals.device,
        )
        proposals = proposals + shift.unsqueeze(0)
        proposals[:, :, 0::2] /= width
        proposals[:, :, 1::2] /= height

        # batch size, L, 4
        proposals = proposals.sigmoid() * scale
        # batch size, L, 4, hidden_size / 4
        pos = proposals[:, :, :, None] / dim_t
        # batch size, L, 1, hidden_size
        pos = torch.stack(
            (pos[:, :, :, 0::2].sin(), pos[:, :, :, 1::2].cos()), dim=4
        ).flatten(2)
        pos = pos.view(pos.shape[0], 25, -1).float()
        reference_point_embed = self.pos_trans_norm(self.pos_trans(pos))
        return reference_point_embed

    def get_proposal_pos_embed(self, proposals, query_hidden_size):
        num_pos_feats = query_hidden_size / proposals.shape[-1]
        temperature = 10000
        scale = 2 * math.pi

        dim_t = torch.arange(
            num_pos_feats, dtype=torch.float32, device=proposals.device
        )
        dim_t = temperature ** (
            2 * (torch.div(dim_t, 2, rounding_mode="trunc")) / num_pos_feats
        )

        # batch size, 4
        proposals = proposals.sigmoid() * scale
        # batch size, 4, hidden_size / 4
        pos = proposals[:, :, None] / dim_t
        # batch size, 1, hidden_size
        pos = torch.stack(
            (pos[:, :, 0::2].sin(), pos[:, :, 1::2].cos()), dim=3
        ).flatten(2)
        pos = pos.view(pos.shape[0], 1, -1).float()
        reference_point_embed = self.pos_trans_norm(self.pos_trans(pos))
        return reference_point_embed

    def encode_video(self, q_hidden_state):
        with self.maybe_autocast():
            # add frame_pos embedding
            batch_size, time_length, _, _ = q_hidden_state.size()  # B T N C
            position_ids = torch.arange(
                time_length, dtype=torch.long, device=q_hidden_state.device
            )
            position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)
            frame_position_embeddings = self.video_frame_position_embedding(
                position_ids
            )

            frame_position_embeddings = frame_position_embeddings.unsqueeze(-2)
            frame_hidden_state = frame_position_embeddings + q_hidden_state
            # frame attention
            frame_hidden_state = einops.rearrange(
                frame_hidden_state,
                "b t q h -> b (t q) h",
                b=batch_size,
                t=time_length,
            )
            frame_atts = torch.ones(
                frame_hidden_state.size()[:-1], dtype=torch.long
            ).to(q_hidden_state.device)
            video_query_tokens = self.video_query_tokens.expand(
                frame_hidden_state.shape[0], -1, -1
            )

            video_query_output = self.video_Qformer.bert(
                query_embeds=video_query_tokens,
                encoder_hidden_states=frame_hidden_state,
                encoder_attention_mask=frame_atts,
                return_dict=True,
            )
            video_hidden = video_query_output.last_hidden_state

            video_tokens = self.llm_proj(video_hidden)
            video_att_mask = torch.ones(
                video_tokens.size()[:-1], dtype=torch.long
            ).to(video_tokens.device)
        return video_tokens, video_att_mask

    def set_tokenizer(self, tokenizer):
        self.tokenizer = tokenizer

    def crop_image_from_boxes(
        self, ref_images, boxes, template_size, box_boundary=0
    ):
        # crop the template from the reference image using prompt_box,
        # and resize the template to the same size as the query image
        # ref_images: B, C, H, W
        # boxes: B, 4
        # template_size: (H, W)
        B = ref_images.shape[0]
        cropped_images = []
        for i in range(B):
            ref_img = ref_images[i]  # C, H, W
            H, W = ref_img.shape[1:]
            ref_box = boxes[i].tolist()  # 4

            x1, y1, x2, y2 = ref_box
            w, h = x2 - x1, y2 - y1
            x1 = int(x1 - box_boundary * w)
            y1 = int(y1 - box_boundary * h)
            x2 = int(x2 + box_boundary * w)
            y2 = int(y2 + box_boundary * h)

            if x1 == x2:
                x1 -= 1
                x2 += 1
            if y1 == y2:
                y1 -= 1
                y2 += 1

            x1 = max(x1, 0)
            y1 = max(y1, 0)
            x2 = min(x2, W)
            y2 = min(y2, H)

            cropped_img = ref_img[:, y1:y2, x1:x2]
            cropped_img = F.interpolate(
                cropped_img.unsqueeze(0),
                size=template_size,
                mode="bicubic",
                align_corners=True,
            ).squeeze(0)

            cropped_images.append(cropped_img)

        cropped_images = torch.stack(cropped_images, dim=0)
        return cropped_images

    def forward_sot(self, samples):
        B, T = samples["video"].shape[:2]
        assert T == 2, "only support 2 frames for SOT now"
        image = samples["video"].view(-1, *samples["video"].shape[2:])
        mask = samples["image_mask"].view(-1, *samples["image_mask"].shape[2:])
        prompt_box = samples["prompt_box"]

        image = image.view(B, T, *image.shape[1:])
        ref_image = image[:, 0]
        tgt_image = image[:, 1]
        # crop the template from the reference image using prompt_box
        ref_image = self.crop_image_from_boxes(
            ref_image,
            prompt_box,
            self.template_size,
            self.template_boundary,
        )

        with self.maybe_autocast():
            input_tokens = samples["prompt"]
            output_tokens = samples["text"]

            input_tokens = Dict(
                {k: v.to(image.device) for k, v in input_tokens.items()}
            )
            output_tokens = Dict(
                {k: v.to(image.device) for k, v in output_tokens.items()}
            )

            targets = output_tokens.input_ids.masked_fill(
                output_tokens.input_ids
                == self.tokenizer.tokenizer.pad_token_id,
                -100,
            )

            if hasattr(self.llm, "encoder"):
                inputs_embeds = self.llm.encoder.embed_tokens(
                    input_tokens.input_ids
                )

            else:
                inputs_embeds = self.llm.model.encoder.embed_tokens(
                    input_tokens.input_ids
                )

            if "videoswin" not in self.vit_name:
                image_shape = (
                    image.shape[3]
                    // self.visual_encoder.patch_embed.patch_size[0],
                    image.shape[4]
                    // self.visual_encoder.patch_embed.patch_size[1],
                )

                ref_feats = self.visual_encoder(ref_image)
                ref_embeds = self.ln_vision(ref_feats)

                ref_image_atts = (
                    torch.ones(ref_embeds.shape[0], ref_embeds.shape[1])
                    .to(torch.long)
                    .to(ref_embeds.device)
                )

                tgt_feats = self.visual_encoder(tgt_image)
                tgt_embeds = self.ln_vision(tgt_feats)

                tgt_mask = mask.view(B, T, *mask.shape[1:])[:, 1]
                tgt_image_atts = (
                    F.interpolate(
                        (~tgt_mask).unsqueeze(1).float(), size=image_shape
                    )
                    .to(torch.long)[:, 0]
                    .view(tgt_embeds.shape[0], -1)
                )
                tgt_cls_atts = (
                    torch.tensor([[True]])
                    .to(tgt_image_atts.device)
                    .repeat(tgt_image_atts.shape[0], 1)
                )
                tgt_image_atts = torch.cat(
                    [tgt_cls_atts, tgt_image_atts], dim=1
                )

            else:
                image_shape = (
                    math.ceil(image.shape[3] / 32),
                    math.ceil(image.shape[4] / 32),
                )

                ref_image = F.interpolate(
                    ref_image, size=tgt_image.shape[2:], mode="bicubic"
                )
                ref_tgt_image = torch.stack(
                    [ref_image, tgt_image], dim=1
                ).transpose(1, 2)
                ref_tgt_feats = self.visual_encoder(ref_tgt_image)

                ref_tgt_embeds = self.ln_vision(ref_tgt_feats)
                _, THW, _ = ref_tgt_embeds.shape
                HW = THW // T
                ref_tgt_embeds = ref_tgt_embeds.view(B, T, HW, -1)

                ref_embeds = ref_tgt_embeds[:, 0]
                tgt_embeds = ref_tgt_embeds[:, 1]

                ref_image_atts = (
                    torch.ones(ref_embeds.shape[0], ref_embeds.shape[1])
                    .to(torch.long)
                    .to(ref_embeds.device)
                )

                tgt_mask = mask.view(B, T, *mask.shape[1:])[:, 1]
                tgt_image_atts = (
                    F.interpolate(
                        (~tgt_mask).unsqueeze(1).float(), size=image_shape
                    )
                    .to(torch.long)[:, 0]
                    .view(tgt_embeds.shape[0], -1)
                )

        if self.cross_frame_fusion == "early_full":
            if "videoswin" not in self.vit_name:
                ref_cls_feats = ref_embeds[:, 0:1]
                ref_image_embeds = ref_embeds[:, 1:]
                ref_image_embeds_h, ref_image_embeds_w = int(
                    math.sqrt(ref_image_embeds.shape[1])
                )

                tgt_cls_feats = tgt_embeds[:, 0:1]
                tgt_image_embeds = tgt_embeds[:, 1:]

            else:
                ref_image_embeds = ref_embeds
                ref_image_embeds_h, ref_image_embeds_w = image_shape
                tgt_image_embeds = tgt_embeds

                ref_cls_feats, tgt_cls_feats = None, None

            feat1 = (
                ref_image_embeds.view(
                    B, ref_image_embeds_h, ref_image_embeds_w, -1
                )
                .permute(0, 3, 1, 2)
                .contiguous()
            )
            feat2 = (
                tgt_image_embeds.view(B, image_shape[0], image_shape[1], -1)
                .permute(0, 3, 1, 2)
                .contiguous()
            )
            srcs = [self.bottleneck(feat1), self.bottleneck(feat2)]

            pos1 = self.fusion_pos_embed(
                B, ref_image_embeds_h, ref_image_embeds_w
            )

            pos1 = F.interpolate(
                pos1,
                size=(ref_image_embeds_h, ref_image_embeds_w),
                mode="bicubic",
            )

            pos2 = self.fusion_pos_embed(B, image_shape[0], image_shape[1])
            pos2 = F.interpolate(
                pos2,
                size=(image_shape[0], image_shape[1]),
                mode="bicubic",
            )

            pos = [pos1, pos2]
            srcs_new = [
                x.flatten(-2).permute((2, 0, 1)).contiguous() for x in srcs
            ]
            pos_new = [
                x.flatten(-2).permute((2, 0, 1)).contiguous() for x in pos
            ]

            out_seq = self.fusion_transformer(
                torch.cat(srcs_new, dim=0),
                mask=None,
                pos_embed=torch.cat(pos_new, dim=0),
            )
            # Split the sequence and reshape back to feature maps
            half_len = ref_image_embeds_h * ref_image_embeds_w
            seq1, seq2 = (
                out_seq[:half_len],
                out_seq[half_len:],
            )  # h1w1, B, C, h2w2, B, C

            if ref_cls_feats is not None:
                ref_embeds = torch.cat(
                    [ref_cls_feats, seq1.permute(1, 0, 2)], dim=1
                )
            else:
                ref_embeds = seq1.permute(1, 0, 2)

            if tgt_cls_feats is not None:
                tgt_embeds = torch.cat(
                    [tgt_cls_feats, seq2.permute(1, 0, 2)], dim=1
                )
            else:
                tgt_embeds = seq2.permute(1, 0, 2)

            query_tokens = self.query_tokens.expand(ref_embeds.shape[0], -1, -1)

        elif self.cross_frame_fusion == "late":
            query_tokens = self.query_tokens.expand(ref_embeds.shape[0], -1, -1)
            ref_query_output = self.Qformer.bert(
                query_embeds=query_tokens,
                encoder_hidden_states=ref_embeds,
                encoder_attention_mask=ref_image_atts,
                return_dict=True,
            )
            ref_hidden = ref_query_output.last_hidden_state

        if self.box_in_query:
            if self.box2query == "add":
                normalized_prompt_box = prompt_box.clone()
                normalized_prompt_box[:, 0::2] /= samples["image_sizes"][0][0][
                    1
                ]
                normalized_prompt_box[:, 1::2] /= samples["image_sizes"][0][0][
                    0
                ]

                prompt_to_reference_embeddings = self.get_proposal_pos_embed(
                    normalized_prompt_box, self.query_tokens.shape[-1]
                )
                prompt_to_reference_embeddings = (
                    prompt_to_reference_embeddings.expand(
                        tgt_embeds.shape[0], -1, -1
                    )
                )
                query_tokens = query_tokens + prompt_to_reference_embeddings

            elif self.box2query == "concat":
                prompt_to_reference_embeddings = self.get_proposal_pos_embed2(
                    prompt_box,
                    self.query_tokens.shape[-1],
                    height=samples["image_sizes"][0][0][0],
                    width=samples["image_sizes"][0][0][1],
                )
                prompt_to_reference_embeddings = (
                    prompt_to_reference_embeddings.expand(
                        tgt_embeds.shape[0], -1, -1
                    )
                )
                query_tokens = torch.cat(
                    [query_tokens, prompt_to_reference_embeddings], dim=1
                )
            else:
                raise NotImplementedError

        if self.language_in_query:
            inputs_embeds_mean = inputs_embeds.mean(dim=1, keepdim=True)
            language_queries = self.text2query_norm(
                self.text2query(inputs_embeds_mean)
            )
            query_tokens = query_tokens + language_queries

        tgt_query_output = self.Qformer.bert(
            query_embeds=query_tokens,
            encoder_hidden_states=tgt_embeds,
            encoder_attention_mask=tgt_image_atts,
            return_dict=True,
        )
        tgt_hidden = tgt_query_output.last_hidden_state

        if self.use_template_feat:
            ref_query_output = self.Qformer.bert(
                query_embeds=query_tokens,
                encoder_hidden_states=ref_embeds,
                encoder_attention_mask=ref_image_atts,
                return_dict=True,
            )
            ref_hidden = ref_query_output.last_hidden_state  # B, L, C
            tgt_hidden = torch.cat([ref_hidden, tgt_hidden], dim=1)

        if self.cross_frame_fusion == "early_full":
            inputs_llm = self.llm_proj(tgt_hidden)
            inputs_llm = inputs_llm.view(B, -1, inputs_llm.shape[-1])

        else:
            cat_visual_hidden = torch.cat([ref_hidden, tgt_hidden], dim=1)
            inputs_llm = self.llm_proj(cat_visual_hidden)

            if self.box2query == "add":
                inputs_llm = inputs_llm.view(B, T, -1, inputs_llm.shape[-1])
            else:
                inputs_llm = inputs_llm.view(B, -1, inputs_llm.shape[-1])

            inputs_llm = self.temporal_encoder(inputs_llm)

        # concat the image embeddings to Qformer output
        if self.cross_frame_fusion == "none":
            projected_image_embeds = self.prompt_proj(tgt_embeds)
            inputs_llm = torch.cat([inputs_llm, projected_image_embeds], dim=1)

        atts_llm = torch.ones(inputs_llm.size()[:-1], dtype=torch.long).to(
            image.device
        )

        with self.maybe_autocast():
            inputs_embeds = torch.cat([inputs_llm, inputs_embeds], dim=1)
            encoder_atts = torch.cat(
                [atts_llm, input_tokens.attention_mask], dim=1
            )

            if self.use_iou_loss:
                gt_boxes = samples["gt_boxes"]
                aug_shapes = samples["image_sizes"]
            else:
                gt_boxes, aug_shapes = None, None

            if self.use_iou_loss:
                gt_boxes = samples["gt_boxes"]
                aug_shapes = samples["image_sizes"]
            else:
                gt_boxes, aug_shapes = None, None

            outputs = self.llm(
                inputs_embeds=inputs_embeds,
                attention_mask=encoder_atts,
                decoder_attention_mask=output_tokens.attention_mask,
                return_dict=True,
                labels=targets,
                tokenizer=self.tokenizer,
                target_token_weights=None,
                gt_boxes=gt_boxes,
                aug_shapes=aug_shapes,
            )

            loss = outputs.loss
            loss_lm_record = outputs.loss.clone()

        return outputs, loss, loss_lm_record

    def forward_others(self, samples):
        image = samples["video"]
        B, T = image.shape[:2]
        T_ori = T

        with self.maybe_autocast():
            if "videoswin" not in self.vit_name:
                # video inputs: B, T, 3, H, W -> BT, 3, H, W
                image = image.view(-1, *image.shape[2:])
                image_feats = self.visual_encoder(image)
                image_embeds = self.ln_vision(image_feats)

                image_atts = torch.ones(
                    image_embeds.size()[:-1], dtype=torch.long
                ).to(image.device)

            else:
                # video inputs: B, T, 3, H, W -> B, 3, T, H, W
                image = image.permute(0, 2, 1, 3, 4)
                image_feats = self.visual_encoder(image)
                # B, L, C (L = T//w_t * H//w_h * W//w_w)
                image_embeds = self.ln_vision(image_feats)
                T = T // self.visual_encoder.patch_embed.patch_size[0]
                # B, L, C -> BdownT, L', C
                image_embeds = image_embeds.view(
                    B * T, -1, image_embeds.shape[-1]
                )

                image_atts = torch.ones(
                    image_embeds.size()[:-1], dtype=torch.long
                ).to(image.device)

        query_tokens = self.query_tokens.expand(image_embeds.shape[0], -1, -1)

        with self.maybe_autocast():
            input_tokens = samples["prompt"]
            input_tokens = Dict(
                {k: v.to(image.device) for k, v in input_tokens.items()}
            )
            if hasattr(self.llm, "encoder"):
                inputs_embeds = self.llm.encoder.embed_tokens(
                    input_tokens.input_ids
                )

            else:
                inputs_embeds = self.llm.model.encoder.embed_tokens(
                    input_tokens.input_ids
                )

        if self.box_in_query:
            normalized_prompt_box = samples["reference_points"]
            prompt_to_reference_embeddings = self.get_proposal_pos_embed(
                normalized_prompt_box, self.query_tokens.shape[-1]
            )
            prompt_to_reference_embeddings = (
                prompt_to_reference_embeddings.expand(
                    inputs_embeds.shape[0], -1, -1
                )
            )
            query_tokens = query_tokens + prompt_to_reference_embeddings

        if self.language_in_query:
            inputs_embeds_mean = inputs_embeds.mean(dim=1, keepdim=True)
            language_queries = self.text2query_norm(
                self.text2query(inputs_embeds_mean)
            )
            # query_tokens: BT, 32, C
            # language_queries: B, 1, C -> B, 1, 1, C -> B, T, 1, C -> BT, 1, C
            language_queries = language_queries.unsqueeze(1).repeat(1, T, 1, 1)
            language_queries = language_queries.view(
                -1, 1, language_queries.shape[-1]
            )
            query_tokens = query_tokens + language_queries

        query_output = self.Qformer.bert(
            query_embeds=query_tokens,
            encoder_hidden_states=image_embeds,
            encoder_attention_mask=image_atts,
            return_dict=True,
        )

        inputs_llm = self.llm_proj(query_output.last_hidden_state)
        inputs_llm = inputs_llm.view(B, T, -1, inputs_llm.shape[-1])

        if self.use_audio:
            audio = samples["audio"]
            audio = audio.view(-1, *audio.shape[2:])
            # BT, 1, 96, 64 -> BT, 128
            audio_feats = self.audio_encoder(audio)
            # BT, 128 -> B, T, 128
            audio_feats = audio_feats.view(B, T_ori, -1)
            audio_feats = audio_feats.view(B, T, -1, audio_feats.shape[-1])
            audio_feats = audio_feats.mean(dim=2)

            audio_feats = self.proj_audio(audio_feats)
            audio_embeds = self.proj_audio_norm(audio_feats)

            atts_audio = torch.ones(
                audio_embeds.size()[:-1], dtype=torch.long
            ).to(audio.device)

        else:
            audio_embeds, atts_audio = None, None

        if audio_embeds is not None and self.fuse_audio == "add":
            # inputs_llm: B, T, L, C
            # audio_embeds: B, T, C
            audio_embeds = audio_embeds.unsqueeze(2).repeat(
                1, 1, inputs_llm.shape[2], 1
            )
            inputs_llm = inputs_llm + audio_embeds

        with self.maybe_autocast():
            if self.cross_frame_fusion == "late":
                inputs_llm = self.temporal_encoder(inputs_llm)

            else:
                # B, T, 32, C -> B, T32, C
                assert hasattr(self, "clip_pos_embed")
                # 1, T, C -> 1, T, 32, C
                clip_pos_embed = self.clip_pos_embed.unsqueeze(2).repeat(
                    1, 1, self.num_query_token, 1
                )
                inputs_llm = inputs_llm + clip_pos_embed
                inputs_llm = inputs_llm.view(B, -1, inputs_llm.shape[-1])

        atts_llm = torch.ones(inputs_llm.size()[:-1], dtype=torch.long).to(
            image.device
        )

        if audio_embeds is not None and self.fuse_audio == "concat":
            inputs_llm = torch.cat([inputs_llm, audio_embeds], dim=1)
            atts_llm = torch.cat([atts_llm, atts_audio], dim=1)

        with self.maybe_autocast():
            output_tokens = samples["text"]
            output_tokens = Dict(
                {k: v.to(image.device) for k, v in output_tokens.items()}
            )

            encoder_atts = torch.cat(
                [atts_llm, input_tokens.attention_mask], dim=1
            )

            targets = output_tokens.input_ids.masked_fill(
                output_tokens.input_ids
                == self.tokenizer.tokenizer.pad_token_id,
                -100,
            )

            targets2 = None
            if hasattr(samples, "text2") and isinstance(samples["text2"], Dict):
                output_tokens2 = samples["text2"]
                output_tokens2 = Dict(
                    {k: v.to(image.device) for k, v in output_tokens2.items()}
                )
                targets2 = output_tokens2.input_ids.masked_fill(
                    output_tokens2.input_ids
                    == self.tokenizer.tokenizer.pad_token_id,
                    -100,
                )

            if self.vl_fuse:
                fused_vision_embeds, fused_text_embeds = self.t2v_encoder(
                    inputs_llm,
                    inputs_embeds,
                    input_tokens.attention_mask,
                    True,
                )

                inputs_embeds = torch.cat(
                    [fused_vision_embeds, fused_text_embeds], dim=1
                )

            else:
                inputs_embeds = torch.cat([inputs_llm, inputs_embeds], dim=1)

            outputs = self.llm(
                inputs_embeds=inputs_embeds,
                attention_mask=encoder_atts,
                decoder_attention_mask=output_tokens.attention_mask,
                return_dict=True,
                labels=targets,
                labels2=targets2,
                tokenizer=self.tokenizer,
            )

            loss = outputs.loss
            loss_lm_record = outputs.loss.clone()

        return outputs, loss, loss_lm_record

    def forward(self, samples):
        task = samples["task"][0]
        if task == "sot":
            _, loss, loss_lm_record = self.forward_sot(samples)
            return {
                "loss": loss,
                "lm_loss": loss_lm_record.item(),
            }

        else:
            _, loss, loss_lm_record = self.forward_others(samples)

            return {
                "loss": loss,
                "lm_loss": loss_lm_record.item(),
            }

    @torch.no_grad()
    def generate(
        self,
        samples,
        use_nucleus_sampling=False,
        num_beams=5,
        max_length=30,
        min_length=1,
        top_p=0.9,
        repetition_penalty=1.0,
        length_penalty=1.0,
        num_captions=1,
        temperature=1,
        output_scores=False,
    ):
        """
        Args:
            samples (dict): A dictionary containing the following keys:
                - videos (tensor): A tensor of shape (batch_size, num_frms, 3, H, W).
                - text: A list of strings of length batch_size.
                - timestamps: A list of tuples of length batch_size.
                - duration: A list of floats of length batch_size.
                - prompt: A list of strings of length batch_size.

            use_nucleus_sampling (bool): Whether to use nucleus sampling. If False, use top-k sampling.
            num_beams (int): Number of beams for beam search. 1 means no beam search.
            max_length (int): The maximum length of the sequence to be generated.
            min_length (int): The minimum length of the sequence to be generated.
            top_p (float): The cumulative probability for nucleus sampling.
            repetition_penalty (float): The parameter for repetition penalty. 1.0 means no penalty.
            num_captions (int): Number of captions to be generated for each image.
        Returns:
            captions (list): A list of strings of length batch_size * num_captions.
        """

        image = samples["video"]
        B, T = image.shape[:2]
        T_ori = T

        with self.maybe_autocast():
            if "videoswin" not in self.vit_name:
                # video inputs: B, T, 3, H, W -> BT, 3, H, W
                image = image.view(-1, *image.shape[2:])
                image_feats = self.visual_encoder(image)
                image_embeds = self.ln_vision(image_feats)

            else:
                # video inputs: B, T, 3, H, W -> B, 3, T, H, W
                image = image.permute(0, 2, 1, 3, 4)
                image_feats = self.visual_encoder(image)
                # B, L, C (L = T//w_t * H//w_h * W//w_w)
                image_embeds = self.ln_vision(image_feats)
                T = T // self.visual_encoder.patch_embed.patch_size[0]
                # B, L, C -> BdownT, L', C
                image_embeds = image_embeds.view(
                    B * T, -1, image_embeds.shape[-1]
                )

        image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(
            image.device
        )

        query_tokens = self.query_tokens.expand(image_embeds.shape[0], -1, -1)

        with self.maybe_autocast():
            input_tokens = samples["prompt"]
            input_tokens = Dict(
                {k: v.to(image.device) for k, v in input_tokens.items()}
            )
            if hasattr(self.llm, "encoder"):
                inputs_embeds = self.llm.encoder.embed_tokens(
                    input_tokens.input_ids
                )

            else:
                inputs_embeds = self.llm.model.encoder.embed_tokens(
                    input_tokens.input_ids
                )

            if self.language_in_query:
                inputs_embeds_mean = inputs_embeds.mean(dim=1, keepdim=True)
                language_queries = self.text2query_norm(
                    self.text2query(inputs_embeds_mean)
                )

                # query_tokens: BT, 32, C
                # language_queries: B, 1, C -> B, 1, 1, C -> B, T, 1, C -> BT, 1, C
                language_queries = language_queries.unsqueeze(1).repeat(
                    1, T, 1, 1
                )
                language_queries = language_queries.view(
                    -1, 1, language_queries.shape[-1]
                )
                query_tokens = query_tokens + language_queries

            query_output = self.Qformer.bert(
                query_embeds=query_tokens,
                encoder_hidden_states=image_embeds,
                encoder_attention_mask=image_atts,
                return_dict=True,
            )

        inputs_llm = self.llm_proj(query_output.last_hidden_state)
        inputs_llm = inputs_llm.view(B, T, -1, inputs_llm.shape[-1])

        if self.use_audio:
            audio = samples["audio"]
            audio = audio.view(-1, *audio.shape[2:])
            # BT, 1, 96, 64 -> BT, 128
            audio_feats = self.audio_encoder(audio)
            # BT, 128 -> B, T, 128
            audio_feats = audio_feats.view(B, T_ori, -1)
            audio_feats = audio_feats.view(B, T, -1, audio_feats.shape[-1])
            audio_feats = audio_feats.mean(dim=2)

            audio_feats = self.proj_audio(audio_feats)
            audio_embeds = self.proj_audio_norm(audio_feats)

            atts_audio = torch.ones(
                audio_embeds.size()[:-1], dtype=torch.long
            ).to(audio.device)

        else:
            audio_embeds, atts_audio = None, None

        if audio_embeds is not None and self.fuse_audio == "add":
            # inputs_llm: B, T, L, C
            # audio_embeds: B, T, C
            audio_embeds = audio_embeds.unsqueeze(2).repeat(
                1, 1, inputs_llm.shape[2], 1
            )
            inputs_llm = inputs_llm + audio_embeds

        if self.cross_frame_fusion == "late":
            with self.maybe_autocast():
                inputs_llm = self.temporal_encoder(inputs_llm)
        else:
            assert hasattr(self, "clip_pos_embed")
            clip_pos_embed = self.clip_pos_embed.unsqueeze(2).repeat(
                1, 1, self.num_query_token, 1
            )
            inputs_llm = inputs_llm + clip_pos_embed
            inputs_llm = inputs_llm.view(B, -1, inputs_llm.shape[-1])

        atts_llm = torch.ones(inputs_llm.size()[:-1], dtype=torch.long).to(
            image.device
        )

        if audio_embeds is not None and self.fuse_audio == "concat":
            inputs_llm = torch.cat([inputs_llm, audio_embeds], dim=1)
            atts_llm = torch.cat([atts_llm, atts_audio], dim=1)

        encoder_atts = torch.cat([atts_llm, input_tokens.attention_mask], dim=1)

        with self.maybe_autocast():
            video_input_length = inputs_llm.shape[1]
            inputs_embeds = torch.cat([inputs_llm, inputs_embeds], dim=1)

            if self.vl_fuse:
                vision_embeds = inputs_embeds[:, :video_input_length]
                text_embeds = inputs_embeds[:, video_input_length:]
                text_masks = input_tokens.attention_mask

                fused_vision_embeds, fused_text_embeds = self.t2v_encoder(
                    vision_embeds,
                    text_embeds,
                    text_masks,
                    True,
                )

                inputs_embeds = torch.cat(
                    [fused_vision_embeds, fused_text_embeds], dim=1
                )

            else:
                inputs_embeds = inputs_embeds

            outputs = self.llm.generate(
                inputs_embeds=inputs_embeds,
                attention_mask=encoder_atts,
                do_sample=use_nucleus_sampling,
                top_p=top_p,
                temperature=temperature,
                num_beams=num_beams,
                max_new_tokens=max_length,
                min_length=min_length,
                repetition_penalty=repetition_penalty,
                length_penalty=length_penalty,
                num_return_sequences=num_captions,
                return_dict_in_generate=output_scores,
                output_scores=output_scores,
            )

        return outputs

    @torch.no_grad()
    def online_caption(
        self,
        samples,
        use_nucleus_sampling=False,
        num_beams=5,
        max_length=30,
        min_length=1,
        top_p=0.9,
        repetition_penalty=1.0,
        length_penalty=1.0,
        num_captions=1,
        temperature=1,
        output_scores=False,
        tokenizer=None,
    ):
        image = samples["video"]
        # B, N, T, 3, H, W
        B, N, T = image.shape[:3]
        assert B == 1, "online captioning only supports batch size 1"

        # we shoule implement a more efficient way to achieve online inference:
        # process the video clip by clip, and then concatenate the results
        image = image.view(N, T, *image.shape[3:])
        if self.use_audio:
            audio = samples["audio"]
            # B, N, T, 1, 96, 64
            audio = audio.view(N, T, *audio.shape[3:])

        else:
            audio = None

        if "videoswin" in self.vit_name:
            T = T // self.visual_encoder.patch_embed.patch_size[0]

        with self.maybe_autocast():
            input_tokens = samples["prompt"]
            input_tokens = Dict({k: v.cuda() for k, v in input_tokens.items()})
            if hasattr(self.llm, "encoder"):
                inputs_embeds = self.llm.encoder.embed_tokens(
                    input_tokens.input_ids
                )

            else:
                inputs_embeds = self.llm.model.encoder.embed_tokens(
                    input_tokens.input_ids
                )

            # print(samples["reference_points"])
            normalized_reference_points = samples["reference_points"].clone()
            normalized_reference_points /= samples["duration"]

            query_tokens = self.query_tokens
            if self.language_in_query:
                inputs_embeds_mean = inputs_embeds.mean(dim=1, keepdim=True)
                language_queries = self.text2query_norm(
                    self.text2query(inputs_embeds_mean)
                )

                # query_tokens: 1, 32, C
                # language_queries: 1, 1, C
                query_tokens = query_tokens + language_queries

        output_captions = []
        output_ious = []
        output_conf = []
        for c in range(N):
            clip_image = image[c : c + 1]  # 1, T, 3, H, W
            clip_audio = (
                audio[c : c + 1] if audio is not None else None
            )  # 1, T, 1, 96, 64

            with self.maybe_autocast():
                if "videoswin" not in self.vit_name:
                    clip_image = clip_image.view(
                        -1, *clip_image.shape[2:]
                    )  # T, 3, H, W
                    image_feats = self.visual_encoder(clip_image)
                    image_embeds = self.ln_vision(image_feats)  # T, L, C

                else:
                    # 1, T, 3, H, W -> 1, 3, T, H, W
                    clip_image = clip_image.permute(0, 2, 1, 3, 4)
                    image_feats = self.visual_encoder(clip_image)
                    image_embeds = self.ln_vision(
                        image_feats
                    )  # 1, L, C (L = T//w_t * H//w_h * W//w_w)
                    # 1, L, C -> 1T', L', C
                    image_embeds = image_embeds.view(
                        T, -1, image_embeds.shape[-1]
                    )

            image_atts = torch.ones(
                image_embeds.size()[:-1], dtype=torch.long
            ).to(clip_image.device)
            clip_query_tokens = query_tokens.expand(
                image_embeds.shape[0], -1, -1
            )

            if self.box_in_query:
                clip_normalized_prompt_box = normalized_reference_points[:, c]
                prompt_to_reference_embeddings = self.get_proposal_pos_embed(
                    clip_normalized_prompt_box, self.query_tokens.shape[-1]
                )
                prompt_to_reference_embeddings = (
                    prompt_to_reference_embeddings.expand(
                        inputs_embeds.shape[0], -1, -1
                    )
                )
                clip_query_tokens = (
                    clip_query_tokens + prompt_to_reference_embeddings
                )

            with self.maybe_autocast():
                query_output = self.Qformer.bert(
                    query_embeds=clip_query_tokens,
                    encoder_hidden_states=image_embeds,
                    encoder_attention_mask=image_atts,
                    return_dict=True,
                )  # T, L, C

            inputs_llm = self.llm_proj(query_output.last_hidden_state)
            inputs_llm = inputs_llm.unsqueeze(0)  # 1, T, L, C

            if clip_audio is not None:
                # 1, T, 1, 96, 64 -> T, 1, 96, 64
                clip_audio = clip_audio.view(-1, *audio.shape[2:])
                # T, 1, 96, 64 -> T, 128
                clip_audio_feats = self.audio_encoder(clip_audio)
                clip_audio_feats = clip_audio_feats.view(
                    T, -1, clip_audio_feats.shape[-1]
                )
                clip_audio_feats = clip_audio_feats.mean(dim=1)

                clip_audio_feats = self.proj_audio(clip_audio_feats)
                clip_audio_embeds = self.proj_audio_norm(
                    clip_audio_feats
                ).unsqueeze(
                    0
                )  # 1, T, C

                atts_audio = torch.ones(
                    clip_audio_embeds.size()[:-1], dtype=torch.long
                ).to(audio.device)

            else:
                audio_embeds, atts_audio = None, None

            if audio_embeds is not None and self.fuse_audio == "add":
                # inputs_llm: 1, T, L, C
                # audio_embeds: 1, T, C
                audio_embeds = audio_embeds.unsqueeze(2).repeat(
                    1, 1, inputs_llm.shape[2], 1
                )
                inputs_llm = inputs_llm + audio_embeds

            if self.cross_frame_fusion == "late":
                with self.maybe_autocast():
                    inputs_llm = self.temporal_encoder(inputs_llm)
            else:
                assert hasattr(self, "clip_pos_embed")
                clip_pos_embed = self.clip_pos_embed.unsqueeze(2).repeat(
                    1, 1, self.num_query_token, 1
                )
                inputs_llm = inputs_llm + clip_pos_embed
                inputs_llm = inputs_llm.view(B, -1, inputs_llm.shape[-1])

            atts_llm = torch.ones(inputs_llm.size()[:-1], dtype=torch.long).to(
                clip_image.device
            )

            if audio_embeds is not None and self.fuse_audio == "concat":
                inputs_llm = torch.cat([inputs_llm, audio_embeds], dim=1)
                atts_llm = torch.cat([atts_llm, atts_audio], dim=1)

            with self.maybe_autocast():
                video_input_length = inputs_llm.shape[1]

                clip_inputs_embeds = torch.cat(
                    [inputs_llm, inputs_embeds], dim=1
                )
                clip_encoder_atts = torch.cat(
                    [atts_llm, input_tokens.attention_mask], dim=1
                )

                if self.vl_fuse:
                    vision_embeds = clip_inputs_embeds[:, :video_input_length]
                    text_embeds = clip_inputs_embeds[:, video_input_length:]
                    text_masks = input_tokens.attention_mask

                    fused_vision_embeds, fused_text_embeds = self.t2v_encoder(
                        vision_embeds,
                        text_embeds,
                        text_masks,
                        True,
                    )

                    clip_inputs_embeds = torch.cat(
                        [fused_vision_embeds, fused_text_embeds], dim=1
                    )

                else:
                    clip_inputs_embeds = clip_inputs_embeds

                outputs = self.llm.generate(
                    inputs_embeds=clip_inputs_embeds,
                    attention_mask=clip_encoder_atts,
                    do_sample=use_nucleus_sampling,
                    top_p=top_p,
                    temperature=temperature,
                    num_beams=num_beams,
                    max_new_tokens=max_length,
                    min_length=min_length,
                    repetition_penalty=repetition_penalty,
                    length_penalty=length_penalty,
                    num_return_sequences=num_captions,
                    return_dict_in_generate=output_scores,
                    output_scores=output_scores,
                )

            if isinstance(outputs, dict):
                token_sequences, token_scores = (
                    outputs["sequences"],
                    outputs["scores"],
                )

                batched_token_scores = []
                # travase each position
                for token_score in token_scores:
                    token_score = torch.nn.functional.softmax(
                        token_score, dim=-1
                    )
                    this_pos = token_score.max(dim=1)[0]
                    batched_token_scores.append(this_pos)

                batched_token_scores = (
                    torch.stack(batched_token_scores, dim=0)
                    .transpose(0, 1)
                    .detach()
                    .cpu()
                )[0].tolist()

            else:
                token_sequences = outputs
                batched_token_scores = None

            (
                clip_caption,
                clip_iou,
                clip_score,
            ) = tokenizer.restore_caption_and_iou(
                token_sequences[0].cpu().tolist()[1:], batched_token_scores
            )
            output_captions.append(clip_caption)
            output_ious.append(clip_iou)
            output_conf.append(clip_score)

            del clip_image
            del image_feats
            del image_embeds
            del image_atts
            del inputs_llm
            del atts_llm
            del clip_query_tokens
            del clip_inputs_embeds
            del clip_encoder_atts
            del outputs

        return output_captions, output_ious, output_conf

    @torch.no_grad()
    def online_generate(
        self,
        samples,
        use_nucleus_sampling=False,
        num_beams=5,
        max_length=30,
        min_length=1,
        top_p=0.9,
        repetition_penalty=1.0,
        length_penalty=1.0,
        num_captions=1,
        temperature=1,
        output_scores=False,
        height=None,
        width=None,
        update_interval=1,
        update_threshold=0.5,
        confidence_threshold=-1e4,
        compensate="last",
        tokenizer=None,
        kf=None,
    ):
        """
        Args:
            samples (dict): A dictionary containing the following keys:
                - videos (tensor): A tensor of shape (batch_size, num_frms, 3, H, W).
                - text: A list of strings of length batch_size.
                - timestamps: A list of tuples of length batch_size.
                - duration: A list of floats of length batch_size.
                - prompt: A list of strings of length batch_size.

            use_nucleus_sampling (bool): Whether to use nucleus sampling. If False, use top-k sampling.
            num_beams (int): Number of beams for beam search. 1 means no beam search.
            max_length (int): The maximum length of the sequence to be generated.
            min_length (int): The minimum length of the sequence to be generated.
            top_p (float): The cumulative probability for nucleus sampling.
            repetition_penalty (float): The parameter for repetition penalty. 1.0 means no penalty.
            num_captions (int): Number of captions to be generated for each image.
        Returns:
            captions (list): A list of strings of length batch_size * num_captions.
        """

        image = samples["video"]
        task = samples["task"][0]

        B, T = image.shape[:2]
        assert B == 1, "online inference for SOT only support batch size 1"
        # video inputs: 1, T, 3, H, W -> T, 3, H, W
        image = image.view(-1, *image.shape[2:])
        mask = samples["image_mask"]
        mask = mask.view(-1, *mask.shape[2:])

        if "videoswin" not in self.vit_name:
            image_shape = (
                image.shape[2] // self.visual_encoder.patch_embed.patch_size[0],
                image.shape[3] // self.visual_encoder.patch_embed.patch_size[1],
            )
        else:
            image_shape = (
                math.ceil(image.shape[2] / 32),
                math.ceil(image.shape[3] / 32),
            )

        input_tokens = samples["prompt"]
        input_tokens = Dict(
            {k: v.to(image.device) for k, v in input_tokens.items()}
        )

        prompt_box = samples["prompt_box"]
        ref_image = image[0:1]  # 1, 3, H, W
        ref_image = self.crop_image_from_boxes(
            ref_image,
            prompt_box,
            self.template_size,
            self.template_boundary,
        )

        if "videoswin" not in self.vit_name:
            with self.maybe_autocast():
                ref_feats = self.visual_encoder(ref_image)
                ref_embeds = self.ln_vision(ref_feats)  # 1, 1+HW, C

                ref_image_atts = (
                    torch.ones(ref_embeds.shape[0], ref_embeds.shape[1])
                    .to(torch.long)
                    .to(ref_embeds.device)
                )

            if self.cross_frame_fusion == "late":
                query_tokens = self.query_tokens.expand(
                    ref_embeds.shape[0], -1, -1
                )
                ref_query_output = self.Qformer.bert(
                    query_embeds=query_tokens,
                    encoder_hidden_states=ref_embeds,
                    encoder_attention_mask=ref_image_atts,
                    return_dict=True,
                )

            elif self.cross_frame_fusion == "early_full":
                query_tokens = self.query_tokens.expand(
                    ref_embeds.shape[0], -1, -1
                )

        else:
            ref_image = F.interpolate(
                ref_image, size=(image.shape[2], image.shape[3]), mode="bicubic"
            )
            query_tokens = self.query_tokens.expand(ref_image.shape[0], -1, -1)

        num_clips = math.ceil(T / (self.num_frms - 1))
        pred_sequences = []
        pred_bboxes = []
        pred_scores = []

        kf_bboxes = []
        iou_scores = []
        for c in range(num_clips):
            start_idx = c * (self.num_frms - 1)
            end_idx = min((c + 1) * (self.num_frms - 1), T)
            clip_image = image[start_idx:end_idx]  # 1, 3, H, W
            clip_length = end_idx - start_idx  # 1
            clip_mask = mask[start_idx:end_idx]  # 1, H, W

            with self.maybe_autocast():
                if hasattr(self.llm, "encoder"):
                    inputs_embeds = self.llm.encoder.embed_tokens(
                        input_tokens.input_ids
                    )

                else:
                    inputs_embeds = self.llm.model.encoder.embed_tokens(
                        input_tokens.input_ids
                    )

                if "videoswin" not in self.vit_name:
                    clip_feats = self.visual_encoder(clip_image)
                    clip_embeds = self.ln_vision(clip_feats)  # 1, 1+HW, C

                    clip_atts = (
                        F.interpolate(
                            (~clip_mask).unsqueeze(1).float(), size=image_shape
                        )
                        .to(torch.long)[:, 0]
                        .view(clip_embeds.shape[0], -1)
                    )
                    cls_atts = (
                        torch.tensor([[True]])
                        .to(clip_atts.device)
                        .repeat(clip_atts.shape[0], 1)
                    )
                    clip_atts = torch.cat([cls_atts, clip_atts], dim=1)

                else:
                    # [1, 3, H, W | 1, 3, H, W] -> [1, 2, 3, H, W]
                    ref_clip_image = torch.stack(
                        (ref_image, clip_image), dim=1
                    ).transpose(1, 2)
                    ref_clip_feats = self.visual_encoder(ref_clip_image)
                    ref_clip_embeds = self.ln_vision(
                        ref_clip_feats
                    )  # 1, THW, C

                    _, THW, _ = ref_clip_embeds.shape
                    HW = THW // 2
                    ref_clip_embeds = ref_clip_embeds.view(B, 2, HW, -1)

                    ref_embeds = ref_clip_embeds[:, 0]
                    clip_embeds = ref_clip_embeds[:, 1]

                    ref_image_atts = (
                        torch.ones(ref_embeds.shape[0], ref_embeds.shape[1])
                        .to(torch.long)
                        .to(ref_embeds.device)
                    )

                    clip_atts = (
                        F.interpolate(
                            (~clip_mask).unsqueeze(1).float(), size=image_shape
                        )
                        .to(torch.long)[:, 0]
                        .view(clip_embeds.shape[0], -1)
                    )

            if self.cross_frame_fusion == "early_full":
                if "videoswin" not in self.vit_name:
                    ref_cls_feats = ref_embeds[:, 0:1]
                    ref_image_embeds = ref_embeds[:, 1:]
                    ref_image_embeds_h = ref_image_embeds_w = int(
                        math.sqrt(ref_image_embeds.shape[1])
                    )

                    tgt_cls_feats = clip_embeds[:, 0:1]
                    tgt_image_embeds = clip_embeds[:, 1:]

                else:
                    ref_image_embeds = ref_embeds
                    ref_image_embeds_h, ref_image_embeds_w = image_shape
                    tgt_image_embeds = clip_embeds

                    ref_cls_feats, tgt_cls_feats = None, None

                feat1 = (
                    ref_image_embeds.view(
                        B, ref_image_embeds_h, ref_image_embeds_w, -1
                    )
                    .permute(0, 3, 1, 2)
                    .contiguous()
                )
                feat2 = (
                    tgt_image_embeds.view(B, image_shape[0], image_shape[1], -1)
                    .permute(0, 3, 1, 2)
                    .contiguous()
                )

                with self.maybe_autocast():
                    srcs = [self.bottleneck(feat1), self.bottleneck(feat2)]

                    pos1 = self.fusion_pos_embed(
                        B, ref_image_embeds_h, ref_image_embeds_w
                    )
                    """Interpolate positional encoding according to input size"""
                    pos1 = F.interpolate(
                        pos1,
                        size=(ref_image_embeds_h, ref_image_embeds_w),
                        mode="bicubic",
                    )

                    pos2 = self.fusion_pos_embed(
                        B, image_shape[0], image_shape[1]
                    )
                    pos2 = F.interpolate(
                        pos2,
                        size=(image_shape[0], image_shape[1]),
                        mode="bicubic",
                    )

                    pos = [pos1, pos2]
                    srcs_new = [
                        x.flatten(-2).permute((2, 0, 1)).contiguous()
                        for x in srcs
                    ]
                    pos_new = [
                        x.flatten(-2).permute((2, 0, 1)).contiguous()
                        for x in pos
                    ]

                    out_seq = self.fusion_transformer(
                        torch.cat(srcs_new, dim=0),
                        mask=None,
                        pos_embed=torch.cat(pos_new, dim=0),
                    )
                    # Split the sequence and reshape back to feature maps
                    half_len = ref_image_embeds_h * ref_image_embeds_w
                    seq1, seq2 = (
                        out_seq[:half_len],
                        out_seq[half_len:],
                    )  # h1w1, B, C, h2w2, B, C

                if ref_cls_feats is not None:
                    ref_clip_embeds = torch.cat(
                        [ref_cls_feats, seq1.permute(1, 0, 2)], dim=1
                    )
                else:
                    ref_clip_embeds = seq1.permute(1, 0, 2)

                if tgt_cls_feats is not None:
                    clip_embeds = torch.cat(
                        [tgt_cls_feats, seq2.permute(1, 0, 2)], dim=1
                    )
                else:
                    clip_embeds = seq2.permute(1, 0, 2)

            box_enhanced_query_tokens = query_tokens.clone()
            if self.box_in_query:
                if self.box2query == "add":
                    normalized_prompt_box = prompt_box.clone()
                    normalized_prompt_box[:, 0::2] /= samples["image_sizes"][0][
                        0
                    ][1]
                    normalized_prompt_box[:, 1::2] /= samples["image_sizes"][0][
                        0
                    ][0]

                    prompt_to_reference_embeddings = (
                        self.get_proposal_pos_embed(
                            normalized_prompt_box, self.query_tokens.shape[-1]
                        )
                    )
                    prompt_to_reference_embeddings = (
                        prompt_to_reference_embeddings.expand(
                            clip_embeds.shape[0], -1, -1
                        )
                    )
                    box_enhanced_query_tokens = (
                        query_tokens + prompt_to_reference_embeddings
                    )

                elif self.box2query == "concat":
                    prompt_to_reference_embeddings = (
                        self.get_proposal_pos_embed2(
                            prompt_box,
                            self.query_tokens.shape[-1],
                            height=samples["image_sizes"][0][0][0],
                            width=samples["image_sizes"][0][0][1],
                        )
                    )
                    prompt_to_reference_embeddings = (
                        prompt_to_reference_embeddings.expand(
                            clip_embeds.shape[0], -1, -1
                        )
                    )
                    box_enhanced_query_tokens = torch.cat(
                        [query_tokens, prompt_to_reference_embeddings], dim=1
                    )
                else:
                    raise NotImplementedError

            if self.language_in_query:
                inputs_embeds_mean = inputs_embeds.mean(dim=1, keepdim=True)
                language_queries = self.text2query_norm(
                    self.text2query(inputs_embeds_mean)
                )
                box_enhanced_query_tokens = (
                    box_enhanced_query_tokens + language_queries
                )

            query_output = self.Qformer.bert(
                query_embeds=box_enhanced_query_tokens,
                encoder_hidden_states=clip_embeds,
                encoder_attention_mask=clip_atts,
                return_dict=True,
            )

            clip_hidden = query_output.last_hidden_state

            if self.use_template_feat:
                ref_query_output = self.Qformer.bert(
                    query_embeds=box_enhanced_query_tokens,
                    encoder_hidden_states=ref_clip_embeds,
                    encoder_attention_mask=ref_image_atts,
                    return_dict=True,
                )

                clip_hidden = torch.cat(
                    [ref_query_output.last_hidden_state, clip_hidden], dim=1
                )

            if self.cross_frame_fusion == "early_full":
                inputs_llm = self.llm_proj(clip_hidden)
                inputs_llm = inputs_llm.view(B, -1, inputs_llm.shape[-1])

            else:
                visual_hidden_states = torch.cat(
                    [
                        ref_query_output.last_hidden_state[
                            :, : self.num_query_token
                        ],  # 1, num_q1, C
                        query_output.last_hidden_state,  # 1, num_q2, C
                    ],
                    dim=1,
                )

                inputs_llm = self.llm_proj(
                    visual_hidden_states
                )  # 1, num_q1 + num_q2, C

                if self.box2query == "add":
                    inputs_llm = inputs_llm.view(
                        B, clip_length + 1, -1, inputs_llm.shape[-1]
                    )
                else:
                    inputs_llm = inputs_llm.view(B, -1, inputs_llm.shape[-1])

                inputs_llm = self.temporal_encoder(
                    inputs_llm,
                )

            # concat the image embeddings to Qformer output
            if self.cross_frame_fusion == "none":
                projected_image_embeds = self.prompt_proj(clip_embeds)
                inputs_llm = torch.cat(
                    [inputs_llm, projected_image_embeds], dim=1
                )

            atts_llm = torch.ones(inputs_llm.size()[:-1], dtype=torch.long).to(
                image.device
            )

            encoder_atts = torch.cat(
                [atts_llm, input_tokens.attention_mask], dim=1
            )

            inputs_embeds = torch.cat([inputs_llm, inputs_embeds], dim=1)

            outputs = self.llm.generate(
                inputs_embeds=inputs_embeds,
                attention_mask=encoder_atts,
                do_sample=use_nucleus_sampling,
                top_p=top_p,
                temperature=temperature,
                num_beams=num_beams,
                max_new_tokens=max_length,
                min_length=min_length,
                repetition_penalty=repetition_penalty,
                length_penalty=length_penalty,
                num_return_sequences=num_captions,
                return_dict_in_generate=output_scores,
                output_scores=output_scores,
            )

            token_sequences, token_scores = (
                outputs["sequences"],
                outputs["scores"],
            )
            batched_token_scores = []
            # travase each position
            for token_score in token_scores:
                token_score = torch.nn.functional.softmax(token_score, dim=-1)
                this_pos = token_score.max(dim=1)[0]
                batched_token_scores.append(this_pos)

            batched_token_scores = (
                torch.stack(batched_token_scores, dim=0)
                .transpose(0, 1)
                .detach()
                .cpu()
            )  # .tolist()

            boxes, _, scores = self.tokenizer.restore_box(
                token_sequences[0].cpu().tolist(),
                height,
                width,
                batched_token_scores[0],
            )

            if c == 0:
                boxes = prompt_box.cpu().tolist()
                scores = [1.0] * len(boxes)

            pred_length = clip_length
            if len(boxes) == 0:
                boxes = [[0, 0, width, height]] * pred_length
                scores = [0] * pred_length
            elif len(boxes) < pred_length:
                boxes = boxes + [boxes[-1]] * (pred_length - len(boxes))
                scores = scores + [0] * (pred_length - len(scores))
            else:
                boxes = boxes[:pred_length]
                scores = scores[:pred_length]

            boxes = torch.FloatTensor(boxes)
            scores = torch.FloatTensor(scores)

            # clamp the boxes
            boxes[:, 0::2].clamp_(min=0, max=width)
            boxes[:, 1::2].clamp_(min=0, max=height)

            if c == 0:
                boxes_cxcyah = bbox_xyxy_to_cxcyah(boxes)
                boxes_cxcyah = boxes_cxcyah.cpu().numpy()[0]
                mean, cov = kf.initiate(boxes_cxcyah)
                track_bboxes = boxes
                last_frame = c
                last_boxes = boxes
                iou = torch.ones(1).to(boxes.device)

            else:
                if last_frame != c - 1:
                    mean[7] = 0

                # predict by kalman filter
                mean, cov = kf.predict(mean, cov)
                track_bboxes = np.zeros((0, 4))
                track_bboxes = np.concatenate(
                    (track_bboxes, mean[:4][None]), axis=0
                )
                track_bboxes = (
                    torch.from_numpy(track_bboxes)
                    .to(boxes.dtype)
                    .to(boxes.device)
                )
                track_bboxes = bbox_cxcyah_to_xyxy(track_bboxes)

                # clamp the boxes
                track_bboxes[:, 0::2].clamp_(min=0, max=width)
                track_bboxes[:, 1::2].clamp_(min=0, max=height)

                try:
                    iou = generalized_box_iou(last_boxes, boxes).squeeze(0)
                except:
                    iou = torch.zeros(1).to(boxes.device)

                area = (boxes[:, 2] - boxes[:, 0]) * (
                    boxes[:, 3] - boxes[:, 1]
                )[0].item()

                if (
                    area < 0.1 or iou < confidence_threshold
                ):  # invalid boxes or boxes with large drift (tentative)
                    # we always record the last boxes
                    last_boxes = boxes
                    if compensate == "last":
                        boxes = last_boxes
                    else:
                        boxes = track_bboxes
                    # don't update the KF and last_frame

                else:
                    boxes_cxcyah = bbox_xyxy_to_cxcyah(boxes)
                    boxes_cxcyah = boxes_cxcyah.cpu().numpy()[0]
                    mean, cov = kf.update(mean, cov, boxes_cxcyah)
                    last_frame = c
                    last_boxes = boxes

            pred_sequences.append(token_sequences)
            pred_bboxes.append(boxes)
            pred_scores.append(scores)

            kf_bboxes.append(track_bboxes)
            iou_scores.append(iou)
            """print(
                f"{c}th frame, boxes: {boxes.cpu().tolist()[0]}, with score: {scores.cpu().tolist()[0]}"
            )"""

            if len(scores) == 0:
                continue
            elif (
                c > 0
                and c % update_interval == 0
                and scores[0] > update_threshold
            ):
                # update the prompt box and ref_query_output
                prompt_box = (
                    torch.FloatTensor(boxes[0]).unsqueeze(0).to(image.device)
                )

                ref_image = clip_image[0:1]  # 1, 3, H, W
                ref_image = self.crop_image_from_boxes(
                    ref_image,
                    prompt_box,
                    self.template_size,
                    self.template_boundary,
                )

                if "videoswin" not in self.vit_name:
                    with self.maybe_autocast():
                        ref_feats = self.visual_encoder(ref_image)
                        ref_embeds = self.ln_vision(ref_feats)  # 1, 1+HW, C

                    if self.cross_frame_fusion == "late":
                        ref_query_output = self.Qformer.bert(
                            query_embeds=query_tokens,
                            encoder_hidden_states=ref_embeds,
                            encoder_attention_mask=ref_image_atts,
                            return_dict=True,
                        )

                else:
                    ref_image = F.interpolate(
                        ref_image,
                        size=(image.shape[2], image.shape[3]),
                        mode="bicubic",
                    )

                if self.box_in_prompt:
                    normalized_prompt_box = prompt_box.clone()
                    normalized_prompt_box[:, 0::2] /= width
                    normalized_prompt_box[:, 1::2] /= height

                    prompt_text = task + ". "
                    if (
                        self.language_in_prompt
                        and samples["template_query_score"][0]
                        > self.score_threshold
                    ):
                        prompt_text = task + ". " + samples["blip2query"][0]

                    # update the prompt text
                    input_tokens = tokenizer(
                        [prompt_text],
                        normalized_prompt_box[:1].tolist(),
                        None,
                        None,
                        tokenize_prompt=True,
                    )[0]

                    input_tokens = Dict(
                        {k: v.to(image.device) for k, v in input_tokens.items()}
                    )

        pred_sequences = torch.cat(pred_sequences, dim=1)
        pred_bboxes = torch.cat(pred_bboxes, dim=0)
        pred_scores = torch.cat(pred_scores, dim=0)
        kf_bboxes = torch.cat(kf_bboxes, dim=0)
        iou_scores = torch.cat(iou_scores, dim=0)

        return (pred_sequences, pred_bboxes, pred_scores, kf_bboxes, iou_scores)

    @torch.no_grad()
    def label_points(self, points, gt_segments, gt_labels):
        # concat points on all fpn levels List[T x 4] -> F T x 4
        # This is shared for all samples in the mini-batch
        num_levels = len(points)
        concat_points = torch.cat(points, dim=0)
        gt_cls, gt_offset = [], []

        # loop over each video sample
        for gt_segment, gt_label in zip(gt_segments, gt_labels):
            cls_targets, reg_targets = self.label_points_single_video(
                concat_points, gt_segment, gt_label
            )
            # append to list (len = # images, each of size FT x C)
            gt_cls.append(cls_targets)
            gt_offset.append(reg_targets)

        return gt_cls, gt_offset

    @torch.no_grad()
    def label_points_single_video(self, concat_points, gt_segment, gt_label):
        # concat_points : F T x 4 (t, regression range, stride)
        # gt_segment : N (#Events) x 2
        # gt_label : N (#Events) x 1
        num_pts = concat_points.shape[0]
        num_gts = gt_segment.shape[0]

        # corner case where current sample does not have actions
        if num_gts == 0:
            cls_targets = gt_segment.new_full((num_pts, self.num_classes), 0)
            reg_targets = gt_segment.new_zeros((num_pts, 2))
            return cls_targets, reg_targets

        # compute the lengths of all segments -> F T x N
        lens = gt_segment[:, 1] - gt_segment[:, 0]
        lens = lens[None, :].repeat(num_pts, 1)

        # compute the distance of every point to each segment boundary
        # auto broadcasting for all reg target-> F T x N x2
        gt_segs = gt_segment[None].expand(num_pts, num_gts, 2)
        left = concat_points[:, 0, None] - gt_segs[:, :, 0]
        right = gt_segs[:, :, 1] - concat_points[:, 0, None]
        reg_targets = torch.stack((left, right), dim=-1)

        if self.train_center_sample == "radius":
            # center of all segments F T x N
            center_pts = 0.5 * (gt_segs[:, :, 0] + gt_segs[:, :, 1])
            # center sampling based on stride radius
            # compute the new boundaries:
            # concat_points[:, 3] stores the stride
            t_mins = (
                center_pts
                - concat_points[:, 3, None] * self.train_center_sample_radius
            )
            t_maxs = (
                center_pts
                + concat_points[:, 3, None] * self.train_center_sample_radius
            )
            # prevent t_mins / maxs from over-running the action boundary
            # left: torch.maximum(t_mins, gt_segs[:, :, 0])
            # right: torch.minimum(t_maxs, gt_segs[:, :, 1])
            # F T x N (distance to the new boundary)
            cb_dist_left = concat_points[:, 0, None] - torch.maximum(
                t_mins, gt_segs[:, :, 0]
            )
            cb_dist_right = (
                torch.minimum(t_maxs, gt_segs[:, :, 1])
                - concat_points[:, 0, None]
            )
            # F T x N x 2
            center_seg = torch.stack((cb_dist_left, cb_dist_right), -1)
            # F T x N
            inside_gt_seg_mask = center_seg.min(-1)[0] > 0
        else:
            # inside an gt action
            inside_gt_seg_mask = reg_targets.min(-1)[0] > 0

        # limit the regression range for each location
        max_regress_distance = reg_targets.max(-1)[0]
        # F T x N
        inside_regress_range = torch.logical_and(
            (max_regress_distance >= concat_points[:, 1, None]),
            (max_regress_distance <= concat_points[:, 2, None]),
        )

        # if there are still more than one actions for one moment
        # pick the one with the shortest duration (easiest to regress)
        lens.masked_fill_(inside_gt_seg_mask == 0, float("inf"))
        lens.masked_fill_(inside_regress_range == 0, float("inf"))
        # F T x N -> F T
        min_len, min_len_inds = lens.min(dim=1)

        # corner case: multiple actions with very similar durations (e.g., THUMOS14)
        min_len_mask = torch.logical_and(
            (lens <= (min_len[:, None] + 1e-3)), (lens < float("inf"))
        ).to(reg_targets.dtype)

        # cls_targets: F T x C; reg_targets F T x 2
        gt_label_one_hot = F.one_hot(gt_label, self.num_classes).to(
            reg_targets.dtype
        )
        cls_targets = min_len_mask @ gt_label_one_hot
        # to prevent multiple GT actions with the same label and boundaries
        cls_targets.clamp_(min=0.0, max=1.0)
        # OK to use min_len_inds
        reg_targets = reg_targets[range(num_pts), min_len_inds]
        # normalization based on stride
        reg_targets /= concat_points[:, 3, None]

        return cls_targets, reg_targets

    def losses(
        self, fpn_masks, out_cls_logits, out_offsets, gt_cls_labels, gt_offsets
    ):
        # fpn_masks, out_*: F (List) [B, T_i, C]
        # gt_* : B (list) [F T, C]
        # fpn_masks -> (B, FT)

        """
        1 torch.Size([1, 200])
        1 torch.Size([1, 200, 200])
        1 torch.Size([1, 200, 2])
        3 torch.Size([200, 200])
        3 torch.Size([200, 2])
        """
        valid_mask = torch.cat(fpn_masks, dim=1)

        # 1. classification loss
        # stack the list -> (B, FT) -> (# Valid, )
        gt_cls = torch.stack(gt_cls_labels)
        pos_mask = torch.logical_and((gt_cls.sum(-1) > 0), valid_mask)

        # cat the predicted offsets -> (B, FT, 2 (xC)) -> # (#Pos, 2 (xC))
        pred_offsets = torch.cat(out_offsets, dim=1)[pos_mask]
        gt_offsets = torch.stack(gt_offsets)[pos_mask]

        # update the loss normalizer
        num_pos = pos_mask.sum().item()
        self.loss_normalizer = (
            self.loss_normalizer_momentum * self.loss_normalizer
            + (1 - self.loss_normalizer_momentum) * max(num_pos, 1)
        )

        # gt_cls is already one hot encoded now, simply masking out
        gt_target = gt_cls[valid_mask]

        # optinal label smoothing
        gt_target *= 1 - self.train_label_smoothing
        gt_target += self.train_label_smoothing / (self.num_classes + 1)

        # focal loss
        cls_loss = sigmoid_focal_loss(
            torch.cat(out_cls_logits, dim=1)[valid_mask],
            gt_target,
            reduction="sum",
        )
        cls_loss /= self.loss_normalizer

        # 2. regression using IoU/GIoU loss (defined on positive samples)
        if num_pos == 0:
            reg_loss = 0 * pred_offsets.sum()
        else:
            # giou loss defined on positive samples
            reg_loss = ctr_diou_loss_1d(
                pred_offsets, gt_offsets, reduction="sum"
            )
            reg_loss /= self.loss_normalizer

        if self.train_loss_weight > 0:
            loss_weight = self.train_loss_weight
        else:
            loss_weight = cls_loss.detach() / max(reg_loss.item(), 0.01)

        # return a dict of losses
        final_loss = cls_loss + reg_loss * loss_weight
        return {
            "cls_loss": cls_loss,
            "reg_loss": reg_loss,
            "final_loss": final_loss,
        }

    @torch.no_grad()
    def inference(self, points, fpn_masks, out_cls_logits, out_offsets):
        # video_list B (list) [dict]
        # points F (list) [T_i, 4]
        # fpn_masks, out_*: F (List) [B, T_i, C]

        results = []
        assert len(fpn_masks) == len(out_cls_logits) == len(out_offsets)
        # 2: inference on each single video and gather the results
        # upto this point, all results use timestamps defined on feature grids
        for idx in range(1):
            # gather per-video outputs
            cls_logits_per_vid = [x[idx] for x in out_cls_logits]
            offsets_per_vid = [x[idx] for x in out_offsets]
            fpn_masks_per_vid = [x[idx] for x in fpn_masks]
            # inference on a single video (should always be the case)
            results_per_vid = self.inference_single_video(
                points, fpn_masks_per_vid, cls_logits_per_vid, offsets_per_vid
            )
            results.append(results_per_vid)

        # step 3: postprocssing
        # results = self.postprocessing(results)

        return results

    @torch.no_grad()
    def inference_single_video(
        self,
        points,
        fpn_masks,
        out_cls_logits,
        out_offsets,
    ):
        # points F (list) [T_i, 4]
        # fpn_masks, out_*: F (List) [T_i, C]
        segs_all = []
        scores_all = []
        cls_idxs_all = []

        # loop over fpn levels
        for cls_i, offsets_i, pts_i, mask_i in zip(
            out_cls_logits, out_offsets, points, fpn_masks
        ):
            # sigmoid normalization for output logits
            pred_prob = (cls_i.sigmoid() * mask_i.unsqueeze(-1)).flatten()

            # Apply filtering to make NMS faster following detectron2
            # 1. Keep seg with confidence score > a threshold
            keep_idxs1 = pred_prob > self.test_pre_nms_thresh
            pred_prob = pred_prob[keep_idxs1]
            topk_idxs = keep_idxs1.nonzero(as_tuple=True)[0]

            # 2. Keep top k top scoring boxes only
            num_topk = min(self.test_pre_nms_topk, topk_idxs.size(0))
            pred_prob, idxs = pred_prob.sort(descending=True)
            pred_prob = pred_prob[:num_topk].clone()
            topk_idxs = topk_idxs[idxs[:num_topk]].clone()

            # fix a warning in pytorch 1.9
            pt_idxs = torch.div(
                topk_idxs, self.num_classes, rounding_mode="floor"
            )
            cls_idxs = torch.fmod(topk_idxs, self.num_classes)

            # 3. gather predicted offsets
            offsets = offsets_i[pt_idxs]
            pts = pts_i[pt_idxs]

            # 4. compute predicted segments (denorm by stride for output offsets)
            seg_left = pts[:, 0] - offsets[:, 0] * pts[:, 3]
            seg_right = pts[:, 0] + offsets[:, 1] * pts[:, 3]
            pred_segs = torch.stack((seg_left, seg_right), -1)

            # 5. Keep seg with duration > a threshold (relative to feature grids)
            seg_areas = seg_right - seg_left
            keep_idxs2 = seg_areas > self.test_duration_thresh

            # *_all : N (filtered # of segments) x 2 / 1
            segs_all.append(pred_segs[keep_idxs2])
            scores_all.append(pred_prob[keep_idxs2])
            cls_idxs_all.append(cls_idxs[keep_idxs2])

        # cat along the FPN levels (F N_i, C)
        segs_all, scores_all, cls_idxs_all = [
            torch.cat(x) for x in [segs_all, scores_all, cls_idxs_all]
        ]
        results = {
            "segments": segs_all,
            "scores": scores_all,
            "labels": cls_idxs_all,
        }

        return results

    def _lemmatize(self, answers):
        def apply(answer):
            doc = self.lemmatizer(answer)

            words = []
            for token in doc:
                if token.pos_ in ["NOUN", "VERB"]:
                    words.append(token.lemma_)
                else:
                    words.append(token.text)
            answer = " ".join(words)

            return answer

        return [apply(answer) for answer in answers]

    @property
    def lemmatizer(self):
        if self._lemmatizer is None:
            try:
                import spacy

                self._lemmatizer = spacy.load("en_core_web_sm")
            except ImportError:
                logging.error(
                    """
                    Please install spacy and en_core_web_sm model to apply lemmatization.
                    python -m spacy download en_core_web_sm
                    OR
                    import spacy.cli
                    spacy.cli.download("en_core_web_sm")
                    """
                )
                exit(1)

        return self._lemmatizer

    @classmethod
    def from_config(cls, cfg):
        vit_model = cfg.get("vit_model", "eva_clip_g")
        img_size = cfg.get("image_size")
        num_query_token = cfg.get("num_query_token")
        llm_model = cfg.llm_model

        drop_path_rate = cfg.get("drop_path_rate", 0)
        use_grad_checkpoint = cfg.get("use_grad_checkpoint", False)
        vit_precision = cfg.get("vit_precision", "fp16")
        freeze_vit = cfg.get("freeze_vit", True)

        prompt = cfg.get("prompt", "")
        max_txt_len = cfg.get("max_txt_len", 32)

        apply_lemmatizer = cfg.get("apply_lemmatizer", False)

        num_frms = cfg.num_frms
        temporal_downrate = cfg.get("temporal_downrate", 1)

        freeze_qformer = cfg.freeze_qformer
        freeze_llm = cfg.freeze_llm
        nollminit = cfg.get("nollminit", False)
        tune_llm_fc = cfg.get("tune_llm_fc", False)
        freeze_audio = cfg.get("freeze_audio", False)

        temporal_depth = cfg.get("temporal_depth", 12)
        use_frame_mask = cfg.get("use_frame_mask", False)

        vocab_size = cfg["vocab_size"]
        roi_size = cfg.get("roi_size", 7)
        cross_frame_fusion = cfg.get("cross_frame_fusion", "late")
        weight_token_loss = cfg.get("weight_token_loss", False)

        use_iou_loss = cfg.get("use_iou_loss", False)
        iou_loss_weight = cfg.get("iou_loss_weight", 1.0)

        box_in_prompt = cfg.get("box_in_prompt", True)
        language_in_prompt = cfg.get("language_in_prompt", False)

        box_in_query = cfg.get("box_in_query", False)
        language_in_query = cfg.get("language_in_query", False)

        box2query = cfg.get("box2query", "add")
        template_size = cfg.get("template_size", 128)
        template_boundary = cfg.get("template_boundary", 0)
        use_template_feat = cfg.get("use_template_feat", False)

        use_audio = cfg.get("use_audio", False)
        fuse_audio = cfg.get("fuse_audio", "concat")
        init_3d_patch_embed = cfg.get("init_3d_patch_embed", True)
        init_2d_patch_embed = cfg.get("init_2d_patch_embed", False)
        video_backbone_pretrained = cfg.get("video_backbone_pretrained", "k400")

        vl_fuse = cfg.get("vl_fuse", False)
        vl_fuse_layer = cfg.get("vl_fuse_layer", 1)

        model = cls(
            vit_model=vit_model,
            img_size=img_size,
            drop_path_rate=drop_path_rate,
            use_grad_checkpoint=use_grad_checkpoint,
            vit_precision=vit_precision,
            freeze_vit=freeze_vit,
            num_query_token=num_query_token,
            llm_model=llm_model,
            prompt=prompt,
            max_txt_len=max_txt_len,
            apply_lemmatizer=apply_lemmatizer,
            num_frms=num_frms,
            temporal_downrate=temporal_downrate,
            temporal_depth=temporal_depth,
            freeze_qformer=freeze_qformer,
            freeze_llm=freeze_llm,
            nollminit=nollminit,
            tune_llm_fc=tune_llm_fc,
            freeze_audio=freeze_audio,
            vocab_size=vocab_size,
            roi_size=roi_size,
            cross_frame_fusion=cross_frame_fusion,
            weight_token_loss=weight_token_loss,
            use_iou_loss=use_iou_loss,
            iou_loss_weight=iou_loss_weight,
            box_in_prompt=box_in_prompt,
            language_in_prompt=language_in_prompt,
            box_in_query=box_in_query,
            language_in_query=language_in_query,
            box2query=box2query,
            template_size=template_size,
            template_boundary=template_boundary,
            use_template_feat=use_template_feat,
            use_frame_mask=use_frame_mask,
            use_audio=use_audio,
            fuse_audio=fuse_audio,
            init_3d_patch_embed=init_3d_patch_embed,
            init_2d_patch_embed=init_2d_patch_embed,
            video_backbone_pretrained=video_backbone_pretrained,
            vl_fuse=vl_fuse,
            vl_fuse_layer=vl_fuse_layer,
        )

        model.load_checkpoint_from_config(cfg)

        return model
