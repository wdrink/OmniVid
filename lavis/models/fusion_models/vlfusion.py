import torch
import torch.utils.checkpoint as checkpoint
from transformers.models.bert.modeling_bert import BertPreTrainedModel
from transformers.modeling_utils import apply_chunking_to_forward
from .fuse_helper import BiAttentionBlockForCheckpoint


class VLFuse(torch.nn.Module):
    """
    Early Fusion Module
    """

    def __init__(
        self,
        lang_dim,
        img_dim,
        n_layers,
        embed_dim,
        n_head=8,
        use_checkpoint=False,
    ):
        super(VLFuse, self).__init__()
        self.lang_dim = lang_dim
        self.img_dim = img_dim
        self.n_layers = n_layers
        self.n_head = n_head
        self.embed_dim = embed_dim

        self.use_checkpoint = use_checkpoint

        # early fusion module
        # bi-direction (text->image, image->text)
        self.b_attn = BiAttentionBlockForCheckpoint(
            v_dim=self.img_dim,  # 256
            l_dim=self.lang_dim,  # 768
            embed_dim=self.embed_dim,  # 2048
            num_heads=self.n_head,  # 8
            dropout=0.1,
            drop_path=0.0,
            init_values=1.0 / 6,
        )

    def forward(
        self,
        visual_features,
        language_hidden,
        language_mask,
        use_fused_language_embeddings,
        task=None,
    ):
        if self.use_checkpoint:
            fused_visual_features, language_features = checkpoint.checkpoint(
                self.b_attn,
                visual_features,
                language_hidden,
                language_mask,
                task,
            )
        else:
            fused_visual_features, language_features = self.b_attn(
                visual_features, language_hidden, language_mask, task
            )

        if use_fused_language_embeddings:
            return fused_visual_features, language_features
        else:
            return fused_visual_features, language_hidden


if __name__ == "__main__":
    visual_features = torch.rand(2, 100, 768)
    language_hidden = torch.rand(2, 256, 768)
    language_mask = torch.ones(2, 256).bool()
    model = VLFuse(768, 768, 2, 2048, 8)
    fused_visual_features, fused_language_hidden = model(
        visual_features, language_hidden, language_mask
    )
    print(fused_visual_features.shape)
    print(fused_language_hidden.shape)
