import torch
import torch.nn as nn
from typing import Tuple, Union, List

from timm.models.layers import trunc_normal_

import clip
from clip_model import load_clip_by_name, convert_weights, CLIP, CLIP_moe

OPENAI_CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
OPENAI_CLIP_STD = (0.26862954, 0.26130258, 0.27577711)

class MLP(nn.Module):
    def __init__(self,
                 visual_dim=1024,
                 mlp_ratio=4,
                 **kwargs
                 ):
        super(MLP, self).__init__()

        self.prompt_proj = nn.Sequential(
            nn.LayerNorm(visual_dim),
            nn.Linear(visual_dim, visual_dim * mlp_ratio),
            nn.ReLU(),
            nn.LayerNorm(visual_dim * mlp_ratio),
            nn.Linear(visual_dim * mlp_ratio, visual_dim)
        )

        self.apply(self._init_weights)

    def forward(self, x):
        '''
        x: (B, D)
        '''
        x = self.prompt_proj(x)
        return x

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)


class Adapter(nn.Module):
    def __init__(self, c_in, reduction=4):
        super(Adapter, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(c_in, c_in // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(c_in // reduction, c_in, bias=False),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.fc(x)
        return x


class CLIP_Adapter_model(CLIP):
    def __init__(self,
                 embed_dim: int,
                 # vision
                 image_resolution: int,
                 vision_layers: Union[Tuple[int, int, int, int], int],
                 vision_width: int,
                 vision_patch_size: int,
                 # text
                 context_length: int,
                 vocab_size: int,
                 transformer_width: int,
                 transformer_heads: int,
                 transformer_layers: int
                 ):
        super(CLIP_Adapter_model, self).__init__(embed_dim,image_resolution,vision_layers,
                                                 vision_width,vision_patch_size,context_length,
                                                 vocab_size,transformer_width,transformer_heads,
                                                 transformer_layers)

    def forward(self, image, text, pos_embedding=False, text_features=None):
        image_features = self.encode_image(image, pos_embedding) # [bs, 1024]
        if text_features is None:
            text_features = self.encode_text(text)

        return image_features, text_features


class CLIPmoe_Adapter_model(CLIP_moe):
    def __init__(self,
                 embed_dim: int,
                 # vision
                 image_resolution: int,
                 vision_layers: Union[Tuple[int, int, int, int], int],
                 vision_width: int,
                 vision_patch_size: int,
                 # text
                 context_length: int,
                 vocab_size: int,
                 transformer_width: int,
                 transformer_heads: int,
                 transformer_layers: int
                 ):
        super(CLIPmoe_Adapter_model, self).__init__(embed_dim,image_resolution,vision_layers,
                                                 vision_width,vision_patch_size,context_length,
                                                 vocab_size,transformer_width,transformer_heads,
                                                 transformer_layers)

    def forward(self, image, text, pos_embedding=False, text_features=None):
        image_features = self.encode_image(image, pos_embedding) # [bs, 1024]
        if text_features is None:
            text_features = self.encode_text(text)

        return image_features, text_features


class Gamma_MoAE(nn.Module):
    def __init__(self,
                 model_type='clip_adapter',
                 backbone='RN50',
                 pretrained=True,
                 pos_embedding=True,
                 clip_finetune=True,
                 sample_size=224,
                 embedding_dim=256,
                 num_experts = 3,
                 num_ft_layer = 6,
                 ) -> None:
        super().__init__()

        self.clip_model = load_clip_by_name(backbone, 'cpu', 
                                             build_clip_type = build_model_clipmoe_adapter,
                                             jit = False,
                                             clip_finetune = clip_finetune)

        # clip_model is a list, len = 1
        # clip_model[0] is the true clip model
        for p in self.clip_model.parameters():
            p.requires_grad = False

        num_experts = [num_experts] * 12
        moe_layers_idx = [i for i in range(num_ft_layer, 12)]
        self.clip_model.initialize_moe_modules(num_experts, moe_layers_idx)

        self.prompt_pairs = clip.tokenize([
            'bad image','poor image','fair image','good image','perfect image'
        ])

        self.model_type = model_type
        self.pos_embedding = pos_embedding
        self.sample_size=sample_size

        self.default_mean = torch.Tensor(OPENAI_CLIP_MEAN).view(1, 3, 1, 1)
        self.default_std = torch.Tensor(OPENAI_CLIP_STD).view(1, 3, 1, 1)

        self.adapter=Adapter(embedding_dim*2,2)
        self.score_level=nn.Parameter(torch.tensor([0.2,0.4,0.6,0.8,1.0]))
    
    def random_crop_test(self, x, sample_num=10):
        b, c, h, w = x.shape
        th = tw = self.sample_size
        cropped_x = []
        for s in range(sample_num):
            i = torch.randint(0, h - th + 1, size=(1, )).item()
            j = torch.randint(0, w - tw + 1, size=(1, )).item()
            cropped_x.append(x[:, :, i:i + th, j:j + tw])
        cropped_x = torch.cat(cropped_x, dim=0)
        results = self.forward_image(cropped_x)
        results = results.reshape(sample_num, b).mean(dim=0)
        return results.unsqueeze(-1)
    
    def random_5crop_test(self, x, sample_num=5):
        b, c, height, width = x.shape
        th = tw = self.sample_size
        cropped_x = []

        # 计算中心裁剪区域的坐标
        center_left = (width - tw) // 2
        center_upper = (height - th) // 2
        center_right = center_left + tw
        center_lower = center_upper + th

        # 计算四个角裁剪区域的坐标
        top_left = (0, 0, th, tw)
        top_right = (width - tw, 0, width, tw)
        bottom_left = (0, height - th, tw, height)
        bottom_right = (width - tw, height - th, width, height)

        cropped_x.append(x[:, :, center_upper:center_lower, center_left:center_right])
        cropped_x.append(x[:, :, top_left[1]:top_left[3], top_left[0]:top_left[2]])
        cropped_x.append(x[:, :, top_right[1]:top_right[3], top_right[0]:top_right[2]])
        cropped_x.append(x[:, :, bottom_left[1]:bottom_left[3], bottom_left[0]:bottom_left[2]])
        cropped_x.append(x[:, :, bottom_right[1]:bottom_right[3], bottom_right[0]:bottom_right[2]])

        cropped_x = torch.cat(cropped_x, dim=0)
        results = self.forward_image(cropped_x)
        results = results.reshape(sample_num, b).mean(dim=0)
        return results.unsqueeze(-1)

    def forward_image(self, x):
        # preprocess image x.shape [bs, 3, H, W]
        x = (x - self.default_mean.to(x)) / self.default_std.to(x)
        clip_model = self.clip_model.to(x) # change device

        prompts = self.prompt_pairs.to(x.device)
        image_features, text_features = clip_model(x, prompts, pos_embedding=self.pos_embedding)

        x = self.adapter(image_features)
        image_features = image_features+x

        # normalized features
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        # cosine similarity as logits
        logit_scale = self.clip_model.logit_scale.exp()
        logits_per_image = logit_scale * image_features @ text_features.t()

        probs = logits_per_image.softmax(dim=-1)
        probs = probs * self.score_level
        probs = torch.sum(probs,dim=-1).reshape(probs.shape[0],-1)

        return probs
    
    def forward(self, x):
        if self.training:
            return self.forward_image(x)
        else:
            score = self.random_crop_test(x)
            return score


def get_text_prompt(img_path):
    """
        img_path: a batch of image path, len(img_path) = batch_size
    """
    text_prompt = []
    for i in img_path:
        if ('koniq10k' in i) or ('ChallengeDB_release') in i or ('kadid10k' in i) or ('databaserelease2' in i) or ('CSIQ' in i) or ('spaq' in i) or ('tid2013' in i):
            text_prompt.append("natural quality image")
        elif ('agiqa3k' in i):
            text_prompt.append("AI-generated quality image")
        elif ('UWIQA' in i):
            text_prompt.append("underwater quality image")
        elif ('GFIQA-20k' in i):
            text_prompt.append("face quality image")
        elif ('aadb' in i) or ('ava' in i):
            text_prompt.append("natural aesthetic image")
        else:
            raise ValueError(f"The image does not belong to five predefined tpyes, place validate the image path: {i}.")

    return text_prompt


class Gamma_MoAE_SDP(nn.Module):
    def __init__(self,
                 model_type='clip_adapter',
                 backbone='RN50',
                 pretrained=True,
                 pos_embedding=True,
                 clip_finetune=True,
                 sample_size=224,
                 embedding_dim=256,
                 num_experts = 3,
                 num_ft_layer = 6,
                 ) -> None:
        super().__init__()

        self.clip_model = load_clip_by_name(backbone, 'cpu', 
                                             build_clip_type = build_model_clipmoe_adapter,
                                             jit = False,
                                             clip_finetune = clip_finetune)

        # clip_model is a list, len = 1
        # clip_model[0] is the true clip model
        for p in self.clip_model.parameters():
            p.requires_grad = False

        num_experts = [num_experts] * 12
        moe_layers_idx = [i for i in range(num_ft_layer, 12)]
        self.clip_model.initialize_moe_modules(num_experts, moe_layers_idx)

        self.prompt_pairs_naive = clip.tokenize([
            'bad image','poor image','fair image','good image','perfect image'
        ])

        self.prompt_pairs_SDP = {
            "natural quality image": [
            'natural bad-quality image','natural poor-quality image','natural fair-quality image','natural good-quality image','natural perfect-quality image'
        ],
            "AI-generated quality image": [
            'AI-generated bad-quality image','AI-generated poor-quality image','AI-generated fair-quality image','AI-generated good-quality image','AI-generated perfect-quality image'
        ],
            "underwater quality image": [
            'underwater bad-quality image','underwater poor-quality image','underwater fair-quality image','underwater good-quality image','underwater perfect-quality image'
        ],
            "face quality image": [
            'face bad-quality image','face poor-quality image','face fair-quality image','face good-quality image','face perfect-quality image'
        ],
            "natural aesthetic image": [
            'natural bad-aesthetics image','natural poor-aesthetics image','natural fair-aesthetics image','natural good-aesthetics image','natural perfect-aesthetics image'
        ]
        }

        self.model_type = model_type
        self.pos_embedding = pos_embedding
        self.sample_size=sample_size

        self.default_mean = torch.Tensor(OPENAI_CLIP_MEAN).view(1, 3, 1, 1)
        self.default_std = torch.Tensor(OPENAI_CLIP_STD).view(1, 3, 1, 1)

        self.adapter=Adapter(embedding_dim*2,2)
        self.score_level=nn.Parameter(torch.tensor([0.2,0.4,0.6,0.8,1.0]))
    
    def random_crop_test(self, x, sample_num=10):
        self.sample_num = sample_num
        x, img_path = x[0], x[1]
        b, c, h, w = x.shape
        th = tw = self.sample_size
        cropped_x = []
        for s in range(sample_num):
            i = torch.randint(0, h - th + 1, size=(1, )).item()
            j = torch.randint(0, w - tw + 1, size=(1, )).item()
            cropped_x.append(x[:, :, i:i + th, j:j + tw])
        cropped_x = torch.cat(cropped_x, dim=0)
        results = self.forward_image((cropped_x, img_path), if_train=False)
        results = results.reshape(sample_num, b).mean(dim=0)
        return results.unsqueeze(-1)

    def random_crop_test_naive(self, x, sample_num=10):
        self.sample_num = sample_num
        x, img_path = x[0], x[1]
        b, c, h, w = x.shape
        th = tw = self.sample_size
        cropped_x = []
        for s in range(sample_num):
            i = torch.randint(0, h - th + 1, size=(1, )).item()
            j = torch.randint(0, w - tw + 1, size=(1, )).item()
            cropped_x.append(x[:, :, i:i + th, j:j + tw])
        cropped_x = torch.cat(cropped_x, dim=0)
        results = self.forward_image_wo_diff((cropped_x, img_path))
        results = results.reshape(sample_num, b).mean(dim=0)
        return results.unsqueeze(-1)
    
    def random_5crop_test(self, x, sample_num=5):
        self.sample_num = sample_num
        b, c, height, width = x.shape
        th = tw = self.sample_size
        cropped_x = []

        # 计算中心裁剪区域的坐标
        center_left = (width - tw) // 2
        center_upper = (height - th) // 2
        center_right = center_left + tw
        center_lower = center_upper + th

        # 计算四个角裁剪区域的坐标
        top_left = (0, 0, th, tw)
        top_right = (width - tw, 0, width, tw)
        bottom_left = (0, height - th, tw, height)
        bottom_right = (width - tw, height - th, width, height)

        cropped_x.append(x[:, :, center_upper:center_lower, center_left:center_right])
        cropped_x.append(x[:, :, top_left[1]:top_left[3], top_left[0]:top_left[2]])
        cropped_x.append(x[:, :, top_right[1]:top_right[3], top_right[0]:top_right[2]])
        cropped_x.append(x[:, :, bottom_left[1]:bottom_left[3], bottom_left[0]:bottom_left[2]])
        cropped_x.append(x[:, :, bottom_right[1]:bottom_right[3], bottom_right[0]:bottom_right[2]])

        cropped_x = torch.cat(cropped_x, dim=0)
        results = self.forward_image(cropped_x)
        results = results.reshape(sample_num, b).mean(dim=0)
        return results.unsqueeze(-1)

    def forward_image_naive(self, x):
        # preprocess image x.shape [bs, 3, H, W]
        x, img_path = x[0], x[1]
        x = (x - self.default_mean.to(x)) / self.default_std.to(x)
        clip_model = self.clip_model.to(x) # change device

        prompts = self.prompt_pairs_wo_diff.to(x.device)
        image_features, text_features = clip_model(x, prompts, pos_embedding=self.pos_embedding)

        x = self.adapter(image_features)
        image_features = image_features+x

        # normalized features
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        # cosine similarity as logits
        logit_scale = self.clip_model.logit_scale.exp()
        logits_per_image = logit_scale * image_features @ text_features.t()

        probs = logits_per_image.softmax(dim=-1)
        probs = probs * self.score_level
        probs = torch.sum(probs,dim=-1).reshape(probs.shape[0],-1)

        return probs

    def forward_image(self, x, if_train = True):
        # preprocess image x.shape [bs, 3, H, W]
        x, img_path = x[0], x[1]
        x = (x - self.default_mean.to(x)) / self.default_std.to(x)
        clip_model = self.clip_model.to(x) # change device
        text_prompt = get_text_prompt(img_path)
        all_prompts = []
        for i in text_prompt:
            prompts = self.prompt_pairs_SDP[i]
            all_prompts.extend(prompts)
        all_prompts = clip.tokenize(all_prompts).to(x.device)
        image_features, text_features = clip_model(x, all_prompts, pos_embedding=self.pos_embedding)

        x = self.adapter(image_features)
        image_features = image_features+x

        # normalized features
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        # cosine similarity as logits
        logit_scale = self.clip_model.logit_scale.exp()
        if not if_train:
            B, C = image_features.shape
            image_features = image_features.reshape(-1, self.sample_num ,C)
            text_features = text_features.reshape(-1, 5 ,C)
            sim = logit_scale * torch.matmul(image_features, text_features.transpose(1, 2))
            logits_per_image = sim.reshape(-1, 5)
        else:
            all_sim = []
            for i in range(image_features.shape[0]):
                one_img_fea = image_features[i].unsqueeze(0)
                one_text_fea = text_features[i*5:(i+1)*5]
                sim = logit_scale * one_img_fea @ one_text_fea.t()
                all_sim.append(sim)
            logits_per_image = torch.cat(all_sim, dim = 0)
        # logits_per_image = logit_scale * image_features @ text_features.t()

        probs = logits_per_image.softmax(dim=-1)
        probs = probs * self.score_level
        probs = torch.sum(probs,dim=-1).reshape(probs.shape[0],-1)

        return probs
    
    def forward(self, x):
        if self.training:
            return self.forward_image(x)
            # return self.forward_image_naive(x) # use naive prompt (w/o SDP)

        else:
            return self.random_crop_test(x)
            # return self.random_crop_test_naive(x) # use naive prompt (w/o SDP)


def build_model_clip_adapter(state_dict: dict):
    vit = "visual.proj" in state_dict

    if vit:
        vision_width = state_dict["visual.conv1.weight"].shape[0]
        vision_layers = len([k for k in state_dict.keys() if k.startswith(
            "visual.") and k.endswith(".attn.in_proj_weight")])
        vision_patch_size = state_dict["visual.conv1.weight"].shape[-1]
        grid_size = round((state_dict["visual.positional_embedding"].shape[0] - 1) ** 0.5)
        image_resolution = vision_patch_size * grid_size
    else:
        counts: list = [len(set(k.split(".")[2]
                            for k in state_dict if k.startswith(f"visual.layer{b}"))) for b in [1, 2, 3, 4]]
        vision_layers = tuple(counts)
        vision_width = state_dict["visual.layer1.0.conv1.weight"].shape[0]
        output_width = round((state_dict["visual.attnpool.positional_embedding"].shape[0] - 1) ** 0.5)
        vision_patch_size = None
        assert output_width ** 2 + 1 == state_dict["visual.attnpool.positional_embedding"].shape[0]
        image_resolution = output_width * 32

    embed_dim = state_dict["text_projection"].shape[1]
    context_length = state_dict["positional_embedding"].shape[0]
    vocab_size = state_dict["token_embedding.weight"].shape[0]
    transformer_width = state_dict["ln_final.weight"].shape[0]
    transformer_heads = transformer_width // 64
    transformer_layers = len(set(k.split(".")[2] for k in state_dict if k.startswith(f"transformer.resblocks")))

    model = CLIP_Adapter_model(
        embed_dim,
        image_resolution, vision_layers, vision_width, vision_patch_size,
        context_length, vocab_size, transformer_width, transformer_heads, transformer_layers
    )

    for key in ["input_resolution", "context_length", "vocab_size"]:
        if key in state_dict:
            del state_dict[key]

    convert_weights(model)
    model.load_state_dict(state_dict)
    return model.eval()

def build_model_clipmoe_adapter(state_dict: dict):
    vit = "visual.proj" in state_dict

    if vit:
        vision_width = state_dict["visual.conv1.weight"].shape[0]
        vision_layers = len([k for k in state_dict.keys() if k.startswith(
            "visual.") and k.endswith(".attn.in_proj_weight")])
        vision_patch_size = state_dict["visual.conv1.weight"].shape[-1]
        grid_size = round((state_dict["visual.positional_embedding"].shape[0] - 1) ** 0.5)
        image_resolution = vision_patch_size * grid_size
    else:
        counts: list = [len(set(k.split(".")[2]
                            for k in state_dict if k.startswith(f"visual.layer{b}"))) for b in [1, 2, 3, 4]]
        vision_layers = tuple(counts)
        vision_width = state_dict["visual.layer1.0.conv1.weight"].shape[0]
        output_width = round((state_dict["visual.attnpool.positional_embedding"].shape[0] - 1) ** 0.5)
        vision_patch_size = None
        assert output_width ** 2 + 1 == state_dict["visual.attnpool.positional_embedding"].shape[0]
        image_resolution = output_width * 32

    embed_dim = state_dict["text_projection"].shape[1]
    context_length = state_dict["positional_embedding"].shape[0]
    vocab_size = state_dict["token_embedding.weight"].shape[0]
    transformer_width = state_dict["ln_final.weight"].shape[0]
    transformer_heads = transformer_width // 64
    transformer_layers = len(set(k.split(".")[2] for k in state_dict if k.startswith(f"transformer.resblocks")))

    model = CLIPmoe_Adapter_model(
        embed_dim,
        image_resolution, vision_layers, vision_width, vision_patch_size,
        context_length, vocab_size, transformer_width, transformer_heads, transformer_layers
    )

    for key in ["input_resolution", "context_length", "vocab_size"]:
        if key in state_dict:
            del state_dict[key]

    convert_weights(model)
    model.load_state_dict(state_dict)
    return model.eval()
