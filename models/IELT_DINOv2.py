import time

import numpy as np
from scipy import ndimage
from torch.nn import CrossEntropyLoss
import torch.nn.functional as F
import torch
import torch.nn as nn
from torch.nn import Linear, Dropout, LayerNorm, Softmax
from models.vit import get_b16_config

from models.DINOv2 import DinoVisionTransformer as DINOv2
from models.dinov2_layers.patch_embed import PatchEmbed as Embeddings
from models.dinov2_layers.block import Block
from functools import partial
import math


def np2th(weights, conv=False):
    """Possibly convert HWIO to OIHW."""
    if conv:
        weights = weights.transpose([3, 2, 0, 1])
    return torch.from_numpy(weights)



class IELT_DINOv2(nn.Module):
    def __init__(self, config, img_size=448, num_classes=200, dataset='cub', smooth_value=0.,
                 loss_alpha=0.4, cam=True, dsm=True, fix=True, update_warm=500,
                 vote_perhead=24, total_num=126, assess=False, forward_features=False, merge_inattentive=False):
        super(IELT_DINOv2, self).__init__()
        self.assess = assess
        self.smooth_value = smooth_value
        self.num_classes = num_classes
        self.loss_alpha = loss_alpha
        self.cam = cam
        self.patch_size = config.patches[0]
        
        self.merge_inattentive = merge_inattentive

        self.embeddings = Embeddings(img_size=img_size, patch_size=config.patches[0], in_chans=3, embed_dim=config.hidden_size)
        self.encoder = IELTEncoder(config, update_warm, vote_perhead, dataset, cam, dsm,
                                   fix, total_num, assess, forward_features=forward_features, merge_inattentive=merge_inattentive)
        self.head = Linear(config.hidden_size, num_classes)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, config.hidden_size))
        self.pos_embed = nn.Parameter(torch.zeros(1, self.patch_size*self.patch_size + 1, config.hidden_size))
        self.softmax = Softmax(dim=-1)
        self.interpolate_antialias=False
        self.interpolate_offset=0.1
        self.forward_features = forward_features
    
        
    
    def interpolate_pos_encoding(self, x, w, h):
        previous_dtype = x.dtype
        npatch = x.shape[1] - 1
        N = self.pos_embed.shape[1] - 1
        if npatch == N and w == h:
            return self.pos_embed
        pos_embed = self.pos_embed.float()
        class_pos_embed = pos_embed[:, 0]
        patch_pos_embed = pos_embed[:, 1:]
        dim = x.shape[-1]
        w0 = w // self.patch_size
        h0 = h // self.patch_size
        M = int(math.sqrt(N))  # Recover the number of patches in each dimension
        assert N == M * M
        kwargs = {}
        if self.interpolate_offset:
            # Historical kludge: add a small number to avoid floating point error in the interpolation, see https://github.com/facebookresearch/dino/issues/8
            # Note: still needed for backward-compatibility, the underlying operators are using both output size and scale factors
            sx = float(w0 + self.interpolate_offset) / M
            sy = float(h0 + self.interpolate_offset) / M
            kwargs["scale_factor"] = (sx, sy)
        else:
            # Simply specify an output size instead of a scale factor
            kwargs["size"] = (w0, h0)
        patch_pos_embed = nn.functional.interpolate(
            patch_pos_embed.reshape(1, M, M, dim).permute(0, 3, 1, 2),
            mode="bicubic",
            antialias=self.interpolate_antialias,
            **kwargs,
        )
        assert (w0, h0) == patch_pos_embed.shape[-2:]
        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
        return torch.cat((class_pos_embed.unsqueeze(0), patch_pos_embed), dim=1).to(previous_dtype)



    def forward(self, x, labels=None, return_cls=False):
            test_mode = False if labels is not None else True

            B, nc, w, h = x.shape
            x = self.embeddings(x)
            x = torch.cat((self.cls_token.expand(x.shape[0], -1, -1), x), dim=1)
            x = x + self.interpolate_pos_encoding(x, w, h)
            if self.assess:
                if return_cls:
                    if self.forward_features:
                        x, xc, assess_list, cls, features = self.encoder(x, test_mode, return_cls=True)
                    else:
                        x, xc, assess_list, cls = self.encoder(x, test_mode, return_cls=True)
                else:
                    if self.forward_features:  
                        x, xc, assess_list, features = self.encoder(x, test_mode, return_cls=False)
                    else:
                        x, xc, assess_list = self.encoder(x, test_mode, return_cls)
            else:
                if return_cls:
                    if self.forward_features:
                        x, xc, cls, features = self.encoder(x, test_mode, return_cls=True)
                    else:
                        x, xc, cls = self.encoder(x, test_mode, return_cls=True)
                else:
                    if self.forward_features:
                        x, xc, features = self.encoder(x, test_mode, return_cls=False)
                    else:
                        x, xc = self.encoder(x, test_mode, return_cls)

            if self.cam:
                complement_logits = self.head(xc)
                probability = self.softmax(complement_logits)
                weight = self.head.weight
                assist_logit = probability * (weight.sum(-1))
                part_logits = self.head(x) + assist_logit
            else:
                part_logits = self.head(x)

            if self.assess:
                if return_cls:
                    return part_logits, xc, assess_list, cls
                return part_logits, assess_list

            elif test_mode:
                if return_cls:
                    if self.forward_features:
                        return part_logits, cls, features
                    return part_logits, cls
                if self.forward_features:
                    return part_logits, features
                return part_logits

            else:
                if self.smooth_value == 0:
                    loss_fct = CrossEntropyLoss()
                else:
                    loss_fct = LabelSmoothing(self.smooth_value)

                if self.cam:
                    loss_p = loss_fct(part_logits.view(-1, self.num_classes), labels.view(-1))
                    loss_c = loss_fct(complement_logits.view(-1, self.num_classes), labels.view(-1))
                    alpha = self.loss_alpha
                    loss = (1 - alpha) * loss_p + alpha * loss_c
                else:
                    loss = loss_fct(part_logits.view(-1, self.num_classes), labels.view(-1))
                
                if return_cls:
                    return part_logits, loss, cls
                return part_logits, loss


    def get_eval_data(self):
        return self.encoder.select_num

    def load_from(self, weights):
        with torch.no_grad():
            nn.init.zeros_(self.head.weight)
            nn.init.zeros_(self.head.bias)
            self.embeddings.proj.weight.copy_(weights["patch_embed.proj.weight"])
            self.embeddings.proj.bias.copy_(weights["patch_embed.proj.bias"])
            #self.pos_embed.copy_(weights["pos_embed"])
            self.cls_token.copy_(weights["cls_token"])
            
            posemb = weights["pos_embed"]
            posemb_new = self.pos_embed
            if posemb.size() == posemb_new.size():
                self.embeddings.position_embeddings.copy_(posemb)
            else:
                ntok_new = posemb_new.size(1)

                posemb_tok, posemb_grid = posemb[:, :1], posemb[0, 1:]
                ntok_new -= 1

                gs_old = int(np.sqrt(len(posemb_grid)))
                gs_new = int(np.sqrt(ntok_new))
                print('load_pretrained: grid-size from %s to %s' % (gs_old, gs_new))
                posemb_grid = posemb_grid.reshape(gs_old, gs_old, -1)

                zoom = (gs_new / gs_old, gs_new / gs_old, 1)
                posemb_grid = ndimage.zoom(posemb_grid, zoom, order=1)
                posemb_grid = posemb_grid.reshape((1, gs_new * gs_new, -1))
                posemb = np.concatenate([posemb_tok, posemb_grid], axis=1)
                self.pos_embed.copy_(np2th(posemb))
            

            for i in range(self.encoder.layer_num - 1):
                # Build a state dictionary for block i
                block_state = {}
                prefix = f"blocks.{i}."
                for k, v in weights.items():
                    if k.startswith(prefix):
                        # Remove the prefix so that the keys match the block's internal names
                        new_key = k[len(prefix):]
                        block_state[new_key] = v
                print(f"Loading weights for block {i} with keys: {list(block_state.keys())}")
                self.encoder.layer[i].load_state_dict(block_state)
                
            


            

            # freeze first 4 layers
            #for i in range(4):
            #    for param in self.encoder.layer[i].parameters():
            #        param.requires_grad = False

class MultiHeadVoting(nn.Module):
    def __init__(self, config, vote_perhead=24, fix=True):
        super(MultiHeadVoting, self).__init__()
        self.fix = fix
        self.num_heads = config.num_heads
        self.vote_perhead = vote_perhead
        self.dilations = [1, 2, 4] ## convolve at multiple scales, d

        if self.fix:
            self.kernel = torch.tensor([[1, 2, 1],
                                        [2, 4, 2],
                                        [1, 2, 1]], device='cuda').unsqueeze(0).unsqueeze(0).half()
        else:
            self.convs = nn.ModuleList([
                nn.Conv2d(1, 1, 3, stride=1, padding=d, dilation=d).half()
                for d in self.dilations
            ])

    def forward(self, x, select_num=None, last=False):
        B, patch_num = x.shape[0], x.shape[3] - 1
        select_num = self.vote_perhead if select_num is None else select_num
        count = torch.zeros((B, patch_num), dtype=torch.int, device='cuda').half()
        score = x[:, :, 0, 1:]
        _, select = torch.topk(score, self.vote_perhead, dim=-1)
        select = select.reshape(B, -1)

        for i, b in enumerate(select):
            count[i, :] += torch.bincount(b, minlength=patch_num)

        if not last:
            count = self.enhace_local(count)

        patch_value, patch_idx = torch.sort(count, dim=-1, descending=True)
        #patch_idx += 1  # Adjust indices to start from 1
        # split into select_num and patch_num-select_num tokens
        selected_tokens = patch_idx[:, :select_num]
        selected_count = count
        unselected_tokens = patch_idx[:, select_num:]
        return selected_tokens, selected_count, unselected_tokens

    def enhace_local(self, count):
        B, patch_num = count.shape[0], count.shape[1]
        H = math.ceil(math.sqrt(patch_num))
        count = count.reshape(B, H, H).unsqueeze(1) #  (B, 1, H, H)

        if self.fix:
            outputs = []
            for d in self.dilations:
                output = F.conv2d(count, self.kernel, stride=1, padding=d, dilation=d)
                outputs.append(output)
            enhanced_count = torch.max(torch.stack(outputs), dim=0)[0]  #(B, 1, H, H)
        else:
            outputs = [conv(count) for conv in self.convs]
            enhanced_count = torch.max(torch.stack(outputs), dim=0)[0]

        enhanced_count = enhanced_count.squeeze(1).reshape(B, -1)
        return enhanced_count

class CrossLayerRefinement(nn.Module):
    def __init__(self, config, clr_layer):
        super(CrossLayerRefinement, self).__init__()
        self.clr_layer = clr_layer
        self.clr_norm = LayerNorm(config.hidden_size, eps=1e-6)

    def forward(self, x, cls):
        out = [torch.stack(token) for token in x]
        out = torch.stack(out).squeeze(1)
        out = torch.cat((cls, out), dim=1)
        out, weights = self.clr_layer(out)
        out = self.clr_norm(out)
        return out, weights

class TokenMerger(nn.Module):
    def __init__(self, config):
        super(TokenMerger, self).__init__()

    def forward(self, tokens, s, idx):
        # weighted sum where idx
        s = s[idx]
        tokens = tokens[idx]
        z = tokens * s.unsqueeze(-1)

        z_s = z.sum(0) / (s.sum(0) +1e-6)

        return z_s.unsqueeze(0)

class IELTEncoder(nn.Module):
    def __init__(self, config, update_warm=500, vote_perhead=24, dataset='cub',
                 cam=True, dsm=True, fix=True, total_num=126, assess=False, forward_features=False, merge_inattentive=False):
        super(IELTEncoder, self).__init__()
        self.assess = assess
        self.warm_steps = update_warm
        self.layer = nn.ModuleList()
        self.layer_num = config.num_layers
        self.vote_perhead = vote_perhead
        self.dataset = dataset
        self.cam = cam
        self.dsm = dsm
        self.forward_features = forward_features
        self.merge_inattentive = merge_inattentive

        #for _ in range(self.layer_num - 1):
        #    self.layer.append(Block(config, assess=self.assess))

        self.backbone = DINOv2(patch_size=config.patches[0],
                                embed_dim=config.hidden_size,
                                depth=config.num_layers,
                                num_heads=config.num_heads,
                                mlp_ratio=config.mlp_dim / config.hidden_size,
                                block_fn=partial(Block, assess=self.assess),
                                num_register_tokens=0,
                                init_values=1e-6 if config.ls else 0,)
        
        # copy dinov2.blocks into self.layer
        #print(self.backbone.blocks)
        print(self.backbone.chunked_blocks)
        if self.backbone.chunked_blocks:
            for c in self.backbone.blocks:
                for i in range(self.layer_num-1):
                    self.layer.append(c[i])
        else:
            for i in range(self.layer_num-1):
                self.layer.append(self.backbone.blocks[i])
            
        
        #self.backbone.load_from(torch.load(config.pretrained))
        print(f"Now containing {len(self.layer)} layers")


        if self.dataset == 'dog' or self.dataset == 'nabrids':
            self.layer.append(Block(config, assess=self.assess))
            self.clr_layer = self.layer[-1]
            if self.cam:
                self.layer.append(Block(config, assess=self.assess))
                self.key_layer = self.layer[-1]
        else:
            self.clr_layer = Block(num_heads=config.num_heads, dim=config.hidden_size,)
            if self.cam:
                self.key_layer = Block(num_heads=config.num_heads, dim=config.hidden_size,)

        if self.cam:
            self.key_norm = LayerNorm(config.hidden_size, eps=1e-6)

        self.patch_select = MultiHeadVoting(config, self.vote_perhead, fix)

        self.total_num = total_num
        ## for CUB and NABirds
        # self.select_rate = torch.tensor([16, 14, 12, 10, 8, 6, 8, 10, 12, 14, 16], device='cuda') / self.total_num
        ## for Others
        self.select_rate = torch.ones(self.layer_num-1,device='cuda')/(self.layer_num-1)

        self.select_num = self.select_rate * self.total_num
        self.clr_encoder = CrossLayerRefinement(config, self.clr_layer)
        if self.merge_inattentive:
            self.merger = TokenMerger(config)
        self.count = 0

    def forward(self, hidden_states, test_mode=False, return_cls=False):
        if not test_mode:
            self.count += 1
        B, N, C = hidden_states.shape
        features = []
        complements = [[] for i in range(B)]
        class_token_list = []
        if self.assess:
            layer_weights = []
            layer_selected = []
            layer_score = []
        else:
            pass

        for t in range(self.layer_num - 1):
            layer = self.layer[t]
            select_num = torch.round(self.select_num[t]).int()
            if self.forward_features:
                features.append(hidden_states[:, 1:])
            hidden_states, weights = layer(hidden_states)

            select_idx, select_score, unselect_idx = self.patch_select(weights, select_num)
            
            for i in range(B):
                if self.merge_inattentive:
                    inattentive_tokens = self.merger(hidden_states[i, 1:], select_score[i, :], unselect_idx[i, :])
                forward_tokens = hidden_states[i, select_idx[i, :]]
                # concat with inattentive tokens
                if self.merge_inattentive:
                    out_tokens = torch.cat((forward_tokens, inattentive_tokens), dim=0)
                else:
                    out_tokens = forward_tokens
                # print argmax 
                complements[i].extend(out_tokens)
            class_token_list.append(hidden_states[:, 0].unsqueeze(1))
            if self.assess:
                layer_weights.append(weights)
                layer_score.append(select_score)
                layer_selected.extend(select_idx)
        cls_token = hidden_states[:, 0].unsqueeze(1)

        clr, weights = self.clr_encoder(complements, cls_token)
        sort_idx, _, _ = self.patch_select(weights, select_num=24, last=True)

        if not test_mode and self.count >= self.warm_steps and self.dsm:
            # if not test_mode and self.count >= 500 and self.dsm:
            layer_count = self.count_patch(sort_idx)
            self.update_layer_select(layer_count)

        class_token_list = torch.cat(class_token_list, dim=1)

        if not self.cam:
            if return_cls:
                return clr[:, 0], None, cls_token
            return clr[:, 0], None
        else:
            out = []
            for i in range(B):
                out.append(clr[i, sort_idx[i, :]])
            out = torch.stack(out).squeeze(1)
            out = torch.cat((cls_token, out), dim=1)
            out, _ = self.key_layer(out)
            key = self.key_norm(out)

        if self.assess:
            assess_list = [layer_weights, layer_selected, layer_score, sort_idx]
            if return_cls:
                if self.forward_features:
                    return key[:, 0], clr[:, 0], assess_list, cls_token, features
                return key[:, 0], clr[:, 0], assess_list, cls_token
            if self.forward_features:
                return key[:, 0], clr[:, 0], assess_list, features
            return key[:, 0], clr[:, 0], assess_list
        else:

            # fused = torch.cat((class_token_list, clr[:, 0].unsqueeze(1)), dim=1)
            # clr[:, 0] = fused.mean(1)
            if return_cls:
                if self.forward_features:
                    return key[:, 0], clr[:, 0], cls_token, features
                return key[:, 0], clr[:, 0], cls_token
            if self.forward_features:
                return key[:, 0], clr[:, 0], features
            return key[:, 0], clr[:, 0]

    def update_layer_select(self, layer_count):
        alpha = 1e-3  # if self.dataset != 'dog' and self.dataset == 'nabirds' else 1e-4
        new_rate = layer_count / layer_count.sum()

        self.select_rate = self.select_rate * (1 - alpha) + alpha * new_rate
        self.select_rate /= self.select_rate.sum()
        self.select_num = self.select_rate * self.total_num

    def count_patch(self, sort_idx):
        layer_count = torch.cumsum(self.select_num, dim=-1)
        sort_idx = (sort_idx - 1).reshape(-1)
        for i in range(self.layer_num - 1):
            mask = (sort_idx < layer_count[i])
            layer_count[i] = mask.sum()
        cum_count = torch.cat((torch.tensor([0], device='cuda'), layer_count[:-1]))
        layer_count -= cum_count
        return layer_count.int()