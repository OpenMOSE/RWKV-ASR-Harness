########################################################################################################
# The RWKV Language Model - https://github.com/BlinkDL/RWKV-LM
########################################################################################################
from torch.utils.checkpoint import checkpoint as torch_checkpoint
from torch.profiler import profile, record_function, ProfilerActivity
#from adam_mini import Adam_mini

import os, math, gc, importlib, re
import torch

import torch.nn as nn
from torch.nn import functional as F
import lightning as pl
from lightning.pytorch.strategies import DeepSpeedStrategy
if importlib.util.find_spec('deepspeed'):
    import deepspeed
    from deepspeed.ops.adam import DeepSpeedCPUAdam, FusedAdam
    

from .llm.rwkv7.model import RWKV7
from .registry import Projector_Registry, Encoder_Registry


class ModRWKV(pl.LightningModule):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.image_token_id = 65532
        self.gnorm_total = self.gnorm_proj = self.gnorm_llm = self.gnorm_encoder = 0.0
        # grad-norm inspection (per-module / per-layer / per-param), strategy-agnostic via autograd hooks
        self._inspect = int(getattr(args, "inspect_grad", 0) or 0)   # print interval in steps (0=off)
        self._inspect_layer = int(getattr(args, "inspect_layer", 0) or 0)  # block idx for per-param breakdown
        self._grad_sq = {}        # bucket -> running sum of squared grads (current step)
        self.grad_buckets = {}    # bucket -> grad norm (filled each step)
        self._inspect_hooked = False
        # grad-norm spike diagnosis: capture per-sample batch stats
        self._spike_thresh = float(getattr(args, "spike_thresh", 0) or 0)
        self._batch_diag = None
        encoder_config = {
            'encoder_path': args.encoder_path,
            'project_dim' : args.n_embd
        }
        self.encoder = Encoder_Registry[args.encoder_type](**encoder_config)
        proj_config = {
            'encoder_dim': getattr(self.encoder, 'encoder_dim', 768),
            'project_dim': args.n_embd,
        }
        self.proj = Projector_Registry[args.encoder_type] (**proj_config)

        self.llm = RWKV7(args)

        # optional LoRA on time-mix / ffn (independent ranks; 0 = off)
        lt = int(getattr(args, "lora_tmix", 0) or 0)
        lf = int(getattr(args, "lora_ffn", 0) or 0)
        if lt > 0 or lf > 0:
            from .llm.rwkv7.lora import inject_lora
            n = inject_lora(self.llm, tmix_rank=lt, ffn_rank=lf,
                            alpha=float(getattr(args, "lora_alpha", 0) or 0),
                            dropout=float(getattr(args, "lora_dropout", 0) or 0))
            print(f"[LoRA] tmix_rank={lt} ffn_rank={lf} -> wrapped {n} linears")

    def get_input_embeddings(self):
        return self.llm.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.llm.set_input_embeddings(value)

    def get_placeholder_mask(
        self, input_ids: torch.LongTensor, inputs_embeds: torch.FloatTensor, image_features: torch.FloatTensor
    ):
        """
        Obtains multimodal placeholder mask from `input_ids` or `inputs_embeds`, and checks that the placeholder token count is
        equal to the length of multimodal features. If the lengths are different, an error is raised.
        """
        if input_ids is None:
            special_image_mask = inputs_embeds == self.get_input_embeddings()(
                torch.tensor(self.image_token_id, dtype=torch.long, device=inputs_embeds.device)
            )
            special_image_mask = special_image_mask.all(-1)
        else:
            special_image_mask = input_ids == self.image_token_id

        n_image_tokens = special_image_mask.sum()
        special_image_mask = special_image_mask.unsqueeze(-1).expand_as(inputs_embeds).to(inputs_embeds.device)
        if inputs_embeds[special_image_mask].numel() != image_features.numel():
            n_image_features = image_features.shape[0] * image_features.shape[1]
            raise ValueError(
                f"Image features and image tokens do not match: tokens: {n_image_tokens}, images_tokens: {image_features.shape[0]}, features {n_image_features}"
            )
        return special_image_mask
    def _set_trainable(self):
        # 1) freeze everything
        for p in self.parameters():
            p.requires_grad = False

        # 2) selectively unfreeze by `--train_step` tokens.
        #    Coarse: encoder | proj | rwkv (whole LLM)
        #    Fine-grained RWKV: emb | head | timemix(att) | ffn | ln
        part = list(self.args.train_step or [])
        alias = {"timemix": "att", "tmix": "att", "time_mix": "att", "cmix": "ffn"}
        part = [alias.get(x, x) for x in part]

        selectors = {
            "encoder": lambda n: n.startswith("encoder."),
            "proj":    lambda n: n.startswith("proj."),
            "rwkv":    lambda n: n.startswith("llm."),                          # whole LLM
            "emb":     lambda n: n.startswith("llm.emb"),                       # token embedding
            "head":    lambda n: n.startswith("llm.head") or n.startswith("llm.ln_out"),  # output head + final norm
            "att":     lambda n: n.startswith("llm.blocks") and ".att." in n,   # time-mixing (incl. att.ln_x)
            "att_noln": lambda n: n.startswith("llm.blocks") and ".att." in n and ".ln_x." not in n,  # time-mixing WITHOUT the ln_x GroupNorm (freezes the layer-0 spike source)
            "ffn":     lambda n: n.startswith("llm.blocks") and ".ffn." in n,   # channel-mixing
            "ln":      lambda n: n.startswith("llm.blocks") and (".ln0." in n or ".ln1." in n or ".ln2." in n),  # block layernorms
            "lora":     lambda n: "lora_" in n,                                  # all injected LoRA params
            "lora_att": lambda n: "lora_" in n and ".att." in n,                # LoRA on time-mix only
            "lora_ffn": lambda n: "lora_" in n and ".ffn." in n,                # LoRA on ffn only
        }
        chosen = [k for k in selectors if k in part]
        unknown = [x for x in part if x not in selectors]
        if unknown:
            print(f"[_set_trainable] WARNING: unrecognized train_step tokens ignored: {unknown}")

        preds = [selectors[k] for k in chosen]
        for n, p in self.named_parameters():
            if any(pred(n) for pred in preds):
                p.requires_grad = True

        n_train = sum(p.numel() for p in self.parameters() if p.requires_grad)
        n_total = sum(p.numel() for p in self.parameters())
        print(f"[_set_trainable] groups={chosen} | trainable={n_train/1e6:.1f}M / {n_total/1e6:.1f}M "
              f"({100*n_train/max(n_total,1):.2f}%)")
    
    def get_images_embeds(self, sign):
        return self.proj(self.encoder(sign))
    
    def forward(self, input_ids=None, inputs_embeds=None, signs= None, state = None):

        if inputs_embeds is None:
            inputs_embeds = self.get_input_embeddings()(input_ids)

        if signs is not None and len(signs)>0:
            images_embeds = []
            for sign in signs:
                images_embed = self.proj(self.encoder(sign))
                # flatten per-sign so variable-length modalities (e.g. audio)
                # concatenate correctly; fixed-length (vision) is unaffected.
                images_embeds.append(images_embed.reshape(-1, inputs_embeds.shape[-1]))
            images_embeds = torch.cat(images_embeds, dim=0).to(inputs_embeds.dtype)
            # images_embeds = torch.cat([self.encoder(sign) for sign in signs], dim=0)
            # images_embeds = images_embeds.view(-1, images_embeds.shape[-1])
            # images_embeds = self.proj(images_embeds)  # images_embeds need [B*num_imgs,llm_dim]
            image_mask = self.get_placeholder_mask(
                input_ids, inputs_embeds=inputs_embeds, image_features=images_embeds
            )
            
            inputs_embeds = inputs_embeds.masked_scatter(image_mask, images_embeds)
        logits = self.llm(inputs_embeds=inputs_embeds)

        return logits

    def training_step(self, batch, batch_idx):
        args = self.args

        
        metas = None
        if len(batch) == 4:               # --debug_data attaches per-sample meta ids
            signs, text_tokens, text_labels, metas = batch
        else:
            signs, text_tokens, text_labels = batch
        if metas is not None and self.trainer.is_global_zero:
            step = getattr(self.trainer, "global_step", -1)
            shown = list(metas)[:8]
            more = " ..." if len(metas) > 8 else ""
            print(f"[data] step={step} bidx={batch_idx} n={len(metas)} :: "
                  + " | ".join(str(m) for m in shown) + more, flush=True)
        signs, idx, targets = [sub for sub in signs if sub is not None] , text_tokens.cuda(), text_labels.cuda()
        logits = self(input_ids=idx, signs=signs)
        B, T, V = logits.shape
        # per-token CE (0 on ignored positions) -> exact mean loss + per-sample stats
        ce = F.cross_entropy(logits.reshape(-1, V), targets.reshape(-1),
                             ignore_index=-100, reduction="none").view(B, T)
        valid = (targets != -100)
        n_valid = valid.sum()
        loss = ce.sum() / n_valid.clamp(min=1)

        # Token accuracy on supervised positions: direct signal that the
        # projector/encoder alignment is working (should climb during Stage 1).
        with torch.no_grad():
            if n_valid > 0:
                pred = logits.argmax(dim=-1)
                acc = (pred[valid] == targets[valid]).float().mean()
            else:
                acc = torch.zeros((), device=loss.device)

            # spike diagnosis: stash per-sample stats for this batch so the
            # callback can dump the offending batch when grad_norm spikes.
            if self._spike_thresh > 0:
                nsup = valid.sum(1)                               # supervised tokens / sample
                sloss = ce.sum(1) / nsup.clamp(min=1)             # mean CE / sample
                tlen = (idx == self.image_token_id).sum(1)        # audio tokens / sample
                worst = int(torch.argmax(sloss).item())
                wids = targets[worst][valid[worst]].detach().to("cpu").tolist()
                self._batch_diag = {
                    "n_supervised": int(n_valid.item()),
                    "min_nsup": int(nsup.min().item()),
                    "max_tlen": int(tlen.max().item()),
                    "nsup": nsup.detach().to("cpu").tolist(),
                    "sloss": [round(x, 3) for x in sloss.detach().float().to("cpu").tolist()],
                    "tlen": tlen.detach().to("cpu").tolist(),
                    "worst": worst,
                    "worst_ids": wids,
                }

        return {"loss": loss, "acc": acc.detach()}
        



    # ---- gradient-norm inspection (works under DDP and DeepSpeed ZeRO) ----
    def _bucket_keys(self, name):
        """Map a parameter name to the grad-norm buckets it contributes to."""
        keys = []
        if name.startswith("proj."):
            keys.append("proj")
        elif name.startswith("encoder."):
            keys.append("encoder")
        elif name.startswith("llm."):
            if name.startswith("llm.emb"):
                keys.append("emb")
            elif name.startswith("llm.head") or name.startswith("llm.ln_out"):
                keys.append("head")
            elif ".att." in name:
                keys.append("att")
            elif ".ffn." in name:
                keys.append("ffn")
            elif (".ln0." in name or ".ln1." in name or ".ln2." in name):
                keys.append("ln")
            else:
                keys.append("llm_other")
            m = re.match(r"llm\.blocks\.(\d+)\.", name)
            if m:
                i = int(m.group(1))
                comp = "att" if ".att." in name else ("ffn" if ".ffn." in name else None)
                if comp:
                    keys.append(f"L{i:02d}.{comp}")                       # per-layer
                    if i == self._inspect_layer:
                        leaf = name.split(f"blocks.{i}.", 1)[1]           # e.g. "att.receptance.weight"
                        keys.append(f"L{i:02d}.{leaf}")                   # per-parameter (target layer)
        return keys

    def setup_grad_inspection(self):
        """Register per-parameter autograd hooks that accumulate per-bucket grad sq-norms.
        Captures the locally-computed gradient at backward time, so it works even under
        DeepSpeed ZeRO where .grad is later partitioned/reduce-scattered."""
        if self._inspect_hooked or not self._inspect:
            return
        for n, p in self.named_parameters():
            if not p.requires_grad:
                continue
            keys = tuple(self._bucket_keys(n))
            if not keys:
                continue
            def hook(g, keys=keys):
                v = g.detach().float().pow(2).sum()
                for k in keys:
                    self._grad_sq[k] = self._grad_sq.get(k, 0.0) + v
                return None
            p.register_hook(hook)
        self._inspect_hooked = True
        print(f"[inspect] registered grad hooks on trainable params (per-module + per-layer)")

    def on_fit_start(self):
        self.setup_grad_inspection()

    def on_before_optimizer_step(self, optimizer):
        # Pre-clip gradient norms. Two modes:
        #  - inspect: use the autograd-hook accumulators (per-module + per-layer),
        #    all-reduced across ranks. Works under DeepSpeed ZeRO.
        #  - default: per-module norms from .grad (valid only for DDP/single-GPU).
        if self._inspect and self._grad_sq:
            # Use LOCAL (this-rank) accumulated norms only — NO cross-rank collective.
            # An all_reduce over per-rank-derived keys can deadlock if ranks ever
            # accumulate different bucket-key sets. Rank-0 local values are enough
            # to identify which module/param spikes (relative comparison).
            self.grad_buckets = {k: (v.sqrt().item() if torch.is_tensor(v) else float(v) ** 0.5)
                                 for k, v in self._grad_sq.items()}
            self._grad_sq = {}
            # total from disjoint coarse buckets only (avoid double-counting per-layer/param)
            coarse = ("proj", "encoder", "emb", "head", "att", "ffn", "ln", "llm_other")
            self.gnorm_total = sum(v ** 2 for k, v in self.grad_buckets.items() if k in coarse) ** 0.5
            self.gnorm_proj = self.grad_buckets.get("proj")
            self.gnorm_llm = (sum(self.grad_buckets.get(k, 0.0) ** 2 for k in ("emb", "head", "att", "ffn", "ln", "llm_other")) ** 0.5) or None
        else:
            def _gnorm(module):
                gs = [p.grad.detach().norm(2) for p in module.parameters() if p.grad is not None]
                return torch.norm(torch.stack(gs), 2).item() if gs else 0.0
            self.gnorm_proj = _gnorm(self.proj)
            self.gnorm_llm = _gnorm(self.llm)
            self.gnorm_encoder = _gnorm(self.encoder)
            self.gnorm_total = (self.gnorm_proj ** 2 + self.gnorm_llm ** 2 + self.gnorm_encoder ** 2) ** 0.5

    def configure_optimizers(self):
        args = self.args

        lr_decay = set()
        lr_1x = set()
        lr_2x = set()
        lr_3x = set()
        lr_encoder = set()  # ✅ encoder参数组

        # --------- 1. 分类参数 ---------
        for n, p in self.named_parameters():
            if not p.requires_grad:
                continue

            if n.startswith("encoder."):
                lr_encoder.add(n)
                continue

            if (("_w1" in n) or ("_w2" in n)) and (args.layerwise_lr > 0):
                lr_1x.add(n)
            elif (("time_mix" in n) or ("time_maa" in n)) and (args.layerwise_lr > 0):
                if args.my_pile_stage == 2:
                    lr_2x.add(n)
                else:
                    lr_1x.add(n)
            elif (("time_decay" in n) or ("time_daaaa" in n)) and (args.layerwise_lr > 0):
                if args.my_pile_stage == 2:
                    lr_3x.add(n)
                else:
                    lr_2x.add(n)
            elif ("time_faaaa" in n) and (args.layerwise_lr > 0):
                if args.my_pile_stage == 2:
                    lr_2x.add(n)
                else:
                    lr_1x.add(n)
            elif ("time_first" in n) and (args.layerwise_lr > 0):
                lr_3x.add(n)
            elif (len(p.squeeze().shape) >= 2) and (args.weight_decay > 0):
                lr_decay.add(n)
            else:
                lr_1x.add(n)

        # --------- 2. 转换为列表并建立参数字典 ---------
        lr_decay = sorted(list(lr_decay))
        lr_1x = sorted(list(lr_1x))
        lr_2x = sorted(list(lr_2x))
        lr_3x = sorted(list(lr_3x))
        lr_encoder = sorted(list(lr_encoder))
        param_dict = {n: p for n, p in self.named_parameters()}

        # --------- 3. 构建优化器参数组 ---------
        optim_groups = []

        # ✅ (1) encoder: 当前lr的0.1倍
        if len(lr_encoder) > 0:
            optim_groups.append({
                "params": [param_dict[n] for n in lr_encoder],
                "weight_decay": 0.0,
                "my_lr_scale": 1.0,  # 当前lr的0.1倍
            })

        # ✅ (2) 主模型参数组
        if args.layerwise_lr > 0:
            if args.my_pile_stage == 2:
                optim_groups += [
                    {"params": [param_dict[n] for n in lr_1x], "weight_decay": 0.0, "my_lr_scale": 1.0},
                    {"params": [param_dict[n] for n in lr_2x], "weight_decay": 0.0, "my_lr_scale": 5.0},
                    {"params": [param_dict[n] for n in lr_3x], "weight_decay": 0.0, "my_lr_scale": 5.0},
                ]
            else:
                optim_groups += [
                    {"params": [param_dict[n] for n in lr_1x], "weight_decay": 0.0, "my_lr_scale": 1.0},
                    {"params": [param_dict[n] for n in lr_2x], "weight_decay": 0.0, "my_lr_scale": 2.0},
                    {"params": [param_dict[n] for n in lr_3x], "weight_decay": 0.0, "my_lr_scale": 3.0},
                ]
        else:
            optim_groups.append({
                "params": [param_dict[n] for n in lr_1x],
                "weight_decay": 0.0,
                "my_lr_scale": 1.0,
            })

        # ✅ (3) 带weight decay的参数组
        if args.weight_decay > 0 and len(lr_decay) > 0:
            optim_groups.append({
                "params": [param_dict[n] for n in lr_decay],
                "weight_decay": args.weight_decay,
                "my_lr_scale": 1.0
            })
            adamw_mode = True
        else:
            adamw_mode = False

        # CPU-offloaded ZeRO genuinely needs DeepSpeedCPUAdam.
        if isinstance(self.trainer.strategy, DeepSpeedStrategy) and self.deepspeed_offload:
            return DeepSpeedCPUAdam(optim_groups, lr=self.args.lr_init, betas=self.args.betas, eps=self.args.adam_eps, bias_correction=True, adamw_mode=adamw_mode, amsgrad=False)

        # Default for single-GPU, DDP, and DeepSpeed ZeRO 1/2/3 (no offload):
        # plain torch AdamW. DeepSpeed ZeRO wraps/partitions a client torch
        # optimizer fine, and this avoids DeepSpeed's FusedAdam, whose fused HIP
        # op build is unreliable on ROCm.
        return torch.optim.AdamW(
            optim_groups, lr=self.args.lr_init, betas=self.args.betas,
            eps=self.args.adam_eps,
            weight_decay=(self.args.weight_decay if adamw_mode else 0.0),
        )

    @property
    def deepspeed_offload(self) -> bool:
        strategy = self.trainer.strategy
        if isinstance(strategy, DeepSpeedStrategy):
            cfg = strategy.config["zero_optimization"]
            return cfg.get("offload_optimizer") or cfg.get("offload_param")
        return False