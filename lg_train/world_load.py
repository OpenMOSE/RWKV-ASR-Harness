from lg_train.model import ModRWKV
import torch
from collections import OrderedDict

def WorldLoading(args):

    model = ModRWKV(args)
    model._set_trainable()
    #model = RWKV(args)
    # print(model)
    print(f"########## Loading {args.load_model}... ##########")
    state_dict = torch.load(args.load_model, map_location="cpu", weights_only=True)

    # Two kinds of checkpoint:
    #  (a) full ModRWKV ckpt (a previous stage's output): keys already carry
    #      llm./proj./encoder. prefixes -> load AS-IS so the trained projector
    #      (and encoder, if any) are restored. This is required to resume Stage-2
    #      from a Stage-1 checkpoint; dropping proj.* here was resetting the
    #      projector to random init (acc fell back from ~0.6 to ~0.3).
    #  (b) bare RWKV LLM ckpt (BlinkDL format): no prefixes -> wrap under `llm.`
    #      (there is no proj/encoder to load).
    is_full = any(k.startswith(('llm.', 'proj.', 'encoder.')) for k in state_dict)
    if is_full:
        new_state_dict = dict(state_dict)
    else:
        new_state_dict = {f"llm.{k}": v for k, v in state_dict.items()}

    missing, unexpected = model.load_state_dict(new_state_dict, strict=False)
    print(f"[WorldLoading] full_ckpt={is_full} "
          f"| proj_in_ckpt={any(k.startswith('proj.') for k in new_state_dict)} "
          f"| encoder_in_ckpt={any(k.startswith('encoder.') for k in new_state_dict)} "
          f"| missing={len(missing)} unexpected={len(unexpected)}")

    return model