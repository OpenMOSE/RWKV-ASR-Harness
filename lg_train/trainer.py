import os, math, time, datetime, subprocess
import torch
from torch.utils.data import DataLoader
from lightning_utilities.core.rank_zero import rank_zero_info, rank_zero_only
import lightning as pl
import re
import numpy as np
import json
from lg_train.lrs import wsd,cos_decay

def my_save(args, trainer, dd, ff):
    if '14b-run1' in ff:
        fn = ff.split('/')[-1]
        fff = '/dev/shm/' + fn
        torch.save(dd, fff)
        subprocess.Popen(f" aws s3 mv {fff} s3://rwkv-14b-4k/{fn} --quiet", shell=True)
    elif ('world/14b' in ff) or ('world/7b' in ff):
        aa = ff.split('/')[1]
        fn = ff.split('/')[-1]
        fff = f'/dev/shm/{aa}-{fn}'
        torch.save(dd, fff)
        subprocess.Popen(f" aws s3 mv {fff} s3://rwkv-world/{aa}-{fn} --quiet", shell=True)
    else:
        torch.save(dd, ff)


def prune_old_checkpoints(proj_dir, keep_last, prefix="rwkv-step-"):
    """Keep only the `keep_last` most recent step checkpoints in `proj_dir`.

    Matches files named `{prefix}{N}.pth` and deletes the oldest (smallest N)
    beyond the most recent `keep_last`. Epoch checkpoints (rwkv-{epoch}.pth) and
    rwkv-final.pth are NOT matched by this prefix, so they are never pruned.
    keep_last <= 0 disables pruning (old unbounded behavior).
    """
    if not keep_last or keep_last <= 0:
        return
    pat = re.compile(rf"^{re.escape(prefix)}(\d+)\.pth$")
    ckpts = []
    try:
        for fn in os.listdir(proj_dir):
            m = pat.match(fn)
            if m:
                ckpts.append((int(m.group(1)), os.path.join(proj_dir, fn)))
    except OSError:
        return
    if len(ckpts) <= keep_last:
        return
    ckpts.sort(key=lambda x: x[0])           # oldest step first
    for _step, path in ckpts[:-keep_last]:    # everything except the last `keep_last`
        try:
            os.remove(path)
            print(f"[checkpoint] pruned old {os.path.basename(path)}")
        except OSError as e:
            print(f"[checkpoint] could not prune {os.path.basename(path)}: {e}")


class train_callback(pl.Callback):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.loss_file = os.path.join(args.proj_dir, "loss_data.jsonl")
        if os.path.exists(self.loss_file):
            os.remove(self.loss_file)
            
    def _grad_wandb(self, trainer):
        """Assemble grad-norm fields for wandb: total (+proj/llm if available) and
        per-module / per-layer inspection buckets."""
        extra = {}
        if getattr(trainer, "my_gnorm", None) is not None:
            extra["grad_norm"] = trainer.my_gnorm
            if getattr(trainer, "my_gnorm_proj", None) is not None:
                extra["grad_norm/proj"] = trainer.my_gnorm_proj
            if getattr(trainer, "my_gnorm_llm", None) is not None:
                extra["grad_norm/llm"] = trainer.my_gnorm_llm
        for k, v in getattr(trainer, "my_grad_buckets", {}).items():
            extra[f"gnorm/{k}"] = v
        # batch covariates for spike correlation
        if getattr(trainer, "my_nsup", None) is not None:
            extra["batch/n_supervised"] = trainer.my_nsup
            extra["batch/min_nsup"] = trainer.my_min_nsup
            extra["batch/max_tlen"] = trainer.my_max_tlen
        return extra

    def write_data(self, loss_data, t_cost, kt_s, lr, acc=None, gnorm=None):
        # 将loss数据写入文件，便于streamlit绘图
        rec = {"loss": float(loss_data), "t_cost": t_cost, "kt_s": kt_s, "lr": lr}
        if acc is not None:
            rec["acc"] = float(acc)
        if gnorm is not None:
            rec["grad_norm"] = float(gnorm)
        with open(self.loss_file, 'a') as f:
            json.dump(rec, f)
            f.write('\n')

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        args = self.args
        # if args.cuda_cleanup > 0:
        #     torch.cuda.empty_cache()
        real_step = trainer.global_step + args.epoch_begin * args.epoch_steps
        # LR schedule
        w_step = args.warmup_steps
        if args.lr_final == args.lr_init or args.epoch_count == 0:
            lr = args.lr_init
        else:
            if 'wsd' == args.lr_schedule:
                lr = wsd(args.lr_init, 0, real_step, trainer.num_training_batches)
            else:
                lr = cos_decay(args.lr_init, args.lr_final, real_step, trainer.num_training_batches)
        if trainer.global_step < w_step:
            lr = lr * (0.01 + 0.99 * trainer.global_step / w_step)

        if args.weight_decay_final > 0:
            wd_now = args.weight_decay * math.exp(math.log(args.weight_decay_final / args.weight_decay) * progress)
        else:
            wd_now = args.weight_decay

        for param_group in trainer.optimizers[0].param_groups:
            if param_group["weight_decay"] > 0:
                param_group["weight_decay"] = wd_now
            if args.layerwise_lr > 0:
                param_group["lr"] = lr * param_group["my_lr_scale"]
                # print(param_group["lr"], param_group["my_lr_scale"])
            else:
                param_group["lr"] = lr

        # optimizer = trainer.optimizers[0]
        # print("=== Optimizer param group LRs ===")
        # for i, group in enumerate(optimizer.param_groups):
        #     wd = group.get("weight_decay", 0)
        #     scale = group.get("my_lr_scale", 1.0)
        #     print(f"[Group {i}] lr = {group['lr']:.2e}, my_lr_scale = {scale}, weight_decay = {wd}")



        trainer.my_lr = lr
        trainer.my_wd = wd_now
        # rank_zero_info(f"{real_step} {lr}")

        if trainer.global_step == 0:
            if trainer.is_global_zero:  # logging
                trainer.my_loss_sum = 0
                trainer.my_loss_count = 0
                trainer.my_log = open(args.proj_dir + "/train_log.txt", "a")
                trainer.my_log.write(f"NEW RUN {args.my_timestamp}\n{vars(self.args)}\n")
                try:
                    print(f"\n{trainer.strategy.config}\n")
                    trainer.my_log.write(f"{trainer.strategy.config}\n")
                except:
                    pass
                trainer.my_log.flush()
                if len(args.wandb) > 0:
                    print("Login to wandb...")
                    import wandb
                    wandb.init(
                        project=args.wandb,
                        name=args.run_name + " " + args.my_timestamp,
                        config=args,
                        save_code=False,
                        mode=getattr(args, "wandb_mode", "online"),

                    )
                    trainer.my_wandb = wandb

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        args = self.args
        token_per_step = args.ctx_len * args.real_bsz
        real_step = trainer.global_step + args.epoch_begin * args.epoch_steps

        acc = None
        if pl.__version__[0]=='2' :
            loss = outputs['loss']
            if isinstance(outputs, dict) and ('acc' in outputs):
                acc = outputs['acc']
            if int(args.devices)>1:
                torch.distributed.all_reduce(loss, op=torch.distributed.ReduceOp.SUM)

        if trainer.is_global_zero:  # logging
            t_now = time.time_ns()
            kt_s = 0
            t_cost = 0
            try:
                t_cost = (t_now - trainer.my_time_ns) / 1e9
                kt_s = token_per_step / t_cost / 1000
                t_cost = 1.0 / t_cost
                self.log("REAL it/s", t_cost, prog_bar=True, on_step=True)
                self.log("Kt/s", kt_s, prog_bar=True, on_step=True)
            except:
                pass
            trainer.my_time_ns = t_now
            if pl.__version__[0]=='2':
                trainer.my_loss = loss*trainer.accumulate_grad_batches/int(args.devices)
            else:
                trainer.my_loss = trainer.my_loss_all.float().mean().item()
            trainer.my_loss_sum += trainer.my_loss
            trainer.my_loss_count += 1
            trainer.my_epoch_loss = trainer.my_loss_sum / trainer.my_loss_count
            self.log("lr", trainer.my_lr, prog_bar=True, on_step=True)
            self.log("sum_loss", trainer.my_epoch_loss, prog_bar=True, on_step=True)
            self.log("loss", trainer.my_loss, prog_bar=True, on_step=True)
            trainer.my_acc = float(acc) if acc is not None else None
            if trainer.my_acc is not None:
                self.log("acc", trainer.my_acc, prog_bar=True, on_step=True)
            # gradient norm. DDP/single-GPU: per-param norms from on_before_optimizer_step.
            # DeepSpeed ZeRO: grads are partitioned -> those are invalid; read the
            # engine's global grad norm (pre-clip) instead (per-module unavailable).
            gnorm = getattr(pl_module, "gnorm_total", None)
            gnorm_proj = getattr(pl_module, "gnorm_proj", None)
            gnorm_llm = getattr(pl_module, "gnorm_llm", None)
            try:
                strat = trainer.strategy
                engine = getattr(strat, "deepspeed_engine", None)
                if engine is None:
                    cand = getattr(strat, "model", None)
                    engine = cand if hasattr(cand, "get_global_grad_norm") else None
                if engine is not None and hasattr(engine, "get_global_grad_norm"):
                    ggn = engine.get_global_grad_norm()
                    if ggn is not None:
                        gnorm = float(ggn)
                        gnorm_proj = gnorm_llm = None  # not available under ZeRO partitioning
            except Exception:
                pass
            trainer.my_gnorm = gnorm
            trainer.my_gnorm_proj = gnorm_proj
            trainer.my_gnorm_llm = gnorm_llm
            # per-module / per-layer buckets from inspection hooks (strategy-agnostic)
            trainer.my_grad_buckets = getattr(pl_module, "grad_buckets", {}) or {}
            if trainer.my_gnorm is not None:
                self.log("gnorm", trainer.my_gnorm, prog_bar=True, on_step=True)

            # spike diagnosis: batch covariates + dump the offending batch
            diag = getattr(pl_module, "_batch_diag", None)
            trainer.my_nsup = trainer.my_min_nsup = trainer.my_max_tlen = None
            if diag is not None:
                trainer.my_nsup = diag["n_supervised"]
                trainer.my_min_nsup = diag["min_nsup"]
                trainer.my_max_tlen = diag["max_tlen"]
                thr = getattr(pl_module, "_spike_thresh", 0.0)
                if thr and (trainer.my_gnorm is not None) and (trainer.my_gnorm > thr):
                    try:
                        from lg_train.utils import pipeline as _pipe
                        wtext = _pipe.decode(diag["worst_ids"]) if diag.get("worst_ids") else ""
                    except Exception:
                        wtext = "<decode failed>"
                    print("[spike] step %d grad_norm=%.1f | n_sup=%d min_nsup=%d max_tlen=%d" % (
                        trainer.global_step, trainer.my_gnorm, diag["n_supervised"],
                        diag["min_nsup"], diag["max_tlen"]), flush=True)
                    print("[spike]   per-sample loss=%s" % diag["sloss"], flush=True)
                    print("[spike]   per-sample nsup=%s" % diag["nsup"], flush=True)
                    print("[spike]   per-sample tlen=%s" % diag["tlen"], flush=True)
                    print("[spike]   worst #%d transcript=%r" % (diag["worst"], str(wtext)[:160]), flush=True)
            # inspection: surface top spiking layers + per-parameter breakdown of the target layer
            interval = getattr(pl_module, "_inspect", 50) or 50
            if trainer.my_grad_buckets and trainer.global_step % interval == 0:
                b = trainer.my_grad_buckets
                tl = int(getattr(pl_module, "_inspect_layer", 0))
                layer_top = sorted([(k, v) for k, v in b.items() if re.fullmatch(r"L\d+\.(att|ffn)", k)],
                                    key=lambda kv: -kv[1])[:6]
                param_top = sorted([(k, v) for k, v in b.items()
                                    if k.startswith(f"L{tl:02d}.att.") or k.startswith(f"L{tl:02d}.ffn.")],
                                   key=lambda kv: -kv[1])[:12]
                if layer_top:
                    print("[inspect] step %d | total=%.2f | top layers: %s" % (
                        trainer.global_step, (trainer.my_gnorm or 0.0),
                        ", ".join(f"{k}={v:.2f}" for k, v in layer_top)), flush=True)
                if param_top:
                    print("[inspect] step %d | L%02d params: %s" % (
                        trainer.global_step, tl,
                        ", ".join(f"{k.split('.', 1)[1]}={v:.2f}" for k, v in param_top)), flush=True)

            # 将loss、t_cost、kt_s写入data.json
            if trainer.accumulate_grad_batches!=None:
                args.avg_loss += trainer.my_loss / trainer.accumulate_grad_batches
                if (batch_idx+1) % trainer.accumulate_grad_batches == 0:
                    if len(args.wandb) > 0:
                        lll = {"loss": args.avg_loss, "lr": trainer.my_lr, "wd": trainer.my_wd, "Gtokens": real_step * token_per_step / 1e9}
                        if kt_s > 0:
                            lll["kt/s"] = kt_s
                        if trainer.my_acc is not None:
                            lll["acc"] = trainer.my_acc
                        lll.update(self._grad_wandb(trainer))
                        trainer.my_wandb.log(lll, step=int(real_step))
                    self.write_data(args.avg_loss, t_cost, kt_s, trainer.my_lr, trainer.my_acc, trainer.my_gnorm)
                    args.avg_loss = 0
            else:
                if len(args.wandb) > 0:
                    lll = {"loss": trainer.my_loss, "lr": trainer.my_lr, "wd": trainer.my_wd, "Gtokens": real_step * token_per_step / 1e9}
                    if kt_s > 0:
                        lll["kt/s"] = kt_s
                    if trainer.my_acc is not None:
                        lll["acc"] = trainer.my_acc
                    lll.update(self._grad_wandb(trainer))
                    trainer.my_wandb.log(lll, step=int(real_step))
                self.write_data(trainer.my_loss, t_cost, kt_s, trainer.my_lr, trainer.my_acc, trainer.my_gnorm)

            if args.save_per_steps > 0 and trainer.global_step > 0 and trainer.global_step % args.save_per_steps == 0:
                to_save_dict = pl_module.state_dict()
                try:
                    my_save(
                        args, trainer,
                        to_save_dict,
                        f"{args.proj_dir}/rwkv-step-{trainer.global_step}.pth",
                    )
                    print(f"\n[checkpoint] saved rwkv-step-{trainer.global_step}.pth\n")
                    prune_old_checkpoints(args.proj_dir, getattr(args, "keep_last_ckpt", 5))
                except Exception as e:
                    print('Error\n\n', e, '\n\n')
                
        if (trainer.is_global_zero) or ('deepspeed_stage_3' in args.strategy): # save pth
            if args.magic_prime > 0:
                expand_factor = 2 if args.my_qa_mask > 0 else 1
                if int(real_step) == int(args.magic_prime * expand_factor // args.real_bsz) - 1 + int(args.my_random_steps):
                    to_save_dict = pl_module.state_dict()
                    my_save(
                        args, trainer,
                        to_save_dict,
                        f"{args.proj_dir}/rwkv-final.pth",
                    )
                

    def on_train_epoch_start(self, trainer, pl_module):
        args = self.args
        if pl.__version__[0]=='2':
            dataset = trainer.train_dataloader.dataset
        else:
            dataset = trainer.train_dataloader.dataset.datasets
        # assert "MyDataset" in str(dataset)
        dataset.global_rank = trainer.global_rank
        dataset.real_epoch = int(args.epoch_begin + trainer.current_epoch)
        dataset.world_size = trainer.world_size
        # print(f'########## world_size {dataset.world_size} global_rank {dataset.global_rank} real_epoch {dataset.real_epoch} ##########')

    def on_train_epoch_end(self, trainer, pl_module):
        args = self.args
        to_save_dict = {}

        if (trainer.is_global_zero) or ('deepspeed_stage_3' in args.strategy):  # save pth
            if (args.epoch_save > 0 and trainer.current_epoch % args.epoch_save == 0) or (trainer.current_epoch == args.epoch_count - 1):
                if args.data_type == 'wds_img':
                    raw_dict = pl_module.state_dict()
                    for k in raw_dict:
                        if k.startswith('encoder.') or k.startswith('decoder.'):
                            to_save_dict[k] = raw_dict[k]
                else:
                    to_save_dict = pl_module.state_dict()

                try:


                    my_save(
                        args, trainer,
                        to_save_dict,
                        f"{args.proj_dir}/rwkv-{args.epoch_begin + trainer.current_epoch}.pth",
                    )
                except Exception as e:
                    print('Error\n\n', e, '\n\n')

        if trainer.is_global_zero:  # logging
            trainer.my_log.write(f"{args.epoch_begin + trainer.current_epoch} {trainer.my_epoch_loss:.6f} {math.exp(trainer.my_epoch_loss):.4f} {trainer.my_lr:.8f} {datetime.datetime.now()} {trainer.current_epoch}\n")
            trainer.my_log.flush()

            trainer.my_loss_sum = 0
            trainer.my_loss_count = 0
            if (args.epoch_begin + trainer.current_epoch) >= args.my_exit:
                exit(0)


@rank_zero_only
def generate_init_weight(model, init_weight_name):
    mm = model.generate_init_weight()

    if model.args.my_pile_stage == 1:
        if len(model.args.load_model) > 0:
            print(f"Combine weights from {model.args.load_model}...")
            load_dict = torch.load(model.args.load_model, map_location="cpu")
            for k in load_dict:
                try:
                    assert k in mm
                except:
                    print('missing', k)
                    exit(0)
                src = load_dict[k]
                try:
                    mm[k] = src.reshape(mm[k].shape)
                except:
                    tmp = mm[k].squeeze().clone()
                    print(k, src.shape, '-->', mm[k].shape)
                    ss = src.shape[0]
                    dd = tmp.shape[0]
                    for i in range(dd):
                        pos = i / dd * ss
                        if pos >= ss - 1:
                            tmp[i] = src[ss-1]
                        else:
                            p0 = int(math.floor(pos))
                            ii = pos - p0
                            tmp[i] = src[p0] * (1-ii) + src[p0+1] * (ii)
                    mm[k] = tmp.reshape(mm[k].shape)
                    sss = src.squeeze().float().cpu().numpy()
                    print(sss[:10], '...', sss[-10:])
                    mmm = mm[k].squeeze().float().cpu().numpy()
                    print(mmm[:10], '...', mmm[-10:])

    print(f"Save to {init_weight_name}...")
    torch.save(mm, init_weight_name)

    if model.args.my_pile_stage == 1:
        print("Done. Now go for stage 2.")
        exit(0)

