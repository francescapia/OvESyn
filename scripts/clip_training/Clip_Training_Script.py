import os
import sys
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel
from torch.optim import AdamW

ROOT = Path(__file__).resolve().parent
REPO_ROOT = ROOT.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from scripts.clip_training.utils.custom_schedulers import get_cosine_schedule_with_warmup
from scripts.clip_training.utils.util import mkdir
from scripts.diff_model_setting import distributed_barrier


def is_distributed() -> bool:
    return dist.is_available() and dist.is_initialized()


def get_rank() -> int:
    return dist.get_rank() if is_distributed() else 0


def get_world_size() -> int:
    return dist.get_world_size() if is_distributed() else 1


def is_main_process() -> bool:
    return get_rank() == 0


def unwrap_model(model):
    return model.module if hasattr(model, "module") else model


def get_tokenizer_and_length(model):
    module = unwrap_model(model)
    return module.tokenizer, module.max_length


def get_logit_scale(model):
    return unwrap_model(model).model.logit_scale.exp()


def reduce_mean(value: torch.Tensor) -> torch.Tensor:
    if not is_distributed():
        return value.detach()

    reduced = value.detach().clone()
    dist.all_reduce(reduced, op=dist.ReduceOp.SUM)
    reduced /= get_world_size()
    return reduced


def gather_features(features: torch.Tensor) -> torch.Tensor:
    if not is_distributed():
        return features

    gathered = [torch.zeros_like(features) for _ in range(get_world_size())]
    dist.all_gather(gathered, features.detach())
    gathered[get_rank()] = features
    return torch.cat(gathered, dim=0)


def tokenize_texts(model, texts, device):
    tokenizer, max_length = get_tokenizer_and_length(model)
    return tokenizer(
        texts,
        truncation=True,
        max_length=max_length,
        padding="max_length",
        return_tensors="pt",
    )["input_ids"].to(device, non_blocking=True)


def encode_and_normalize(model, images, token_ids):
    image_features, text_features = model((images, token_ids), "encode_both")

    if image_features.dim() == 3:
        image_features = image_features.mean(dim=1)
    if text_features.dim() == 3:
        text_features = text_features.mean(dim=1)

    image_features = image_features / image_features.norm(dim=-1, keepdim=True)
    text_features = text_features / text_features.norm(dim=-1, keepdim=True)
    return image_features, text_features


def compute_clip_loss(model, image_features, text_features, device, gather_distributed=True):
    if gather_distributed and is_distributed():
        all_image_features = gather_features(image_features)
        all_text_features = gather_features(text_features)
    else:
        all_image_features = image_features
        all_text_features = text_features
    logit_scale = get_logit_scale(model)

    local_batch = image_features.shape[0]
    labels = torch.arange(local_batch, device=device)
    if gather_distributed and is_distributed():
        labels = labels + get_rank() * local_batch

    logits_per_image = logit_scale * (image_features @ all_text_features.T)
    logits_per_text = logit_scale * (text_features @ all_image_features.T)

    image_loss = F.cross_entropy(logits_per_image, labels)
    text_loss = F.cross_entropy(logits_per_text, labels)
    return (image_loss + text_loss) / 2


@torch.no_grad()
def eval_retrieval(config, dataloader, model):
    module = unwrap_model(model)
    module.eval()
    device = torch.device(config.device)
    tokenizer, max_length = get_tokenizer_and_length(module)

    img_embs = []
    txt_embs = []

    for batch in dataloader:
        images = batch["image"].to(device, non_blocking=True)
        texts = batch["impression"]

        toks = tokenizer(
            texts,
            truncation=True,
            max_length=max_length,
            padding="max_length",
            return_tensors="pt",
        )["input_ids"].to(device, non_blocking=True)

        with torch.amp.autocast("cuda", enabled=getattr(config, "use_amp", False)):
            v, t = encode_and_normalize(module, images, toks)

        img_embs.append(v.detach().float().cpu())
        txt_embs.append(t.detach().float().cpu())

    V = torch.cat(img_embs, dim=0)
    T = torch.cat(txt_embs, dim=0)
    N = V.shape[0]

    sim_i2t = V @ T.T
    sim_t2i = T @ V.T

    def compute_metrics(sim):
        ranks = []
        for i in range(N):
            s = sim[i]
            gt = s[i].item()
            rank = 1 + int((s > gt).sum().item())
            ranks.append(rank)

        ranks_t = torch.tensor(ranks)
        return {
            "R@1": (ranks_t <= 1).float().mean().item(),
            "R@5": (ranks_t <= 5).float().mean().item(),
            "R@10": (ranks_t <= 10).float().mean().item(),
            "median_rank": int(torch.median(ranks_t).item()),
        }

    out_i2t = compute_metrics(sim_i2t)
    out_t2i = compute_metrics(sim_t2i)

    module.train()
    return out_i2t, out_t2i


@torch.no_grad()
def eval_epoch_loss(config, dataloader, model):
    module = unwrap_model(model)
    module.eval()
    total = 0.0
    n_steps = 0
    device = torch.device(config.device)

    for batch in dataloader:
        images = batch["image"].to(device, non_blocking=True)
        texts = batch["impression"]

        if images.shape[0] < 2:
            continue

        toks = tokenize_texts(module, texts, device)

        with torch.amp.autocast("cuda", enabled=getattr(config, "use_amp", False)):
            image_features, text_features = encode_and_normalize(module, images, toks)
            loss = compute_clip_loss(module, image_features, text_features, device, gather_distributed=False)

        total += loss.item()
        n_steps += 1

    module.train()
    return total / max(1, n_steps)


def save_state_dict(model, path):
    mkdir(os.path.dirname(path))
    torch.save(unwrap_model(model).state_dict(), path)


def train(
    config,
    dataloader,
    model,
    logger,
    val_retrieval_dataloader=None,
    train_retrieval_dataloader=None,
    val_loss_dataloader=None,
    train_loss_dataloader=None,
):
    device = torch.device(config.device)
    distributed = bool(getattr(config, "distributed", False))
    main_process = is_main_process()

    if distributed:
        ddp_kwargs = {
            "device_ids": [config.local_rank],
            "output_device": config.local_rank,
            "find_unused_parameters": bool(getattr(config, "ddp_find_unused_parameters", False)),
        }
        model = DistributedDataParallel(model, **ddp_kwargs)
        if bool(getattr(config, "ddp_static_graph", False)) and hasattr(model, "_set_static_graph"):
            model._set_static_graph()
    else:
        model = model.to(device)

    config.train_batch_size = config.per_gpu_train_batch_size * max(1, getattr(config, "world_size", 1))
    train_dataloader = dataloader

    t_total = max(1, len(train_dataloader) // config.gradient_accumulation_steps * int(config.num_train_epochs))

    if hasattr(config, "lora") and config.lora.get("enabled", False):
        from lora_layers import get_lora_parameters

        optimizer_params = get_lora_parameters(unwrap_model(model))
        if main_process:
            logger.info(f"LoRA enabled: training {sum(p.numel() for p in optimizer_params):,} parameters")
    else:
        optimizer_params = model.parameters()
        if main_process:
            logger.info("Full fine-tuning: training all parameters")

    optimizer = AdamW(
        optimizer_params,
        lr=config.optimizer.params.lr,
        eps=config.optimizer.params.eps,
        weight_decay=config.optimizer.params.weight_decay,
    )

    scaler = torch.amp.GradScaler("cuda", enabled=getattr(config, "use_amp", False))

    warmup_ratio = float(getattr(config, "warmup_ratio", 0.20))
    num_warmup_steps = int(warmup_ratio * t_total)
    if config.get("scheduler") == "constant_with_warmup":
        from transformers import get_constant_schedule_with_warmup

        scheduler = get_constant_schedule_with_warmup(optimizer, num_warmup_steps)
    else:
        scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, t_total)

    model.train()

    if main_process:
        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", len(train_dataloader.dataset))
        logger.info("  Num Epochs = %d", config.num_train_epochs)
        logger.info("  Number of GPUs = %d", getattr(config, "world_size", 1))
        logger.info("  Batch size per GPU = %d", config.per_gpu_train_batch_size)
        logger.info(
            "  Total train batch size (w. distributed & accumulation) = %d",
            config.train_batch_size * config.gradient_accumulation_steps,
        )
        logger.info("  Gradient Accumulation steps = %d", config.gradient_accumulation_steps)
        logger.info("  Total optimization steps = %d", t_total)
        logger.info("  Warmup steps = %d", num_warmup_steps)
        logger.info("  DDP static graph = %s", bool(getattr(config, "ddp_static_graph", False)))
        logger.info("  DDP find unused parameters = %s", bool(getattr(config, "ddp_find_unused_parameters", False)))

    global_step = 0
    global_loss_sum = 0.0
    global_loss_count = 0

    from datetime import datetime

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    metrics_path = os.path.join(config.logs, f"metrics_{config.name}_{run_id}.csv")
    metrics_epoch_path = os.path.join(config.logs, f"metrics_epoch_{config.name}_{run_id}.csv")

    if main_process:
        mkdir(config.logs)
        with open(metrics_path, "w") as f:
            f.write("epoch,global_step,lr,loss,avg_loss,logit_scale\n")
        if not os.path.exists(metrics_epoch_path):
            with open(metrics_epoch_path, "w") as f:
                f.write(
                    "epoch,train_loss_epoch,train_loss_eval,val_loss_epoch,lr,logit_scale,"
                    "train_i2t_R1,train_i2t_R5,train_i2t_R10,train_i2t_median,"
                    "train_t2i_R1,train_t2i_R5,train_t2i_R10,train_t2i_median,"
                    "val_i2t_R1,val_i2t_R5,val_i2t_R10,val_i2t_median,"
                    "val_t2i_R1,val_t2i_R5,val_t2i_R10,val_t2i_median\n"
                )

    es_cfg = config.get("early_stopping", {})
    es_enabled = bool(es_cfg.get("enabled", False)) and (val_loss_dataloader is not None)
    es_patience = int(es_cfg.get("patience", 100))
    es_min_delta = float(es_cfg.get("min_delta", 0.0))

    best_val_loss = float("inf")
    bad_epochs = 0
    best_epoch = -1

    model.zero_grad(set_to_none=True)
    for epoch in range(int(config.num_train_epochs)):
        sampler = getattr(train_dataloader, "sampler", None)
        if distributed and hasattr(sampler, "set_epoch"):
            sampler.set_epoch(epoch)

        max_train_batches = getattr(config, "max_train_batches_per_epoch", None)
        for step, batch in enumerate(train_dataloader):
            input_images = batch["image"].to(device, non_blocking=True)
            input_texts = batch["impression"]
            input_ids = tokenize_texts(model, input_texts, device)

            with torch.amp.autocast("cuda", enabled=getattr(config, "use_amp", False)):
                image_features, text_features = encode_and_normalize(model, input_images, input_ids)
                loss = compute_clip_loss(model, image_features, text_features, device)

            loss_to_log = float(reduce_mean(loss).item())
            if config.gradient_accumulation_steps > 1:
                loss = loss / config.gradient_accumulation_steps

            scaler.scale(loss).backward()
            global_loss_sum += loss_to_log
            global_loss_count += 1

            if (step + 1) % config.gradient_accumulation_steps == 0:
                global_step += 1
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                unwrap_model(model).model.logit_scale.data.clamp_(0, 4.6052)
                model.zero_grad(set_to_none=True)

                if main_process and global_step % config.logging_steps == 0:
                    lr = optimizer.param_groups[0]["lr"]
                    avg_loss = global_loss_sum / max(1, global_loss_count)
                    logger.info(
                        "Epoch: %s, global_step: %s, lr: %.6f, loss: %.4f (%.4f)",
                        epoch,
                        global_step,
                        lr,
                        loss_to_log,
                        avg_loss,
                    )
                    with open(metrics_path, "a") as f:
                        f.write(
                            f"{epoch},{global_step},{lr},{loss_to_log},{avg_loss},{get_logit_scale(model).item()}\n"
                        )

            if max_train_batches is not None and (step + 1) >= int(max_train_batches):
                break

        eval_every = int(getattr(config, "eval_every_epochs", 1))
        should_eval = (epoch % eval_every == 0) or (epoch == int(config.num_train_epochs) - 1)
        eval_train_retrieval = bool(getattr(config, "eval_train_retrieval", True))
        eval_train_loss = bool(getattr(config, "eval_train_loss", True))
        eval_val_retrieval = bool(getattr(config, "eval_val_retrieval", True))

        stop_training = False
        train_i2t, train_t2i, val_i2t, val_t2i = {}, {}, {}, {}
        train_loss_eval = None
        val_loss = None

        if main_process and should_eval:
            if eval_train_retrieval and train_retrieval_dataloader is not None:
                train_i2t, train_t2i = eval_retrieval(config, train_retrieval_dataloader, model)
                logger.info(f"[Epoch {epoch}] TRAIN Image->Text: {train_i2t} | Text->Image: {train_t2i}")

            if eval_val_retrieval and val_retrieval_dataloader is not None:
                val_i2t, val_t2i = eval_retrieval(config, val_retrieval_dataloader, model)
                logger.info(f"[Epoch {epoch}] VAL   Image->Text: {val_i2t} | Text->Image: {val_t2i}")

            if val_loss_dataloader is not None and es_enabled:
                val_loss = eval_epoch_loss(config, val_loss_dataloader, model)
                logger.info(f"[Epoch {epoch}] VAL loss: {val_loss:.6f}")

            if eval_train_loss and train_loss_dataloader is not None:
                train_loss_eval = eval_epoch_loss(config, train_loss_dataloader, model)
                logger.info(f"[Epoch {epoch}] TRAIN eval loss: {train_loss_eval:.6f}")

            lr_now = optimizer.param_groups[0]["lr"]
            logit_now = get_logit_scale(model).item()
            train_loss_epoch = global_loss_sum / max(1, global_loss_count)

            def g(d, k):
                return d.get(k, None) if isinstance(d, dict) else None

            with open(metrics_epoch_path, "a") as f:
                f.write(
                    f"{epoch},{train_loss_epoch},{train_loss_eval},{val_loss},{lr_now},{logit_now},"
                    f"{g(train_i2t,'R@1')},{g(train_i2t,'R@5')},{g(train_i2t,'R@10')},{g(train_i2t,'median_rank')},"
                    f"{g(train_t2i,'R@1')},{g(train_t2i,'R@5')},{g(train_t2i,'R@10')},{g(train_t2i,'median_rank')},"
                    f"{g(val_i2t,'R@1')},{g(val_i2t,'R@5')},{g(val_i2t,'R@10')},{g(val_i2t,'median_rank')},"
                    f"{g(val_t2i,'R@1')},{g(val_t2i,'R@5')},{g(val_t2i,'R@10')},{g(val_t2i,'median_rank')}\n"
                )

            if es_enabled and val_loss is not None:
                improved = (best_val_loss - val_loss) > es_min_delta
                if improved:
                    best_val_loss = val_loss
                    best_epoch = epoch
                    bad_epochs = 0
                    best_path = os.path.join(config.saved_checkpoints, f"checkpoint_best_{config.name}.pt")
                    save_state_dict(model, best_path)
                    logger.info(f"[Epoch {epoch}] New best val_loss. Saved: {best_path}")
                else:
                    bad_epochs += 1
                    logger.info(f"[Epoch {epoch}] No improvement in val_loss. bad_epochs={bad_epochs}/{es_patience}")
                    if bad_epochs >= es_patience:
                        logger.info(
                            f"EARLY STOPPING triggered at epoch {epoch}. "
                            f"Best epoch was {best_epoch} with val_loss={best_val_loss:.6f}"
                        )
                        stop_training = True

            if config.save_steps_epochs > 0 and epoch % config.save_steps_epochs == 0:
                save_path = os.path.join(config.saved_checkpoints, f"checkpoint_{epoch}_epoch_{config.name}.pt")
                save_state_dict(model, save_path)

        if distributed:
            stop_tensor = torch.tensor(int(stop_training), device=device)
            dist.broadcast(stop_tensor, src=0)
            distributed_barrier(device)
            stop_training = bool(stop_tensor.item())

        if stop_training:
            break

    if main_process and config.save_steps_epochs > 0:
        save_path = os.path.join(config.saved_checkpoints, f"checkpoint_{epoch}_epoch_{config.name}.pt")
        save_state_dict(model, save_path)

    return global_step, global_loss_sum / max(1, global_loss_count)
