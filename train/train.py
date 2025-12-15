# -*- coding: UTF-8 -*-

"""
train.py: Script for training a model.
"""

### IMPORTS ###
from transformers import PreTrainedTokenizer
import time
import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import Dataset
from typing import Generator
from os.path import exists

from train.state import TrainState
from data.batch import Batch
from data.dataloader import create_dataloaders
from data.dataset import load_multi30k
from data.tokenizer import load_tokenizers
from model.transformer import Transformer
from train.loss import SimpleLossCompute
from train.label_smoothing import LabelSmoothing
from train.rate import rate
from utils import DummyOptimizer, DummyScheduler

### CONSTANTS ###
DEFAULT_D_MODEL = 512
DEFAULT_NUM_HEADS = 6
DEFAULT_SMOOTHING=0.1
STEPS_UNTIL_LOG_WITHIN_EPOCH = 40 # number of steps within an epoch until we log metrics
MODE_EVAL = "eval"
MODE_TRAIN = 'train'
MODE_TRAIN_AND_LOG = 'train+log'
LEARNING_RATE_PARAM_GROUP_SPECIFIER = 'lr'

def run_epoch(
    data_iter: Generator[Batch, None, None],
    model: Transformer,
    loss_compute,
    optimizer: Optimizer,
    scheduler: Optimizer,
    mode: str = MODE_TRAIN,
    accum_iter: int = 1,
    train_state: TrainState =TrainState()
):
    """Train the model for a single epoch.
    
    Args:
        data_iter: Data-iterator that holds the training data
        model: The model to be trained
        loss_compute: Loss computation 
        optimizer: The optimizer being used 
        scheduler: The learning rate scheduler being used
        mode: Current training mode; either `train` or `train+log`
        accum_iter: The number of passes to perform after which we update the parameters
        train_state: The current state of the training process
    
    Returns:
        Tuple: The overall loss and the final train state.
    """

    start_time = time.time()
    total_tokens = 0
    total_loss = 0
    tokens = 0
    num_accumulated = 0
    for i, batch in enumerate(data_iter):
        # Step 1: Reset gradients
        optimizer.zero_grad(set_to_none=True)

        # Step 2: Forward-pass the training data through the model
        out = model.forward(
            batch.src, batch.tgt, batch.src_mask, batch.tgt_mask
        )

        # Step 3: Compute the loss
        loss, loss_node = loss_compute(out, batch.tgt_y, batch.ntokens)

        if mode == MODE_TRAIN or mode == MODE_TRAIN_AND_LOG:
            # Step 4: Backpropagate the gradients
            loss_node.backward()
            train_state.step += 1
            train_state.samples += batch.src.shape[0]
            train_state.tokens += batch.ntokens

            if i % accum_iter == 0:
                # Update the parameters
                optimizer.step()
                num_accumulated += 1
                train_state.accum_step += 1
            scheduler.step()

        total_loss += loss
        total_tokens += batch.ntokens
        tokens += batch.ntokens

        if i % STEPS_UNTIL_LOG_WITHIN_EPOCH == 1 and (mode == MODE_TRAIN or mode == MODE_TRAIN_AND_LOG):
            lr = optimizer.param_groups[0][LEARNING_RATE_PARAM_GROUP_SPECIFIER]
            elapsed_time = time.time() - start_time
            print(
                (
                    "Epoch Step: %6d | Accumulation Step: %3d | Loss: %6.2f "
                    + "| Tokens / Sec: %7.1f | Learning Rate: %6.1e"
                )
                % (i, num_accumulated, (loss / batch.ntokens), (tokens / elapsed_time), lr)
            )
            start_time = time.time()
            tokens = 0
        del loss
        del loss_node
    return (total_loss / total_tokens), train_state

def train_worker(
    device: torch.device,
    ngpus_per_node: int,
    src_tok: PreTrainedTokenizer,
    tgt_tok: PreTrainedTokenizer,
    train_ds: Dataset,
    valid_ds: Dataset,
    config: dict,
    is_distributed: bool = False,
) -> None:
    """Docstring here."""
    print(f"Train worker process using device: {device} for training", flush=True)
    torch.cuda.set_device(device)

    pad_idx = tgt_tok.pad_token_id
    d_model = DEFAULT_D_MODEL
    model = Transformer.make_model(src_tok.vocab_size, tgt_tok.vocab_size, N=DEFAULT_NUM_HEADS)
    model.cuda(device)
    
    module = model
    is_main_process = True

    # Let's implement this later.
    if is_distributed:
        pass
    
    criterion = LabelSmoothing(
        size=tgt_tok.vocab_size, padding_idx=pad_idx, smoothing=DEFAULT_SMOOTHING
    )
    criterion.cuda(device)

    optimizer = torch.optim.Adam(
        model.parameters(), lr=config["base_lr"], betas=(0.9, 0.98), eps=1e-9
    )
    lr_scheduler = LambdaLR(
        optimizer=optimizer,
        lr_lambda=lambda step: rate(
            step, d_model, factor=1, warmup_steps=config["warmup"]
        ),
    )
    train_state = TrainState()

    train_dataloader, valid_dataloader = create_dataloaders(
        device=device, 
        src_tok=src_tok, 
        tgt_tok=tgt_tok, 
        train_ds=train_ds, 
        valid_ds=valid_ds,
        batch_size=config["batch_size"],
        max_len=config["max_length"],
        is_distributed=is_distributed,
    )

    for epoch in range(config["num_epochs"]):
        if is_distributed:
            pass

        model.train()
        print(f"[Device:{device}] Epoch {epoch} Training ====", flush=True)
        _, train_state = run_epoch(
            (Batch(b[0], b[1], pad_idx) for b in train_dataloader),
            model,
            SimpleLossCompute(module.generator, criterion),
            optimizer,
            lr_scheduler,
            mode=MODE_TRAIN_AND_LOG,
            accum_iter=config["accum_iter"],
            train_state=train_state,
        )

        if is_main_process:
            file_path = "%s%.2d.pt" % (config["file_prefix"], epoch)
            torch.save(module.state_dict(), file_path)
        torch.cuda.empty_cache()

        print(f"[Device:{device}] Epoch {epoch} Validation ====", flush=True)
        model.eval()
        sloss = run_epoch(
            (Batch(b[0], b[1], pad_idx) for b in valid_dataloader),
            model,
            SimpleLossCompute(module.generator, criterion),
            DummyOptimizer(),
            DummyScheduler(),
            mode=MODE_EVAL,
        )
        print(sloss)
        torch.cuda.empty_cache()
    
    if is_main_process:
        file_path = "%sfinal.pt" % config["file_prefix"]
        torch.save(module.state_dict(), file_path)

def train_model(
    src_tok: PreTrainedTokenizer,
    tgt_tok: PreTrainedTokenizer,
    train_ds: Dataset,
    valid_ds: Dataset,
    config: dict,
) -> None:
    if config["distributed"]:
        pass
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        train_worker(
            device, 1, src_tok, tgt_tok, train_ds, valid_ds, config, is_distributed=False,
        )

def load_trained_model():
    config = {
        "batch_size": 32,
        "distributed": False,
        "num_epochs": 8,
        "accum_iter": 10,
        "base_lr": 1.0,
        "max_padding": 72,
        "warmup": 3000,
        "file_prefix": "multi30k_model_"
    }
    model_path = "multi30k_model_final.pt"
    if not exists(model_path):
        print("No model found. Training new model ...")
        src_tok, tgt_tok = load_tokenizers()
        train_ds, valid_ds = load_multi30k()
        train_model(src_tok, tgt_tok, train_ds, valid_ds, config)
    
    model: Transformer = Transformer.make_model(src_tok.vocab_size, tgt_tok.vocab_size, N=6)
    model.load_state_dict(torch.load("multi30k_model_final.pt"))
    return model