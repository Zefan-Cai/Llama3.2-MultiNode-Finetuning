from dataclasses import dataclass, field
import json
import math
import logging
import os
from typing import Dict, Optional, List
import torch
from torch.utils.data import Dataset, DataLoader
from deepspeed import zero
from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus
import transformers
from transformers import GPTQConfig, deepspeed
from transformers.trainer_pt_utils import LabelSmoother
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from accelerate.utils import DistributedType
from transformers import BitsAndBytesConfig
from torch.optim import AdamW
# from torch.optim.lr_scheduler import get_scheduler
from transformers import get_linear_schedule_with_warmup
import torch.distributed as dist

IGNORE_TOKEN_ID = LabelSmoother.ignore_index


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="meta-llama/Meta-Llama-3-8B-Instruct")


@dataclass
class DataArguments:
    data_path: str = field(
        default=None, metadata={"help": "Path to the training data."}
    )
    eval_data_path: str = field(
        default=None, metadata={"help": "Path to the evaluation data."}
    )
    lazy_preprocess: bool = False


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(
        default=8192,
        metadata={
            "help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )
    use_lora: bool = False


@dataclass
class LoraArguments:
    lora_r: int = 64
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    # ['gate_proj', 'o_proj', 'k_proj', 'q_proj', 'up_proj', 'down_proj', 'v_proj']
    lora_target_modules: List[str] = field(
        default_factory=lambda: ['o_proj', 'k_proj', 'q_proj', 'v_proj']
    )
    # lora_target_modules = None
    lora_weight_path: str = ""
    lora_bias: str = "none"
    q_lora: bool = False
    load_in_4bit: bool = False
    load_in_8bit: bool = False


def maybe_zero_3(param):
    if hasattr(param, "ds_id"):
        assert param.ds_status == ZeroParamStatus.NOT_AVAILABLE
        with zero.GatheredParameters([param]):
            param = param.data.detach().cpu().clone()
    else:
        param = param.detach().cpu().clone()
    return param


# Borrowed from peft.utils.get_peft_model_state_dict
def get_peft_state_maybe_zero_3(named_params, bias):
    if bias == "none":
        to_return = {k: t for k, t in named_params if "lora_" in k}
    elif bias == "all":
        to_return = {k: t for k, t in named_params if "lora_" in k or "bias" in k}
    elif bias == "lora_only":
        to_return = {}
        maybe_lora_bias = {}
        lora_bias_names = set()
        for k, t in named_params:
            if "lora_" in k:
                to_return[k] = t
                bias_name = k.split("lora_")[0] + "bias"
                lora_bias_names.add(bias_name)
            elif "bias" in k:
                maybe_lora_bias[k] = t
        for k, t in maybe_lora_bias:
            if bias_name in lora_bias_names:
                to_return[bias_name] = t
    else:
        raise NotImplementedError
    to_return = {k: maybe_zero_3(v) for k, v in to_return.items()}
    return to_return


local_rank = None


def rank0_print(*args):
    if local_rank == 0:
        print(*args)


def safe_save_model_for_hf_trainer(model, tokenizer, output_dir: str, bias="none"):
    """Collects the state dict and dump to disk."""
    # Unwrap the model from DDP if necessary
    if isinstance(model, torch.nn.parallel.DistributedDataParallel):
        model_to_save = model.module
    else:
        model_to_save = model

    # Handle DeepSpeed ZeRO stage 3 saving if enabled
    if deepspeed.is_deepspeed_zero3_enabled():
        # DeepSpeed engine stores the model in engine.module
        state_dict = model_to_save._zero3_consolidated_16bit_state_dict()
    else:
        if hasattr(model_to_save, 'peft_config'):
            state_dict = get_peft_state_maybe_zero_3(
                model_to_save.named_parameters(), bias
            )
        else:
            state_dict = model_to_save.state_dict()

    if local_rank == 0:
        os.makedirs(output_dir, exist_ok=True)
        model_to_save.save_pretrained(output_dir, state_dict=state_dict)
        tokenizer.save_pretrained(output_dir)


def preprocess(
        sources,
        tokenizer: transformers.PreTrainedTokenizer,
        max_len: int,
        system_message: str = "You are a pirate chatbot who always responds in pirate speak!"
) -> Dict:

    # im_start = tokenizer.im_start_id
    # im_end = tokenizer.im_end_id

    begin_of_text_id = tokenizer.get_vocab()["<|begin_of_text|>"]
    start_header_id = tokenizer.get_vocab()["<|start_header_id|>"]
    end_header_id = tokenizer.get_vocab()["<|end_header_id|>"]
    eot_id = tokenizer.get_vocab()["<|eot_id|>"]
    nl_tokens = tokenizer('\n').input_ids
    _system = tokenizer('system').input_ids
    _user = tokenizer('user').input_ids
    _assistant = tokenizer('assistant').input_ids

    # Apply prompt templates
    input_ids, targets = [], []
    for i, source in enumerate(sources):
        input_id, target = [], []
        system = [begin_of_text_id] + [start_header_id] + _system + [end_header_id] + nl_tokens + tokenizer(system_message).input_ids + [eot_id]
        input_id += system
        target += [IGNORE_TOKEN_ID] * len(input_id)
        assert len(input_id) == len(target)
        for j, sentence in enumerate(source):
            role = sentence["from"]
            value = sentence["value"]
            if role == 'user':
                _input_id = [start_header_id] + _user + [end_header_id] + nl_tokens + tokenizer(value).input_ids + [
                    eot_id]
                _target = [IGNORE_TOKEN_ID] * len(_input_id)
            elif role == 'assistant':
                _input_id = [start_header_id] + _assistant + [end_header_id] + nl_tokens + tokenizer(value).input_ids + [
                    eot_id]
                _target = [IGNORE_TOKEN_ID] + [IGNORE_TOKEN_ID] * len(_assistant) + \
                          [IGNORE_TOKEN_ID] + [IGNORE_TOKEN_ID] * len(nl_tokens) + tokenizer(value).input_ids + [eot_id]
            else:
                raise NotImplementedError
            input_id += _input_id
            target += _target
        # print(input_id)
        # print(target)
        # print(tokenizer.decode(input_id))
        # print(len(input_id), len(target))
        assert len(input_id) == len(target)
        input_id += [tokenizer.pad_token_id] * (max_len - len(input_id))
        target += [IGNORE_TOKEN_ID] * (max_len - len(target))
        input_ids.append(input_id[:max_len])
        targets.append(target[:max_len])
    input_ids = torch.tensor(input_ids, dtype=torch.long)
    targets = torch.tensor(targets, dtype=torch.long)

    return dict(
        input_ids=input_ids,
        labels=targets,
        attention_mask=input_ids.ne(tokenizer.pad_token_id),
    )


class SupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, raw_data, tokenizer: transformers.PreTrainedTokenizer, max_len: int):
        super(SupervisedDataset, self).__init__()

        rank0_print("Formatting inputs...")
        sources = [example["conversations"] for example in raw_data]
        data_dict = preprocess(sources, tokenizer, max_len)

        self.input_ids = data_dict["input_ids"]
        self.labels = data_dict["labels"]
        self.attention_mask = data_dict["attention_mask"]

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        return dict(
            input_ids=self.input_ids[i],
            labels=self.labels[i],
            attention_mask=self.attention_mask[i],
        )


class LazySupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, raw_data, tokenizer: transformers.PreTrainedTokenizer, max_len: int):
        super(LazySupervisedDataset, self).__init__()
        self.tokenizer = tokenizer
        self.max_len = max_len

        rank0_print("Formatting inputs...Skip in lazy mode")
        self.tokenizer = tokenizer
        self.raw_data = raw_data
        self.cached_data_dict = {}

    def __len__(self):
        return len(self.raw_data)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        if i in self.cached_data_dict:
            return self.cached_data_dict[i]

        ret = preprocess([self.raw_data[i]["conversations"]], self.tokenizer, self.max_len)
        ret = dict(
            input_ids=ret["input_ids"][0],
            labels=ret["labels"][0],
            attention_mask=ret["attention_mask"][0],
        )
        self.cached_data_dict[i] = ret

        return ret


def make_supervised_data_module(
        tokenizer: transformers.PreTrainedTokenizer, data_args, max_len,
) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    dataset_cls = (
        LazySupervisedDataset if data_args.lazy_preprocess else SupervisedDataset
    )
    rank0_print("Loading data...")

    train_json = json.load(open(data_args.data_path, "r"))
    train_dataset = dataset_cls(train_json, tokenizer=tokenizer, max_len=max_len)

    if data_args.eval_data_path:
        eval_json = json.load(open(data_args.eval_data_path, "r"))
        eval_dataset = dataset_cls(eval_json, tokenizer=tokenizer, max_len=max_len)
    else:
        eval_dataset = None

    return dict(train_dataset=train_dataset, eval_dataset=eval_dataset)


def get_quantization_config(model_args):
    if model_args.load_in_4bit:
        compute_dtype = torch.float16
        # if model_args.torch_dtype not in {"auto", None}:
        #     compute_dtype = getattr(torch, model_args.torch_dtype)

        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=False,
        )
    elif model_args.load_in_8bit:
        quantization_config = BitsAndBytesConfig(
            load_in_8bit=True,
        )
    else:
        quantization_config = None

    return quantization_config


def train():
    global local_rank

    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments, LoraArguments)
    )
    (
        model_args,
        data_args,
        training_args,
        lora_args,
    ) = parser.parse_args_into_dataclasses()

    # This serves for single-gpu qlora.
    if getattr(training_args, 'deepspeed', None) and int(os.environ.get("WORLD_SIZE", 1)) == 1:
        training_args.distributed_state.distributed_type = DistributedType.DEEPSPEED

    local_rank = training_args.local_rank

    device_map = None
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    ddp = world_size != 1
    if lora_args.q_lora:
        device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)} if ddp else "auto"
        if len(training_args.fsdp) > 0 or deepspeed.is_deepspeed_zero3_enabled():
            logging.warning(
                "FSDP or ZeRO3 are incompatible with QLoRA."
            )

    is_chat_model = 'instruct' in model_args.model_name_or_path.lower()
    if (
            training_args.use_lora
            and not lora_args.q_lora
            and deepspeed.is_deepspeed_zero3_enabled()
            and not is_chat_model
    ):
        raise RuntimeError("ZeRO3 is incompatible with LoRA when finetuning on base model.")

    model_load_kwargs = {
        'low_cpu_mem_usage': not deepspeed.is_deepspeed_zero3_enabled(),
    }

    # Set RoPE scaling factor
    config = transformers.AutoConfig.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        trust_remote_code=True,
    )
    config.use_cache = False

    # Load model and tokenizer
    quantization_config = get_quantization_config(lora_args)

    print("quantization_config：", quantization_config)

    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        config=config,
        cache_dir=training_args.cache_dir,
        device_map=device_map,
        trust_remote_code=True,
        token='YOUR_HUGGINGFACE_TOKEN',
        quantization_config=quantization_config if lora_args.q_lora else None,
        **model_load_kwargs,
    )
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=False,
        trust_remote_code=True,
        token='YOUR_HUGGINGFACE_TOKEN',
    )

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    if training_args.use_lora:
        if is_chat_model:
            modules_to_save = None
        else:
            modules_to_save = ["wte", "lm_head"]

        def find_all_linear_names(args, model):
            import bitsandbytes as bnb
            cls = bnb.nn.Linear4bit if args.load_in_4bit == 4 else (
                bnb.nn.Linear8bitLt if args.load_in_8bit == 8 else torch.nn.Linear)
            lora_module_names = set()
            for name, module in model.named_modules():
                if isinstance(module, cls):
                    names = name.split('.')
                    lora_module_names.add(names[0] if len(names) == 1 else names[-1])

            if 'lm_head' in lora_module_names:  # needed for 16-bit
                lora_module_names.remove('lm_head')
            return list(lora_module_names)

        if lora_args.lora_target_modules is None:
            lora_args.lora_target_modules = find_all_linear_names(lora_args, model)

        print(lora_args.lora_target_modules)
        print(modules_to_save)

        lora_config = LoraConfig(
            r=lora_args.lora_r,
            lora_alpha=lora_args.lora_alpha,
            target_modules=lora_args.lora_target_modules,
            lora_dropout=lora_args.lora_dropout,
            bias=lora_args.lora_bias,
            task_type="CAUSAL_LM",
            modules_to_save=modules_to_save  # This argument serves for adding new tokens.
        )
        if lora_args.q_lora:
            model = prepare_model_for_kbit_training(
                model, use_gradient_checkpointing=training_args.gradient_checkpointing
            )

        model = get_peft_model(model, lora_config)

        # Print peft trainable params
        model.print_trainable_parameters()

    if training_args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()

    # Load data
    data_module = make_supervised_data_module(
        tokenizer=tokenizer, data_args=data_args, max_len=training_args.model_max_length
    )
    train_dataset = data_module['train_dataset']
    eval_dataset = data_module['eval_dataset']

    # Set up DataLoaders
    train_sampler = None
    if training_args.local_rank != -1:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=training_args.per_device_train_batch_size,
        sampler=train_sampler,
        shuffle=(train_sampler is None),
    )

    if eval_dataset is not None:
        eval_sampler = None
        if training_args.local_rank != -1:
            eval_sampler = torch.utils.data.distributed.DistributedSampler(eval_dataset)
        eval_dataloader = DataLoader(
            eval_dataset,
            batch_size=training_args.per_device_eval_batch_size,
            sampler=eval_sampler,
            shuffle=False,
        )
    else:
        eval_dataloader = None

    # Prepare optimizer and scheduler
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if p.requires_grad],
            "weight_decay": training_args.weight_decay,
        },
    ]

    optimizer = AdamW(optimizer_grouped_parameters, lr=training_args.learning_rate)

    # Prepare scheduler
    total_steps = (
        len(train_dataloader) // training_args.gradient_accumulation_steps * training_args.num_train_epochs
    )
    warmup_steps = training_args.get_warmup_steps(total_steps)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
    )

    # Prepare model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Mixed precision
    if training_args.fp16:
        scaler = torch.cuda.amp.GradScaler()
    else:
        scaler = None

    # Distributed training
    if training_args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[training_args.local_rank], output_device=training_args.local_rank
        )

    # Training loop
    global_step = 0
    # for epoch in range(int(training_args.num_train_epochs)):
    # Inside the training loop
    for epoch in range(int(training_args.num_train_epochs)):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        model.train()
        running_loss = 0.0
        for step, batch in enumerate(train_dataloader):
            batch = {k: v.to(device) for k, v in batch.items()}
            # with torch.cuda.amp.autocast(enabled=training_args.fp16):
            with torch.amp.autocast('cuda', enabled=training_args.fp16):
                outputs = model(**batch)
                loss = outputs.loss
                loss = loss / training_args.gradient_accumulation_steps

            if scaler is not None:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            # Compute gradient norm
            total_norm = 0.0
            parameters = [p for p in model.parameters() if p.grad is not None]
            if training_args.max_grad_norm is not None and training_args.max_grad_norm > 0:
                # Unscaling gradients before computing the norm if using mixed precision
                if scaler is not None:
                    scaler.unscale_(optimizer)
                total_norm = torch.norm(
                    torch.stack([torch.norm(p.grad.detach(), 2) for p in parameters]), 2
                ).item()

            if (step + 1) % training_args.gradient_accumulation_steps == 0:
                if training_args.max_grad_norm is not None and training_args.max_grad_norm > 0:
                    # Gradient clipping
                    if scaler is not None:
                        torch.nn.utils.clip_grad_norm_(parameters, training_args.max_grad_norm)
                    else:
                        torch.nn.utils.clip_grad_norm_(parameters, training_args.max_grad_norm)
                if scaler is not None:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

                # Get current learning rate
                current_lr = scheduler.get_last_lr()[0]

                # Print learning rate and gradient norm
                # if training_args.logging_steps > 0 and global_step % training_args.logging_steps == 0:
                rank0_print(
                    f"Epoch {epoch+1}, Global Step {global_step}, Loss: {loss.item() * training_args.gradient_accumulation_steps:.4f}, "
                    f"LR: {current_lr:.8f}, Grad Norm: {total_norm:.4f}"
                )

                if local_rank == 0:
                    if training_args.save_steps > 0 and global_step % training_args.save_steps == 0 and local_rank == 0:
                        # Save the model
                        output_dir = os.path.join(training_args.output_dir, f"checkpoint-{global_step}")
                        safe_save_model_for_hf_trainer(model, tokenizer, output_dir, bias=lora_args.lora_bias)


        # Evaluation at the end of each epoch
        if eval_dataloader is not None:
            model.eval()
            eval_loss = 0.0
            eval_steps = 0
            for batch in eval_dataloader:
                batch = {k: v.to(device) for k, v in batch.items()}
                with torch.no_grad():
                    with torch.cuda.amp.autocast(enabled=training_args.fp16):
                        outputs = model(**batch)
                        loss = outputs.loss
                        eval_loss += loss.item()
                        eval_steps += 1
            eval_loss = eval_loss / eval_steps
            rank0_print(f"Epoch {epoch+1}, Evaluation Loss: {eval_loss}")

    # Save the final model
    if local_rank == 0:
        safe_save_model_for_hf_trainer(model, tokenizer, training_args.output_dir, bias=lora_args.lora_bias)


if __name__ == "__main__":
    train()
