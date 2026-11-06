import multiprocessing as mp
import queue
import time
import traceback

import torch
from peft import (
    LoraConfig,
    get_peft_model_state_dict,
    set_peft_model_state_dict,
)
from transformers import AutoTokenizer, set_seed
from trl import AutoModelForCausalLMWithValueHead


def _adapter_model(model):
    return model.pretrained_model if hasattr(model, "pretrained_model") else model


def _adapter_metadata(model):
    state = get_peft_model_state_dict(_adapter_model(model))
    return [
        (name, tuple(tensor.shape), tensor.numel())
        for name, tensor in sorted(state.items())
    ]


def _copy_adapter_to_shared(model, shared_state, metadata):
    state = get_peft_model_state_dict(_adapter_model(model))
    offset = 0
    for name, shape, numel in metadata:
        tensor = state.get(name)
        if tensor is None or tuple(tensor.shape) != shape:
            raise RuntimeError(f"PPO adapter state mismatch for {name}")
        shared_state[offset : offset + numel].copy_(
            tensor.detach().reshape(-1).to(device="cpu", dtype=torch.float32)
        )
        offset += numel


def _copy_shared_to_adapter(model, shared_state, metadata):
    current = get_peft_model_state_dict(_adapter_model(model))
    state = {}
    offset = 0
    for name, shape, numel in metadata:
        if name not in current or tuple(current[name].shape) != shape:
            raise RuntimeError(f"rollout-worker adapter state mismatch for {name}")
        state[name] = shared_state[offset : offset + numel].reshape(shape).clone()
        offset += numel
    result = set_peft_model_state_dict(_adapter_model(model), state)
    unexpected = list(getattr(result, "unexpected_keys", []) or [])
    if unexpected:
        raise RuntimeError(f"unexpected rollout adapter keys: {unexpected[:5]}")


def _load_worker_model(
    model_dir,
    resume_dir,
    target_modules,
    device,
    torch_dtype,
):
    if resume_dir:
        model = AutoModelForCausalLMWithValueHead.from_pretrained(
            resume_dir,
            torch_dtype=torch_dtype,
            trust_remote_code=True,
            local_files_only=True,
            is_trainable=False,
        )
    else:
        peft_config = LoraConfig(
            r=8,
            lora_alpha=16,
            lora_dropout=0.05,
            bias="none",
            fan_in_fan_out=False,
            target_modules=target_modules,
        )
        model = AutoModelForCausalLMWithValueHead.from_pretrained(
            model_dir,
            torch_dtype=torch_dtype,
            peft_config=peft_config,
            trust_remote_code=True,
            local_files_only=True,
        )
    model.to(device)
    model.train()
    return model


def _generate_batch(model, tokenizer, query_ids, generation_kwargs, device):
    queries = [torch.tensor(ids, dtype=torch.long) for ids in query_ids]
    attention_masks = [torch.ones_like(query) for query in queries]
    padding_side = tokenizer.padding_side
    tokenizer.padding_side = "left"
    try:
        padded = tokenizer.pad(
            {"input_ids": queries, "attention_mask": attention_masks},
            padding=True,
            return_tensors="pt",
        ).to(device)
        with torch.inference_mode():
            generations = model.generate(**padded, **generation_kwargs)

        outputs = []
        for generation, mask in zip(generations, padded["attention_mask"]):
            output = generation[(1 - mask).sum() :]
            eos_positions = torch.nonzero(
                output == tokenizer.eos_token_id, as_tuple=False
            )
            if eos_positions.numel():
                output = output[: eos_positions[0, 0].item() + 1]
            outputs.append(output.detach().cpu().tolist())
        return outputs
    finally:
        tokenizer.padding_side = padding_side


def _rollout_worker(
    worker_index,
    local_device,
    model_dir,
    resume_dir,
    target_modules,
    tokenizer_dir,
    shared_state,
    metadata,
    command_queue,
    result_queue,
    seed,
):
    try:
        torch.cuda.set_device(local_device)
        device = torch.device(f"cuda:{local_device}")
        set_seed(seed + worker_index + 1)
        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_dir, trust_remote_code=True, local_files_only=True
        )
        if tokenizer.eos_token is None:
            raise ValueError("rollout-worker tokenizer has no EOS token")
        tokenizer.pad_token = tokenizer.eos_token
        model = _load_worker_model(
            model_dir=model_dir,
            resume_dir=resume_dir,
            target_modules=target_modules,
            device=device,
            torch_dtype=torch.bfloat16,
        )
        model.pretrained_model.config.pad_token_id = tokenizer.pad_token_id
        _copy_shared_to_adapter(model, shared_state, metadata)
        result_queue.put(("ready", worker_index, local_device, None))

        while True:
            command = command_queue.get()
            if command[0] == "stop":
                break
            _, batch_id, positions, query_ids, generation_kwargs = command
            started = time.perf_counter()
            _copy_shared_to_adapter(model, shared_state, metadata)
            outputs = _generate_batch(
                model, tokenizer, query_ids, generation_kwargs, device
            )
            result_queue.put(
                (
                    "result",
                    worker_index,
                    batch_id,
                    positions,
                    outputs,
                    time.perf_counter() - started,
                )
            )
    except BaseException:
        result_queue.put(("error", worker_index, traceback.format_exc()))


class ParallelRolloutPool:
    def __init__(
        self,
        policy_model,
        model_dir,
        resume_dir,
        target_modules,
        tokenizer_dir,
        worker_devices,
        seed,
        timeout=600,
    ):
        if not worker_devices:
            raise ValueError("at least one rollout worker device is required")
        self.timeout = timeout
        self.metadata = _adapter_metadata(policy_model)
        total_parameters = sum(item[2] for item in self.metadata)
        self.shared_state = torch.empty(total_parameters, dtype=torch.float32)
        self.shared_state.share_memory_()
        _copy_adapter_to_shared(policy_model, self.shared_state, self.metadata)

        self.context = mp.get_context("spawn")
        self.result_queue = self.context.Queue()
        self.command_queues = []
        self.processes = []
        for worker_index, local_device in enumerate(worker_devices):
            command_queue = self.context.Queue(maxsize=1)
            process = self.context.Process(
                target=_rollout_worker,
                args=(
                    worker_index,
                    local_device,
                    str(model_dir),
                    str(resume_dir) if resume_dir else "",
                    list(target_modules),
                    str(tokenizer_dir),
                    self.shared_state,
                    self.metadata,
                    command_queue,
                    self.result_queue,
                    seed,
                ),
                daemon=True,
            )
            process.start()
            self.command_queues.append(command_queue)
            self.processes.append(process)

        ready = set()
        deadline = time.monotonic() + timeout
        while len(ready) != len(self.processes):
            message = self._get_result(deadline)
            if message[0] == "error":
                raise RuntimeError(
                    f"rollout worker {message[1]} failed during startup:\n{message[2]}"
                )
            if message[0] != "ready":
                raise RuntimeError(f"unexpected rollout-worker message: {message[0]}")
            ready.add(message[1])

        self.batch_id = 0
        self.total_generation_seconds = 0.0
        self.total_sync_seconds = 0.0

    def _get_result(self, deadline):
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError("timed out waiting for rollout worker")
        try:
            return self.result_queue.get(timeout=remaining)
        except queue.Empty as exc:
            raise TimeoutError("timed out waiting for rollout worker") from exc

    def publish_policy(self, policy_model):
        started = time.perf_counter()
        _copy_adapter_to_shared(policy_model, self.shared_state, self.metadata)
        self.total_sync_seconds += time.perf_counter() - started

    def generate(self, query_tensors, generation_kwargs, local_generate):
        executor_count = len(self.processes) + 1
        if len(query_tensors) < executor_count:
            raise ValueError(
                "rollout batch must contain at least one query per GPU executor"
            )
        self.batch_id += 1
        wall_started = time.perf_counter()
        local_assignment = []
        assignments = [[] for _ in self.processes]
        for position, query in enumerate(query_tensors):
            executor = position % executor_count
            item = (position, query.detach().cpu().tolist())
            if executor == 0:
                local_assignment.append(item)
            else:
                assignments[executor - 1].append(item)

        active_workers = 0
        for worker_index, assignment in enumerate(assignments):
            if not assignment:
                continue
            positions, query_ids = zip(*assignment)
            self.command_queues[worker_index].put(
                (
                    "generate",
                    self.batch_id,
                    list(positions),
                    list(query_ids),
                    dict(generation_kwargs),
                )
            )
            active_workers += 1

        outputs = [None] * len(query_tensors)
        local_positions = [item[0] for item in local_assignment]
        local_queries = [query_tensors[position] for position in local_positions]
        for position, output in zip(local_positions, local_generate(local_queries)):
            outputs[position] = output

        deadline = time.monotonic() + self.timeout
        for _ in range(active_workers):
            message = self._get_result(deadline)
            if message[0] == "error":
                raise RuntimeError(
                    f"rollout worker {message[1]} failed:\n{message[2]}"
                )
            if message[0] != "result" or message[2] != self.batch_id:
                raise RuntimeError("received an out-of-order rollout-worker result")
            _, _, _, positions, generated, _ = message
            for position, output in zip(positions, generated):
                outputs[position] = torch.tensor(
                    output,
                    dtype=torch.long,
                    device=query_tensors[position].device,
                )
        if any(output is None for output in outputs):
            raise RuntimeError("parallel rollout returned an incomplete batch")
        self.total_generation_seconds += time.perf_counter() - wall_started
        return outputs

    def close(self):
        for command_queue in self.command_queues:
            try:
                command_queue.put_nowait(("stop",))
            except queue.Full:
                pass
        for process in self.processes:
            process.join(timeout=30)
            if process.is_alive():
                process.terminate()
                process.join(timeout=10)
