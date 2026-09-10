#!/usr/bin/env python3
"""Portable Spatial checkpoint evaluation with batched episodes and routing diagnostics.

This explicit subset experiment records a new serving topology and does not emit
the repository's preregistered official-score schema. Checkpoint bytes and model,
normalization, LoRA, and prefix identities remain authenticated.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import sys
import time
from collections import deque
from multiprocessing.connection import Client, Listener
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
BATCH_SIZE = 8


def write_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def send(connection: Any, message: dict[str, Any]) -> None:
    connection.send_bytes(json.dumps(message, allow_nan=False, separators=(",", ":")).encode())


def receive(connection: Any) -> dict[str, Any]:
    value = json.loads(connection.recv_bytes(16 * 1024 * 1024))
    if not isinstance(value, dict):
        raise ValueError("IPC message must be an object")
    return value


def encode_image(image: np.ndarray) -> str:
    if image.shape != (256, 256, 3) or image.dtype != np.uint8:
        raise ValueError("expected uint8 RGB image [256,256,3]")
    return base64.b64encode(image.tobytes()).decode("ascii")


def decode_image(value: str) -> np.ndarray:
    data = base64.b64decode(value, validate=True)
    if len(data) != 256 * 256 * 3:
        raise ValueError("invalid image byte count")
    return np.frombuffer(data, dtype=np.uint8).reshape(256, 256, 3).copy()


class SpatialPolicy:
    """Keep each tied encoder/decoder layer pair on the same one of two GPUs."""

    def __init__(self, checkpoint: Path, nfe: int, trace_directory: Path | None = None) -> None:
        import torch
        from accelerate.hooks import remove_hook_from_module
        from transformers import AutoProcessor, DiffusionGemmaForBlockDiffusion

        from duo_vla.action_interface import ActionInputProjector, VelocityHead
        from duo_vla.backbones.diffusion_gemma import DiffusionGemmaActionDecoder
        from duo_vla.backbones.sample_isolated_experts_v2 import (
            install_sample_isolated_grouped_mm_experts_v2,
            verify_sample_isolated_grouped_mm_experts_v2,
        )
        from duo_vla.checkpointing import load_checkpoint_manifest, load_interface_state_dict, load_lora_checkpoint
        from duo_vla.config import ActionInterfaceConfig
        from duo_vla.data.libero_stats import load_libero_normalizers
        from duo_vla.expert_routing import DecoderRoutingTrace, RoutingAccumulator
        from duo_vla.modeling import DuoVLADenoiser
        from duo_vla.policy_contract import validate_manifest_policy_contract

        if torch.cuda.device_count() != 2:
            raise RuntimeError("this serving profile requires exactly two visible CUDA devices")
        self.torch, self.device, self.nfe = torch, torch.device("cuda:0"), nfe
        self.manifest = load_checkpoint_manifest(checkpoint)
        resolved = json.loads((checkpoint / "artifacts/resolved_config.json").read_text())
        self.config = config = resolved["config"]
        config_digest = hashlib.sha256(json.dumps(config, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        if config_digest != resolved["config_sha256"] or config_digest != self.manifest["config_sha256"]:
            raise ValueError("resolved checkpoint config hash mismatch")
        contract = validate_manifest_policy_contract(self.manifest, config)
        if contract.objective != "rectified_flow" or contract.action_horizon != 8 or contract.action_dim != 7:
            raise ValueError("Spatial runner requires the eight-action rectified-flow checkpoint")
        if config["model"]["expert_batch_isolation"] != "sample_isolated_grouped_mm_v2":
            raise ValueError("this portable profile requires the fused-v2 checkpoint")
        self.prefix_geometry = json.loads((checkpoint / "artifacts/prefix_geometry.json").read_text())
        self.valid_lengths = {
            record["instruction"]: record["valid_prefix_length"]
            for record in self.prefix_geometry["instruction_inventory"]["records"]
        }
        model_spec = config["model"]
        self.processor = AutoProcessor.from_pretrained(
            model_spec["id"],
            revision=model_spec["revision"],
            local_files_only=True,
        )
        # Mapping the root as well as descendants makes Accelerate's root hook
        # relocate all descendants to GPU 0. Use disjoint complete blocks.
        self.device_map = {
            "model.encoder.vision_tower": 0,
            "model.encoder.embed_vision": 0,
            "model.decoder.self_conditioning": 0,
            "lm_head": 0,
        }
        for stack in ("model.encoder.language_model", "model.decoder"):
            for component in ("embed_tokens", "norm", "rotary_emb"):
                self.device_map[f"{stack}.{component}"] = 0
            for index in range(30):
                self.device_map[f"{stack}.layers.{index}"] = int(index >= 15)
        self.model = DiffusionGemmaForBlockDiffusion.from_pretrained(
            model_spec["id"],
            revision=model_spec["revision"],
            local_files_only=True,
            dtype=torch.bfloat16,
            device_map=self.device_map,
            attn_implementation="sdpa",
            experts_implementation="grouped_mm",
        )
        self.model.requires_grad_(False)
        for stack in ("model.encoder.language_model", "model.decoder"):
            for layer in self.model.get_submodule(stack).layers:
                # The enclosing layer already aligns inputs to its GPU. Remove
                # redundant expert dispatch wrappers before installing fused-v2.
                experts = layer.experts
                remove_hook_from_module(experts)
                if getattr(experts.forward, "__func__", None) is not type(experts).forward:
                    raise RuntimeError("expert dispatch did not restore its original forward")
                experts.__dict__.pop("forward", None)
        install_sample_isolated_grouped_mm_experts_v2(self.model, physical_batch_size=BATCH_SIZE)
        self.adapted, _ = load_lora_checkpoint(
            checkpoint,
            self.model,
            is_trainable=False,
            validate_decoder_contract=True,
            expected_rank=config["lora"]["rank"],
        )
        verify_sample_isolated_grouped_mm_experts_v2(self.model, physical_batch_size=BATCH_SIZE)
        action = config["action"]
        interface = ActionInterfaceConfig(
            hidden_size=2816,
            state_dim=8,
            action_horizon=action["horizon"],
            action_dim=action["dimension"],
            timestep_embedding_dim=action["timestep_embedding_dimension"],
            timestep_scale=action["timestep_scale"],
            timestep_max_period=action["timestep_max_period"],
            output_init_std=action["output_head_initialization_std"],
        )
        projector = ActionInputProjector(interface).to(self.device)
        head = VelocityHead(interface.hidden_size, interface.action_dim).to(self.device)
        load_interface_state_dict(
            checkpoint / "interface.safetensors",
            {"action_projector": projector, "velocity_head": head},
        )
        backend = DiffusionGemmaActionDecoder.from_block_diffusion_model(self.model)
        self.denoiser = DuoVLADenoiser(projector, backend, head).eval()
        self.adapted.eval()
        state_norm, self.action_norm, _ = load_libero_normalizers(checkpoint / "artifacts/normalization.json")
        self.state_norm = state_norm.to(device=self.device, dtype=torch.float32)
        self.routing = RoutingAccumulator(128)
        self.trace = DecoderRoutingTrace(trace_directory, nfe) if trace_directory is not None else None
        self.phase = "disabled"
        self.valid_mask: np.ndarray | None = None
        self.task_ids: list[int] = []
        self.hooks = []
        for stack in ("model.encoder.language_model", "model.decoder"):
            for index, layer in enumerate(self.model.get_submodule(stack).layers):
                name = f"{stack}.layers.{index}"
                self._guard_expert_device(layer.experts)
                self.hooks.append(layer.experts.register_forward_pre_hook(self._routing_hook(name), with_kwargs=True))

    def _guard_expert_device(self, experts: Any) -> None:
        # Triton uses the current CUDA device for its launch. Accelerate moves
        # tensors but does not enter a CUDA device context around custom kernels.
        guards = []

        def enter(module: Any, _args: Any) -> None:
            guard = self.torch.cuda.device(module.gate_up_proj.device)
            guard.__enter__()
            guards.append(guard)

        def leave(_module: Any, _args: Any, _output: Any) -> None:
            if guards:
                guards.pop().__exit__(None, None, None)

        self.hooks.append(experts.register_forward_pre_hook(enter))
        self.hooks.append(experts.register_forward_hook(leave, always_call=True))

    def _routing_hook(self, name: str) -> Any:
        def hook(_module: Any, args: tuple[Any, ...], kwargs: dict[str, Any]) -> None:
            if self.phase == "disabled":
                return
            indices = args[1] if len(args) > 1 else kwargs["top_k_index"]
            weights = args[2] if len(args) > 2 else kwargs["top_k_weights"]
            ids = indices.detach().reshape(BATCH_SIZE, -1, indices.shape[-1]).cpu().numpy()
            gates = weights.detach().float().reshape(ids.shape).cpu().numpy()
            assert self.valid_mask is not None
            self.routing.observe(name, self.phase, ids, gates, self.valid_mask, self.task_ids)
            if self.trace is not None and name.startswith("model.decoder."):
                self.trace.observe(
                    int(name.rsplit(".", 1)[1]), int(self.phase.split("_")[1]), ids, gates, self.valid_mask
                )

        return hook

    def predict(self, rows: list[dict[str, Any]], *, record: bool = True) -> dict[str, Any]:
        from PIL import Image

        from duo_vla.backbones.diffusion_gemma import encode_diffusion_gemma_prefix
        from duo_vla.prefix_geometry import apply_fixed_prefix_chat_template

        torch = self.torch
        if not 1 <= len(rows) <= BATCH_SIZE:
            raise ValueError("policy batch must contain one to eight observations")
        active = len(rows)
        if record and self.trace is not None:
            self.trace.begin(rows)
        padded = rows + [rows[-1]] * (BATCH_SIZE - active)
        self.task_ids = [row["task_id"] for row in padded]
        conversations = []
        for row in padded:
            if row["instruction"] not in self.valid_lengths:
                raise ValueError("instruction is missing from the checkpoint prefix inventory")
            conversations.append(
                [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "image": Image.fromarray(decode_image(row["agentview"]))},
                            {"type": "image", "image": Image.fromarray(decode_image(row["wrist"]))},
                            {"type": "text", "text": row["instruction"]},
                        ],
                    }
                ]
            )
        for device in range(2):
            torch.cuda.synchronize(device)
        started = time.perf_counter()
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            inputs = apply_fixed_prefix_chat_template(
                self.processor,
                conversations,
                fixed_physical_prefix_width=self.config["benchmark"]["fixed_physical_prefix_width"],
                padding_side=self.prefix_geometry["tokenization"]["padding_side"],
                expected_batch_size=BATCH_SIZE,
                images_per_prefix=2,
            ).to(self.device)
            mask = inputs["attention_mask"].bool()
            if mask.sum(dim=1).tolist() != [self.valid_lengths[row["instruction"]] for row in padded]:
                raise ValueError("processor token counts differ from the checkpoint prefix geometry")
            self.valid_mask = mask.cpu().numpy().copy()
            self.valid_mask[active:] = False
            self.phase = "prefix" if record else "disabled"
            prefix = encode_diffusion_gemma_prefix(self.model, inputs)
            state = torch.tensor([row["state"] for row in padded], dtype=torch.float32, device=self.device)
            state = self.state_norm.normalize(state)
            actions = torch.cat(
                [
                    torch.randn(
                        (1, 8, 7),
                        device=self.device,
                        dtype=torch.float32,
                        generator=torch.Generator(device=self.device).manual_seed(row["inference_seed"]),
                    )
                    for row in padded
                ]
            )
            action_mask = torch.ones((BATCH_SIZE, 8), dtype=torch.bool, device=self.device)
            self.valid_mask = np.ones((BATCH_SIZE, 8), dtype=bool)
            self.valid_mask[active:] = False
            for step in range(self.nfe):
                self.phase = f"denoise_{step}" if record else "disabled"
                velocity = self.denoiser(
                    actions,
                    torch.full((BATCH_SIZE,), step / self.nfe, device=self.device),
                    state,
                    prefix_cache=prefix.past_key_values,
                    prefix_attention_mask=prefix.attention_mask,
                    action_valid_mask=action_mask,
                )
                actions = actions + velocity.float() / self.nfe
            self.phase = "disabled"
            if not bool(torch.isfinite(actions).all()):
                raise FloatingPointError("policy generated nonfinite normalized actions")
            clip_fraction = (actions.abs() > 1).float().mean(dim=(1, 2))[:active].cpu().tolist()
            output = self.action_norm.unnormalize(actions.clamp(-1, 1))[:active].float().cpu().numpy()
        for device in range(2):
            torch.cuda.synchronize(device)
        if record and self.trace is not None:
            self.trace.finish()
        return {"actions": output.tolist(), "clip_fraction": clip_fraction, "seconds": time.perf_counter() - started}


def run_server(args: argparse.Namespace) -> None:
    import torch

    from duo_vla.runtime_determinism import configure_strict_cuda_determinism

    configure_strict_cuda_determinism(torch)
    args.output.mkdir(parents=True, exist_ok=True)
    policy = SpatialPolicy(args.checkpoint, args.nfe, args.output / "routing_trace" if args.trace_routing else None)
    metadata = {
        "schema": "duo-vla-portable-spatial-serving-v1",
        "nfe": args.nfe,
        "checkpoint_manifest_sha256": sha256(args.checkpoint / "manifest.json"),
        "checkpoint": str(args.checkpoint),
        "training_execution_geometry": policy.manifest["execution_geometry"],
        "serving_topology": "two GPUs with tied layer pairs placed together; no tensor parallelism",
        "cuda_visible_devices": os.environ["CUDA_VISIBLE_DEVICES"],
        "device_map": policy.device_map,
        "physical_batch_size": BATCH_SIZE,
        "decoder_trace": {
            "enabled": args.trace_routing,
            "directory": "routing_trace",
            "axes": ["active_episode", "denoising_phase", "decoder_layer", "expert"],
            "policy_step": "zero-based model invocation index within each episode; excludes warmup",
            "environment_step": "executed actions before this invocation; excludes settling",
            "counts": "all top-8 expert assignments over the eight predicted action tokens",
            "top1_counts": "largest actual gate weight per action token",
        },
        "torch": torch.__version__,
        "source_files": {
            str(path.relative_to(ROOT)): sha256(path)
            for path in [
                Path(__file__),
                ROOT / "src/duo_vla/expert_routing.py",
                ROOT / "src/duo_vla/backbones/shared_weight_grouped_mm_triton.py",
            ]
        },
    }
    write_json(args.output / "serving.json", metadata)
    authkey = args.authkey.read_bytes()
    with Listener(str(args.socket), family="AF_UNIX", authkey=authkey) as listener:
        os.chmod(args.socket, 0o600)
        print(json.dumps({"status": "ready", "socket": str(args.socket)}), flush=True)
        with listener.accept() as connection:
            try:
                while True:
                    message = receive(connection)
                    operation = message["operation"]
                    if operation == "shutdown":
                        send(connection, {"status": "stopped"})
                        break
                    if operation == "routing":
                        write_json(args.output / "expert_routing.json", policy.routing.report())
                        send(connection, {"status": "saved"})
                        continue
                    if operation != "predict":
                        raise ValueError("unknown policy operation")
                    response = policy.predict(message["rows"], record=message.get("record", True))
                    send(connection, response)
            finally:
                write_json(args.output / "expert_routing.json", policy.routing.report())


def run_evaluation(args: argparse.Namespace) -> None:
    from evaluate_libero import _construct_environment, libero_env_action, libero_state, rotate_eval_rgb_180, wilson95
    from libero.libero import benchmark
    from libero_bridge import libero_replan_seed

    args.output.mkdir(parents=True, exist_ok=False)
    suite = benchmark.get_benchmark_dict()["libero_spatial"]()
    matrix = [(task_id, reset_id) for task_id in range(10) for reset_id in range(args.resets_per_task)]
    write_json(
        args.output / "experiment.json",
        {
            "schema": "duo-vla-spatial-subset-experiment-v1",
            "suite": "libero_spatial",
            "nfe": args.nfe,
            "execution_horizon": args.execution_horizon,
            "evaluation_seed": args.evaluation_seed,
            "environment_seed": 7,
            "max_policy_steps": 220,
            "settle_steps": 10,
            "episode_matrix": matrix,
            "target_episodes": len(matrix),
            "routing_instrumented": True,
            "reset_source": "published official initial states",
        },
    )
    pending = deque(matrix)
    active: list[dict[str, Any]] = []
    completed: list[dict[str, Any]] = []
    started = time.perf_counter()
    batch_calls = 0
    batch_seconds: list[float] = []

    def initialize(task_id: int, reset_id: int) -> dict[str, Any]:
        task = suite.get_task(task_id)
        initial_state = suite.get_task_init_states(task_id)[reset_id]
        environment = _construct_environment(task)
        environment.seed(7)
        environment.reset()
        observation = environment.set_init_state(initial_state)
        for _ in range(10):
            observation, _, _, _ = environment.step(np.array([0, 0, 0, 0, 0, 0, -1], dtype=np.float32))
        return {
            "task_id": task_id,
            "reset_id": reset_id,
            "task_name": task.name,
            "instruction": task.language,
            "environment": environment,
            "observation": observation,
            "steps": 0,
            "calls": 0,
            "success": bool(environment.check_success()),
            "started": time.perf_counter(),
            "normalized_clip_fractions": [],
            "action_clipped_channels": 0,
        }

    with Client(str(args.socket), family="AF_UNIX", authkey=args.authkey.read_bytes()) as connection:
        try:
            while pending or active:
                while pending and len(active) < BATCH_SIZE:
                    active.append(initialize(*pending.popleft()))
                rows = []
                for episode in active:
                    observation = episode["observation"]
                    rows.append(
                        {
                            "task_id": episode["task_id"],
                            "reset_id": episode["reset_id"],
                            "policy_step": episode["calls"],
                            "environment_step": episode["steps"],
                            "instruction": episode["instruction"],
                            "state": libero_state(observation).tolist(),
                            "agentview": encode_image(rotate_eval_rgb_180(observation["agentview_image"])),
                            "wrist": encode_image(rotate_eval_rgb_180(observation["robot0_eye_in_hand_image"])),
                            "inference_seed": libero_replan_seed(
                                args.evaluation_seed,
                                "libero_spatial",
                                episode["task_id"],
                                "official",
                                episode["reset_id"],
                                None,
                                episode["calls"],
                            ),
                        }
                    )
                if batch_calls == 0:
                    warmup = []
                    for probe_rows in (rows, rows, list(reversed(rows))):
                        send(connection, {"operation": "predict", "rows": probe_rows, "record": False})
                        warmup.append(receive(connection))
                    first, repeated, reversed_batch = [np.asarray(value["actions"]) for value in warmup]
                    np.testing.assert_array_equal(first, repeated)
                    np.testing.assert_allclose(first, reversed_batch[::-1], rtol=0, atol=1e-5)
                    write_json(
                        args.output / "inference_validation.json",
                        {
                            "repeat_max_abs_error": float(np.abs(first - repeated).max()),
                            "permutation_max_abs_error": float(np.abs(first - reversed_batch[::-1]).max()),
                            "warmup_calls": 3,
                            "warmup_included_in_routing": False,
                        },
                    )
                send(connection, {"operation": "predict", "rows": rows})
                response = receive(connection)
                if "actions" not in response:
                    raise RuntimeError(f"policy prediction failed: {response}")
                if batch_calls == 0:
                    np.testing.assert_array_equal(first, np.asarray(response["actions"]))
                batch_calls += 1
                batch_seconds.append(response["seconds"])
                remaining = []
                for episode, actions, fraction in zip(
                    active, response["actions"], response["clip_fraction"], strict=True
                ):
                    episode["calls"] += 1
                    episode["normalized_clip_fractions"].append(fraction)
                    for raw_action in actions[: args.execution_horizon]:
                        if episode["success"] or episode["steps"] >= 220:
                            break
                        action, clipped = libero_env_action(np.asarray(raw_action))
                        episode["action_clipped_channels"] += clipped
                        episode["observation"], _, _, _ = episode["environment"].step(action)
                        episode["steps"] += 1
                        episode["success"] = bool(episode["environment"].check_success())
                    if episode["success"] or episode["steps"] >= 220:
                        episode["environment"].close()
                        result = {
                            key: episode[key]
                            for key in (
                                "task_id",
                                "reset_id",
                                "task_name",
                                "steps",
                                "calls",
                                "success",
                                "action_clipped_channels",
                            )
                        }
                        result.update(
                            {
                                "execution_horizon": args.execution_horizon,
                                "nfe": args.nfe,
                                "elapsed_seconds": time.perf_counter() - episode["started"],
                                "normalized_clip_fraction": float(np.mean(episode["normalized_clip_fractions"])),
                            }
                        )
                        completed.append(result)
                        with (args.output / "episodes.jsonl").open("a") as log:
                            log.write(json.dumps(result) + "\n")
                        print(json.dumps({"completed": len(completed), "total": len(matrix), **result}), flush=True)
                    else:
                        remaining.append(episode)
                active = remaining
                if batch_calls % 10 == 0 or not active:
                    send(connection, {"operation": "routing"})
                    receive(connection)
                successes = sum(episode["success"] for episode in completed)
                write_json(
                    args.output / "summary.json",
                    {
                        "status": "complete" if len(completed) == len(matrix) else "running",
                        "completed": len(completed),
                        "target": len(matrix),
                        "successes": successes,
                        "success_rate": successes / len(completed) if completed else None,
                        "wilson95": wilson95(successes, len(completed)),
                        "execution_horizon": args.execution_horizon,
                        "nfe": args.nfe,
                        "elapsed_seconds": time.perf_counter() - started,
                        "batch_calls": batch_calls,
                        "mean_instrumented_batch_seconds": float(np.mean(batch_seconds)),
                        "tasks": [
                            {
                                "task_id": task_id,
                                "completed": sum(x["task_id"] == task_id for x in completed),
                                "successes": sum(x["success"] for x in completed if x["task_id"] == task_id),
                            }
                            for task_id in range(10)
                        ],
                    },
                )
            send(connection, {"operation": "shutdown"})
            receive(connection)
        finally:
            for episode in active:
                episode["environment"].close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("server", "evaluate"))
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--socket", type=Path, required=True)
    parser.add_argument("--authkey", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--nfe", type=int, default=2)
    parser.add_argument("--execution-horizon", type=int, choices=(4, 8), default=4)
    parser.add_argument("--evaluation-seed", type=int, default=0)
    parser.add_argument("--resets-per-task", type=int, default=10)
    parser.add_argument("--trace-routing", action="store_true")
    args = parser.parse_args()
    if args.nfe <= 0 or not 1 <= args.resets_per_task <= 50:
        parser.error("invalid NFE or reset count")
    if args.mode == "server":
        if args.checkpoint is None:
            parser.error("server requires --checkpoint")
        run_server(args)
    else:
        run_evaluation(args)


if __name__ == "__main__":
    main()
