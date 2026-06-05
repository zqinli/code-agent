# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""HF actor-backed AgentLoopManager for code-agent rollout.

This manager keeps the existing AgentLoop/DataProto postprocessing but replaces
the OpenAI-compatible rollout server client with direct actor HF generation.
"""

import asyncio
import logging
import os
from typing import Any

import numpy as np

from verl.experimental.agent_loop.agent_loop import AgentLoopWorker
from verl.protocol import DataProto
from verl.utils.ray_utils import auto_await
from verl.workers.rollout.replica import TokenOutput

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


class HFActorLLMClient:
    """Small LLM client facade that calls the actor worker group's HF generate RPC."""

    def __init__(self, actor_rollout_wg, rollout_config, tokenizer):
        self.actor_rollout_wg = actor_rollout_wg
        self.rollout_config = rollout_config
        self.tokenizer = tokenizer
        self._semaphore = asyncio.Semaphore(1)

    async def generate(
        self,
        request_id,
        *,
        prompt_ids: list[int],
        sampling_params: dict[str, Any],
        image_data=None,
        video_data=None,
        **kwargs,
    ) -> TokenOutput:
        del request_id, image_data, video_data, kwargs

        hf_sampling_params = dict(sampling_params or {})
        hf_sampling_params.setdefault("max_new_tokens", int(self.rollout_config.response_length))
        hf_sampling_params.setdefault("do_sample", bool(self.rollout_config.get("do_sample", True)))
        hf_sampling_params.setdefault("eos_token_id", getattr(self.tokenizer, "eos_token_id", None))
        hf_sampling_params.setdefault("pad_token_id", getattr(self.tokenizer, "pad_token_id", None))

        loop = asyncio.get_running_loop()
        async with self._semaphore:
            outputs = await loop.run_in_executor(
                None,
                lambda: self.actor_rollout_wg.hf_generate(
                    prompt_ids=[int(token_id) for token_id in prompt_ids],
                    sampling_params=hf_sampling_params,
                ),
            )

        output = self._select_output(outputs)
        return TokenOutput(
            token_ids=list(output.get("token_ids", [])),
            log_probs=output.get("log_probs"),
            routed_experts=output.get("routed_experts"),
            stop_reason=output.get("stop_reason"),
            num_preempted=output.get("num_preempted", 0),
            extra_fields=output.get("extra_fields", {}),
        )

    @staticmethod
    def _select_output(outputs):
        if isinstance(outputs, dict):
            return outputs
        if not isinstance(outputs, list):
            raise RuntimeError(f"Unknown HF actor generation output format: {type(outputs)}")
        for output in outputs:
            if isinstance(output, dict) and "token_ids" in output:
                return output
        raise RuntimeError(f"Could not find token_ids in HF actor generation outputs: {outputs!r}")


class HFAgentLoopManager:
    """Agent loop manager that runs code-agent against the actor HF policy."""

    def __init__(
        self,
        config,
        actor_rollout_wg,
        tokenizer,
        processor=None,
        reward_loop_worker_handles=None,
    ):
        self.config = config
        self.rollout_config = config.actor_rollout_ref.rollout
        self.tokenizer = tokenizer
        self.processor = processor
        self.llm_client = HFActorLLMClient(actor_rollout_wg, self.rollout_config, tokenizer)
        self.worker = AgentLoopWorker(
            config=config,
            llm_client=self.llm_client,
            teacher_client=None,
            reward_loop_worker_handles=reward_loop_worker_handles,
        )
        self.worker.tokenizer = tokenizer
        self.worker.processor = processor

    @classmethod
    @auto_await
    async def create(cls, *args, **kwargs):
        return cls(*args, **kwargs)

    @auto_await
    async def generate_sequences(self, prompts: DataProto) -> DataProto:
        output = await self.worker.generate_sequences(prompts)
        if "multi_modal_inputs" not in output.non_tensor_batch:
            empty_multi_modal_inputs = np.empty(len(output), dtype=object)
            empty_multi_modal_inputs[:] = [{} for _ in range(len(output))]
            output.non_tensor_batch["multi_modal_inputs"] = empty_multi_modal_inputs
        return output

    def start_profile(self):
        logger.debug("hf_agent rollout does not use an external rollout profiler")

    def stop_profile(self):
        logger.debug("hf_agent rollout does not use an external rollout profiler")
