"""verl AgentLoop for code generation with search and sandbox execution."""

from __future__ import annotations

import asyncio
import os
import time
from pathlib import Path
from typing import Any
from uuid import uuid4

from verl.experimental.agent_loop.agent_loop import AgentLoopBase, AgentLoopMetrics, AgentLoopOutput, register
from verl.workers.rollout.replica import TokenOutput

from code_agent.protocols.actions import ParsedAction, parse_action, parse_final_action
from code_agent.tools.retrieval import run_search
from code_agent.tools.sandbox import run_code


def _to_positive_int(value: Any) -> int | None:
    try:
        value = int(value)
    except Exception:
        return None
    return value if value > 0 else None


def _value_from_obj_or_dict(obj: Any, key: str) -> Any:
    if obj is None:
        return None
    if isinstance(obj, dict):
        return obj.get(key)
    return getattr(obj, key, None)


def _normalize_messages(raw_prompt: Any) -> list[dict[str, Any]]:
    if hasattr(raw_prompt, "tolist"):
        raw_prompt = raw_prompt.tolist()
    if isinstance(raw_prompt, tuple):
        raw_prompt = list(raw_prompt)
    if isinstance(raw_prompt, list):
        return raw_prompt
    raise RuntimeError(f"Expected raw_prompt/prompt as a list of chat messages, got {type(raw_prompt)}")


@register("code_search_agent")
class CodeSearchAgentLoop(AgentLoopBase):
    """Code/Search AgentLoop for autonomous code-generation RL."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.max_turns = int(os.environ.get("CODE_AGENT_MAX_TURNS", "2"))
        self.max_obs_length = int(os.environ.get("CODE_AGENT_MAX_OBS_LENGTH", "1024"))
        self.enable_final_rollout = os.environ.get("CODE_AGENT_ENABLE_FINAL_ROLLOUT", "1") == "1"
        self.response_length = int(
            getattr(self.rollout_config, "response_length", os.environ.get("CODE_AGENT_MAX_RESPONSE_LENGTH", "2048"))
        )

    def _encode(self, text: str) -> list[int]:
        if not text:
            return []
        return self.tokenizer.encode(text, add_special_tokens=False)

    def _decode(self, ids: list[int]) -> str:
        if not ids:
            return ""
        return self.tokenizer.decode(ids, skip_special_tokens=True)

    def _pad_token_id(self) -> int:
        pad_token_id = getattr(self.tokenizer, "pad_token_id", None)
        if pad_token_id is not None:
            return int(pad_token_id)
        eos_token_id = getattr(self.tokenizer, "eos_token_id", None)
        if eos_token_id is not None:
            return int(eos_token_id)
        return 0

    def _resolve_response_length(self, sampling_params: dict[str, Any] | Any) -> int:
        base_length = None

        for key in ["response_length", "max_response_length"]:
            value = _to_positive_int(_value_from_obj_or_dict(sampling_params, key))
            if value is not None:
                base_length = value
                break

        if base_length is None:
            for key in ["response_length", "max_response_length"]:
                value = _to_positive_int(getattr(self.rollout_config, key, None))
                if value is not None:
                    base_length = value
                    break

        if base_length is None:
            value = _to_positive_int(os.environ.get("CODE_AGENT_MAX_RESPONSE_LENGTH"))
            if value is not None:
                base_length = value

        if base_length is None:
            base_length = self.response_length

        hard_caps: list[int] = []
        for key in ["max_new_tokens", "max_tokens"]:
            value = _to_positive_int(_value_from_obj_or_dict(sampling_params, key))
            if value is not None:
                hard_caps.append(value)

        if hard_caps:
            return min(base_length, min(hard_caps))
        return base_length

    def _truncate_response(
        self,
        response_ids: list[int],
        response_mask: list[int],
        max_response_length: int,
    ) -> tuple[list[int], list[int]]:
        if len(response_ids) != len(response_mask):
            min_len = min(len(response_ids), len(response_mask))
            response_ids = response_ids[:min_len]
            response_mask = response_mask[:min_len]

        response_ids = response_ids[:max_response_length]
        response_mask = response_mask[:max_response_length]

        return response_ids, response_mask

    def _truncate_by_tokens(self, text: str, max_tokens: int) -> str:
        if not text:
            return ""
        ids = self._encode(text)
        if len(ids) <= max_tokens:
            return text
        return self._decode(ids[:max_tokens]) + "\n\n[TRUNCATED: tool output exceeded limit]"

    def _append_tokens(
        self,
        response_ids: list[int],
        response_mask: list[int],
        cur_prompt_ids: list[int],
        new_ids: list[int],
        mask_value: int,
        max_response_length: int,
    ) -> bool:
        if not new_ids:
            return True
        remain = max_response_length - len(response_ids)
        if remain <= 0:
            return False
        kept = new_ids[:remain]
        response_ids.extend(kept)
        response_mask.extend([mask_value] * len(kept))
        cur_prompt_ids.extend(kept)
        return len(kept) == len(new_ids)

    async def _run_code(self, code: str) -> str:
        return await asyncio.to_thread(run_code, code)

    async def _run_search(self, query: str) -> str:
        return await asyncio.to_thread(run_search, query)

    def _workspace_root(self) -> Path | None:
        for key in ["CODE_AGENT_WORKSPACE_PATH", "CODE_AGENT_REPO_ROOT", "CODE_AGENT_REPO_DIR"]:
            value = os.environ.get(key)
            if value:
                return Path(value).expanduser().resolve()
        return None

    def _open_file_sync(self, file_path: str) -> str:
        root = self._workspace_root()
        if root is None:
            return "open_file unavailable: workspace path is not configured"
        if not root.exists() or not root.is_dir():
            return f"open_file unavailable: workspace path does not exist: {root}"

        raw_path = Path(str(file_path).strip())
        if not str(raw_path):
            return "open_file unavailable: empty path"
        target = raw_path if raw_path.is_absolute() else root / raw_path
        try:
            target = target.resolve()
        except Exception as exc:
            return f"open_file unavailable: cannot resolve path: {exc}"

        try:
            target.relative_to(root)
        except ValueError:
            return "open_file unavailable: path is outside configured workspace"
        if not target.exists():
            return f"open_file unavailable: file does not exist: {raw_path}"
        if not target.is_file():
            return f"open_file unavailable: path is not a file: {raw_path}"

        try:
            text = target.read_text(encoding="utf-8", errors="replace")
        except Exception as exc:
            return f"open_file unavailable: failed to read file: {exc}"

        rel_path = target.relative_to(root)
        numbered = [f"{i:>6} | {line}" for i, line in enumerate(text.splitlines(), 1)]
        if not numbered:
            numbered = ["     1 | "]
        return f"path: {rel_path}\n" + "\n".join(numbered)

    async def _open_file(self, file_path: str) -> str:
        return await asyncio.to_thread(self._open_file_sync, file_path)

    async def _generate_once(self, prompt_ids: list[int], sampling_params: dict[str, Any]) -> list[int]:
        output = await self.server_manager.generate(
            request_id=uuid4().hex,
            prompt_ids=prompt_ids,
            sampling_params=sampling_params,
        )
        if isinstance(output, TokenOutput):
            return list(output.token_ids)
        if isinstance(output, list):
            return list(output)
        if isinstance(output, dict):
            for key in ["token_ids", "response_ids", "output_ids", "tokens"]:
                if key in output:
                    return list(output[key])
        for attr in ["token_ids", "response_ids", "output_ids", "tokens"]:
            if hasattr(output, attr):
                return list(getattr(output, attr))
        raise RuntimeError(f"Unknown LLM output format: {type(output)}")

    def _invalid_observation(self) -> str:
        return (
            "\n\n<observation>"
            "Invalid action. Output exactly one action: "
            "<search_code>query</search_code>, "
            "<open_file>path</open_file>, "
            "<run_sandbox>command</run_sandbox>, "
            "<generate_patch>unified diff patch</generate_patch>, "
            "or <final>status</final>. "
            "Put candidate patches in <generate_patch>. Use <final> only to finish or summarize status. "
            "Do not generate <information> or <observation> yourself."
            "</observation>\n\n"
        )

    async def _handle_action(self, action: ParsedAction) -> tuple[str | None, str]:
        if action.action_type == "search_code":
            information = await self._run_search(action.content)
            return "information", information
        if action.action_type == "open_file":
            observation = await self._open_file(action.content)
            return "observation", observation
        if action.action_type == "run_sandbox":
            observation = await self._run_code(action.content)
            return "observation", observation
        if action.action_type == "generate_patch":
            return "observation", "patch received"
        return None, ""

    async def run(self, sampling_params: dict[str, Any], **kwargs) -> AgentLoopOutput:
        max_response_length = self._resolve_response_length(sampling_params)
        max_obs_length = self.max_obs_length

        raw_prompt = kwargs.get("raw_prompt", None)
        if raw_prompt is None:
            raw_prompt = kwargs.get("prompt", None)
        if raw_prompt is None:
            raise RuntimeError("CodeSearchAgentLoop requires raw_prompt or prompt. Set data.return_raw_chat=True.")
        raw_prompt = _normalize_messages(raw_prompt)

        multi_modal_data = await self.process_vision_info(raw_prompt)
        prompt_ids = await self.apply_chat_template(raw_prompt)

        cur_prompt_ids = list(prompt_ids)
        response_ids: list[int] = []
        response_mask: list[int] = []
        action_trace: list[dict[str, Any]] = []
        candidate_patch = ""

        t_generate = 0.0
        t_tool = 0.0
        num_turns = 0
        finished = False
        can_final_rollout = True

        for _ in range(self.max_turns):
            if len(response_ids) >= max_response_length:
                can_final_rollout = False
                break

            num_turns += 1
            t0 = time.time()
            gen_ids = await self._generate_once(cur_prompt_ids, sampling_params)
            t_generate += time.time() - t0

            action = parse_action(self._decode(gen_ids))
            action_ids = self._encode(action.raw_text)
            if not self._append_tokens(response_ids, response_mask, cur_prompt_ids, action_ids, 1, max_response_length):
                can_final_rollout = False
                break

            action_trace.append({"turn": num_turns, "action_type": action.action_type})

            if action.action_type == "generate_patch":
                candidate_patch = action.content
                finished = True
                break

            if action.action_type == "final":
                finished = True
                break

            if action.action_type in {"search_code", "open_file", "run_sandbox"}:
                t0 = time.time()
                env_tag, env_content = await self._handle_action(action)
                t_tool += time.time() - t0
                env_content = self._truncate_by_tokens(env_content, max_obs_length)
                env_text = f"\n\n<{env_tag}>{env_content}</{env_tag}>\n\n"
            else:
                env_text = self._invalid_observation()

            env_ids = self._encode(env_text)
            if not self._append_tokens(response_ids, response_mask, cur_prompt_ids, env_ids, 0, max_response_length):
                can_final_rollout = False
                break

        if self.enable_final_rollout and not finished and can_final_rollout and len(response_ids) < max_response_length:
            num_turns += 1
            t0 = time.time()
            gen_ids = await self._generate_once(cur_prompt_ids, sampling_params)
            t_generate += time.time() - t0

            final_action = parse_final_action(self._decode(gen_ids))
            final_ids = self._encode(final_action.raw_text)
            self._append_tokens(response_ids, response_mask, cur_prompt_ids, final_ids, 1, max_response_length)
            action_trace.append({"turn": num_turns, "action_type": final_action.action_type, "final_rollout": True})
            if final_action.action_type == "generate_patch":
                candidate_patch = final_action.content

        response_ids, response_mask = self._truncate_response(response_ids, response_mask, max_response_length)

        return AgentLoopOutput(
            prompt_ids=list(prompt_ids),
            response_ids=response_ids,
            response_mask=response_mask,
            multi_modal_data=multi_modal_data,
            num_turns=num_turns,
            metrics=AgentLoopMetrics(generate_sequences=t_generate, tool_calls=t_tool),
            extra_fields={
                "action_trace": action_trace,
                "candidate_patch": candidate_patch,
                "turn_scores": [],
                "tool_rewards": [],
            },
        )
