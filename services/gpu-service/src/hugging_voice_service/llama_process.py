"""Controlled lifecycle for the single loopback-only llama-server child."""

from __future__ import annotations

import asyncio
import json
import logging
import os
from enum import StrEnum
from pathlib import Path
from typing import Any

import aiohttp

from .llm_profiles import LLMProfile, resolve_llm_profile

LLAMA_CPP_COMMIT = "3ce7da2c852c538c4c5f9806da27029cf8c9cc4a"

logger = logging.getLogger(__name__)


class LlamaProcessState(StrEnum):
    STOPPED = "stopped"
    STARTING = "starting"
    READY = "ready"
    STOPPING = "stopping"
    FAILED = "failed"


class LlamaProcessError(RuntimeError):
    pass


class LlamaProcess:
    """Own one llama-server process; shell execution and remote models are impossible."""

    host = "127.0.0.1"

    def __init__(
        self,
        *,
        binary: Path,
        model: Path,
        profile: LLMProfile | None = None,
        port: int = 8081,
        parallel_slots: int = 2,
        context_size: int = 32_768,
        flash_attention: str = "auto",
        continuous_batching: bool = True,
        batch_size: int = 2_048,
        ubatch_size: int = 512,
        cache_type_k: str = "f16",
        cache_type_v: str = "f16",
        cache_reuse: int = 0,
        metrics: bool = True,
        startup_timeout: float = 600.0,
        shutdown_timeout: float = 15.0,
    ) -> None:
        if not 1 <= parallel_slots <= 64:
            raise ValueError("parallel_slots must be between 1 and 64")
        if context_size < parallel_slots * 2_048:
            raise ValueError("context_size must provide at least 2048 tokens per slot")
        if flash_attention not in {"auto", "on"}:
            raise ValueError("flash_attention must be 'auto' or 'on'")
        if not continuous_batching:
            raise ValueError("continuous batching is required")
        if not 32 <= ubatch_size <= batch_size <= 4_096:
            raise ValueError("batch sizes must satisfy 32 <= ubatch_size <= batch_size <= 4096")
        if cache_type_k not in {"f16", "q8_0"} or cache_type_v not in {"f16", "q8_0"}:
            raise ValueError("KV cache types must be 'f16' or 'q8_0'")
        if not 0 <= cache_reuse <= 2_048:
            raise ValueError("cache_reuse must be between 0 and 2048")
        if not metrics:
            raise ValueError("the llama.cpp metrics endpoint is required")
        self.binary = binary
        self.model = model
        self.profile = profile or resolve_llm_profile("compat_gemma31")
        self.port = port
        self.parallel_slots = parallel_slots
        self.context_size = context_size
        self.flash_attention = flash_attention
        self.batch_size = batch_size
        self.ubatch_size = ubatch_size
        self.cache_type_k = cache_type_k
        self.cache_type_v = cache_type_v
        self.cache_reuse = cache_reuse
        self.startup_timeout = startup_timeout
        self.shutdown_timeout = shutdown_timeout
        self.state = LlamaProcessState.STOPPED
        self.failure: str | None = None
        self.failure_event = asyncio.Event()
        self._process: asyncio.subprocess.Process | None = None
        self._monitor_task: asyncio.Task[None] | None = None
        self._log_tasks: list[asyncio.Task[None]] = []

    @property
    def base_url(self) -> str:
        return f"http://{self.host}:{self.port}"

    @property
    def command(self) -> tuple[str, ...]:
        return (
            str(self.binary),
            "--model",
            str(self.model),
            "--host",
            self.host,
            "--port",
            str(self.port),
            "--parallel",
            str(self.parallel_slots),
            "--ctx-size",
            str(self.context_size),
            "--cont-batching",
            "--flash-attn",
            self.flash_attention,
            "--batch-size",
            str(self.batch_size),
            "--ubatch-size",
            str(self.ubatch_size),
            "--cache-type-k",
            self.cache_type_k,
            "--cache-type-v",
            self.cache_type_v,
            "--cache-reuse",
            str(self.cache_reuse),
            "--metrics",
            "--n-predict",
            "256",
            "--n-gpu-layers",
            "all",
            "--jinja",
            "--reasoning-format",
            "deepseek" if self.profile.reasoning_mode == "filtered" else "none",
            "--no-webui",
            "--no-ui-mcp-proxy",
        )

    @property
    def returncode(self) -> int | None:
        return None if self._process is None else self._process.returncode

    async def start(self) -> None:
        if self.state is not LlamaProcessState.STOPPED:
            raise LlamaProcessError(f"cannot start llama-server while {self.state}")
        if not self.binary.is_file() or not os.access(self.binary, os.X_OK):
            raise LlamaProcessError(f"llama-server is missing or not executable: {self.binary}")
        if not self.model.is_file():
            raise LlamaProcessError(f"selected LLM GGUF is missing: {self.model}")
        self.state = LlamaProcessState.STARTING
        self.failure = None
        self.failure_event.clear()
        try:
            self._process = await asyncio.create_subprocess_exec(
                *self.command,
                stdin=asyncio.subprocess.DEVNULL,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env={
                    **os.environ,
                    "HF_HUB_OFFLINE": "1",
                    "TRANSFORMERS_OFFLINE": "1",
                },
            )
            assert self._process.stdout is not None
            assert self._process.stderr is not None
            self._log_tasks = [
                asyncio.create_task(self._forward_logs(self._process.stdout, logging.INFO)),
                asyncio.create_task(self._forward_logs(self._process.stderr, logging.WARNING)),
            ]
            self._monitor_task = asyncio.create_task(self._monitor())
            await self._wait_until_ready()
            self.state = LlamaProcessState.READY
        except BaseException:
            await self.stop()
            self.state = LlamaProcessState.FAILED
            raise

    async def stop(self) -> None:
        process = self._process
        if process is None:
            if self.state is not LlamaProcessState.FAILED:
                self.state = LlamaProcessState.STOPPED
            return
        self.state = LlamaProcessState.STOPPING
        if process.returncode is None:
            process.terminate()
            try:
                await asyncio.wait_for(process.wait(), timeout=self.shutdown_timeout)
            except TimeoutError:
                process.kill()
                await process.wait()
        if self._monitor_task is not None:
            await asyncio.gather(self._monitor_task, return_exceptions=True)
        if self._log_tasks:
            await asyncio.gather(*self._log_tasks, return_exceptions=True)
        self._process = None
        self._monitor_task = None
        self._log_tasks.clear()
        self.state = LlamaProcessState.STOPPED

    async def metrics(self) -> bytes:
        """Return the pinned server's raw Prometheus exposition."""

        timeout = aiohttp.ClientTimeout(total=2.0)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.get(f"{self.base_url}/metrics") as response:
                payload = await response.read()
                if response.status != 200:
                    raise LlamaProcessError(f"llama-server metrics failed status={response.status}")
                return payload

    async def _wait_until_ready(self) -> None:
        deadline = asyncio.get_running_loop().time() + self.startup_timeout
        timeout = aiohttp.ClientTimeout(total=5.0)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            while asyncio.get_running_loop().time() < deadline:
                if self._process is None or self._process.returncode is not None:
                    raise LlamaProcessError(
                        f"llama-server exited during startup with code {self.returncode}"
                    )
                try:
                    async with session.get(f"{self.base_url}/health") as response:
                        if response.status == 200:
                            await self._probe_generation(session)
                            return
                except (aiohttp.ClientError, TimeoutError):
                    pass
                await asyncio.sleep(0.25)
        raise LlamaProcessError(f"llama-server did not become ready within {self.startup_timeout}s")

    async def _probe_generation(self, session: aiohttp.ClientSession) -> None:
        payload: dict[str, Any] = {
            "model": self.profile.llama_server_alias,
            "messages": [{"role": "user", "content": "Addiere 19 und 23."}],
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "add_numbers",
                        "description": "Add two integers.",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "a": {"type": "integer"},
                                "b": {"type": "integer"},
                            },
                            "required": ["a", "b"],
                            "additionalProperties": False,
                        },
                        "strict": True,
                    },
                }
            ],
            "tool_choice": "required",
            "parallel_tool_calls": False,
            "max_tokens": self.profile.tool_decision_max_tokens,
            "temperature": self.profile.tool_temperature,
            "stream": False,
            "cache_prompt": True,
            "id_slot": 0,
        }
        if self.profile.chat_template_kwargs:
            payload["chat_template_kwargs"] = dict(self.profile.chat_template_kwargs)
        async with session.post(f"{self.base_url}/v1/chat/completions", json=payload) as response:
            body = await response.text()
            if response.status != 200:
                raise LlamaProcessError(
                    f"llama-server readiness generation failed status={response.status}"
                )
        try:
            first = json.loads(body)
            message = first["choices"][0]["message"]
            calls = message["tool_calls"]
            if len(calls) != 1 or calls[0]["function"]["name"] != "add_numbers":
                raise ValueError("unexpected structured tool call")
            arguments = json.loads(calls[0]["function"]["arguments"])
            if arguments != {"a": 19, "b": 23}:
                raise ValueError("unexpected readiness arguments")
            call_id = calls[0]["id"]
        except (KeyError, IndexError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise LlamaProcessError("llama-server structured tool-call probe failed") from exc

        payload["messages"] = [
            payload["messages"][0],
            {"role": "assistant", "content": None, "tool_calls": calls},
            {"role": "tool", "tool_call_id": call_id, "name": "add_numbers", "content": "42"},
        ]
        payload["tool_choice"] = "none"
        async with session.post(f"{self.base_url}/v1/chat/completions", json=payload) as response:
            body = await response.text()
            if response.status != 200:
                raise LlamaProcessError(
                    f"llama-server tool-result probe failed status={response.status}"
                )
        try:
            final = json.loads(body)["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError, json.JSONDecodeError) as exc:
            raise LlamaProcessError("llama-server tool-result probe was malformed") from exc
        if "42" not in str(final):
            raise LlamaProcessError("llama-server did not use the readiness tool result")

    async def _monitor(self) -> None:
        process = self._process
        if process is None:
            return
        returncode = await process.wait()
        if self.state not in {LlamaProcessState.STOPPING, LlamaProcessState.STOPPED}:
            self.failure = f"llama-server exited unexpectedly with code {returncode}"
            self.state = LlamaProcessState.FAILED
            self.failure_event.set()
            logger.error("llama_server_exit", extra={"returncode": returncode})

    async def _forward_logs(self, stream: asyncio.StreamReader, level: int) -> None:
        while line := await stream.readline():
            # llama.cpp may include prompts or tool payloads in diagnostics.
            # Preserve stream activity and size without forwarding content.
            logger.log(
                level,
                "llama_server",
                extra={"child_message": "<redacted>", "child_message_bytes": len(line)},
            )
