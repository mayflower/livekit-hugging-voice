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
        port: int = 8081,
        parallel_slots: int = 2,
        context_size: int = 32_768,
        startup_timeout: float = 600.0,
        shutdown_timeout: float = 15.0,
    ) -> None:
        if parallel_slots != 2:
            raise ValueError("Gemma requires exactly two llama.cpp sequence slots")
        self.binary = binary
        self.model = model
        self.port = port
        self.parallel_slots = parallel_slots
        self.context_size = context_size
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
            "--n-predict",
            "256",
            "--n-gpu-layers",
            "all",
            "--jinja",
            "--reasoning-format",
            "deepseek",
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
            raise LlamaProcessError(f"Gemma GGUF is missing: {self.model}")
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
            "model": "gemma-4-31b",
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
            "max_tokens": 128,
            "temperature": 0,
            "stream": False,
            "cache_prompt": True,
            "id_slot": 0,
            "chat_template_kwargs": {"enable_thinking": False},
        }
        async with session.post(f"{self.base_url}/v1/chat/completions", json=payload) as response:
            body = await response.text()
            if response.status != 200:
                raise LlamaProcessError(
                    f"llama-server readiness generation failed status={response.status} "
                    f"body={body[:512]!r}"
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
                    f"llama-server tool-result probe failed status={response.status} "
                    f"body={body[:512]!r}"
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
            logger.log(
                level,
                "llama_server",
                extra={"child_message": line.decode(errors="replace").rstrip()},
            )
