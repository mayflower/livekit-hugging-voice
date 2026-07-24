"""Closed set of llama.cpp chat profiles evaluated by version 0.3."""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Literal

LLMProfileId = Literal["compat_gemma31", "gemma4_26b_a4b", "qwen3_30b_a3b_2507"]
ReasoningMode = Literal["filtered", "off"]


@dataclass(frozen=True, slots=True)
class LLMProfile:
    public_id: LLMProfileId
    model_id: str
    local_artifact_key: str
    llama_server_alias: str
    reasoning_mode: ReasoningMode
    chat_template_kwargs: MappingProxyType[str, bool]
    tool_decision_max_tokens: int
    final_reply_max_tokens: int
    tool_temperature: float
    reply_temperature: float
    readiness_probe: Literal["two_step_tool"]
    quantization: str


LLM_PROFILES = MappingProxyType(
    {
        "compat_gemma31": LLMProfile(
            public_id="compat_gemma31",
            model_id="google/gemma-4-31B-it",
            local_artifact_key="gemma-4-31B-it-Q4_0.gguf",
            llama_server_alias="gemma-4-31b",
            reasoning_mode="filtered",
            chat_template_kwargs=MappingProxyType({"enable_thinking": False}),
            tool_decision_max_tokens=96,
            final_reply_max_tokens=128,
            tool_temperature=0.0,
            reply_temperature=0.7,
            readiness_probe="two_step_tool",
            quantization="Q4_0",
        ),
        "gemma4_26b_a4b": LLMProfile(
            public_id="gemma4_26b_a4b",
            model_id="google/gemma-4-26B-A4B-it",
            local_artifact_key="gemma-4-26B-A4B-it-Q4_0.gguf",
            llama_server_alias="gemma-4-26b-a4b",
            reasoning_mode="filtered",
            chat_template_kwargs=MappingProxyType({"enable_thinking": False}),
            tool_decision_max_tokens=96,
            final_reply_max_tokens=128,
            tool_temperature=0.0,
            reply_temperature=0.7,
            readiness_probe="two_step_tool",
            quantization="Q4_0",
        ),
        "qwen3_30b_a3b_2507": LLMProfile(
            public_id="qwen3_30b_a3b_2507",
            model_id="Qwen/Qwen3-30B-A3B-Instruct-2507",
            local_artifact_key="Qwen_Qwen3-30B-A3B-Instruct-2507-Q4_K_M.gguf",
            llama_server_alias="qwen3-30b-a3b-2507",
            reasoning_mode="off",
            # Instruct-2507 is the non-thinking model. Deliberately do not pass
            # enable_thinking, which would be a substitute rather than the model contract.
            chat_template_kwargs=MappingProxyType({}),
            tool_decision_max_tokens=96,
            final_reply_max_tokens=128,
            tool_temperature=0.0,
            reply_temperature=0.7,
            readiness_probe="two_step_tool",
            quantization="Q4_K_M",
        ),
    }
)


def resolve_llm_profile(profile_id: LLMProfileId) -> LLMProfile:
    return LLM_PROFILES[profile_id]
