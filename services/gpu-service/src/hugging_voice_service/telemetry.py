"""Low-cardinality service metrics owned by one application lifecycle."""

from __future__ import annotations

from prometheus_client import CollectorRegistry, Counter, Gauge, Histogram, generate_latest


class ServiceTelemetry:
    def __init__(self) -> None:
        self.registry = CollectorRegistry()
        self.ready = Gauge(
            "hugging_voice_service_ready",
            "Whether all verified local models are warm and admission may start",
            registry=self.registry,
        )
        self.model_loads = Counter(
            "hugging_voice_model_loads",
            "Concrete model/runtime constructions in this service lifecycle",
            ("model",),
            registry=self.registry,
        )
        self.lifecycle_failures = Counter(
            "hugging_voice_lifecycle_failures",
            "Service startup/runtime failures by fixed lifecycle stage",
            ("stage",),
            registry=self.registry,
        )
        self.reasoning_violations = Counter(
            "hugging_voice_reasoning_violations",
            "Reasoning fields or leading thinking blocks suppressed from visible output",
            registry=self.registry,
        )
        self.tool_call_generations = Counter(
            "hugging_voice_tool_call_generations_total",
            "Completed structured Gemma tool-call generations",
            registry=self.registry,
        )
        self.tool_call_parse_failures = Counter(
            "hugging_voice_tool_call_parse_failures_total",
            "Rejected malformed model tool calls",
            registry=self.registry,
        )
        self.tool_call_rejections = Counter(
            "hugging_voice_tool_call_rejections_total",
            "Rejected tool calls or tool results",
            registry=self.registry,
        )
        self.tool_decision_seconds = Histogram(
            "hugging_voice_tool_decision_seconds",
            "Gemma generation start to validated tool call",
            registry=self.registry,
        )
        self.tool_result_wait_seconds = Histogram(
            "hugging_voice_tool_result_wait_seconds",
            "Tool call emission to acknowledged result",
            registry=self.registry,
        )
        self.tool_result_to_first_text_seconds = Histogram(
            "hugging_voice_tool_result_to_first_text_seconds",
            "Tool result acknowledgement to first final text",
            registry=self.registry,
        )
        self.tool_result_to_first_audio_seconds = Histogram(
            "hugging_voice_tool_result_to_first_audio_seconds",
            "Tool result acknowledgement to first final audio",
            registry=self.registry,
        )
        self.tool_schema_bytes = Gauge(
            "hugging_voice_tool_schema_bytes",
            "Canonical bytes in the active session tool schemas",
            registry=self.registry,
        )
        self.sessions_active = Gauge(
            "hugging_voice_sessions_active",
            "Connected active sessions",
            registry=self.registry,
        )
        self.sessions_available = Gauge(
            "hugging_voice_sessions_available",
            "Immediately claimable session slots",
            registry=self.registry,
        )
        self.sessions_rejected = Counter(
            "hugging_voice_sessions_rejected",
            "Sessions rejected because the service is full or draining",
            registry=self.registry,
        )
        self.sessions_draining = Gauge(
            "hugging_voice_sessions_draining",
            "Slots waiting for complete worker drain",
            registry=self.registry,
        )
        self.sessions_stuck = Gauge(
            "hugging_voice_sessions_stuck",
            "Quarantined slots whose worker chain did not drain",
            registry=self.registry,
        )
        self.turns = Counter(
            "hugging_voice_turns",
            "Completed input turns",
            registry=self.registry,
        )
        self.turns_cancelled = Counter(
            "hugging_voice_turns_cancelled",
            "Cancelled response generations",
            registry=self.registry,
        )
        self.stt_queue_seconds = Histogram(
            "hugging_voice_stt_queue_seconds",
            "STT scheduler queue wait",
            registry=self.registry,
        )
        self.stt_inference_seconds = Histogram(
            "hugging_voice_stt_inference_seconds",
            "STT inference duration",
            registry=self.registry,
        )
        self.transcription_delay_seconds = Histogram(
            "hugging_voice_transcription_delay_seconds",
            "Speech stop to final transcription",
            registry=self.registry,
        )
        self.partial_jobs_submitted = Counter(
            "hugging_voice_partial_jobs_submitted_total",
            "Optional partial STT jobs submitted to the shared scheduler",
            registry=self.registry,
        )
        self.partial_jobs_dropped = Counter(
            "hugging_voice_partial_jobs_dropped_total",
            "Optional partial STT jobs dropped or superseded by the shared scheduler",
            registry=self.registry,
        )
        self.partial_audio_seconds = Counter(
            "hugging_voice_partial_audio_seconds_total",
            "Audio seconds processed by optional partial STT jobs",
            registry=self.registry,
        )
        self.final_audio_seconds = Counter(
            "hugging_voice_final_audio_seconds_total",
            "Audio seconds processed by mandatory final STT jobs",
            registry=self.registry,
        )
        self.llm_ttft_seconds = Histogram(
            "hugging_voice_llm_ttft_seconds",
            "LLM request to first visible text",
            registry=self.registry,
        )
        self.llm_prefix_prefill_seconds = Histogram(
            "hugging_voice_llm_prefix_prefill_seconds",
            "Background prefix prefill duration for one fixed session slot",
            registry=self.registry,
        )
        self.llm_prefix_prefill_tokens = Histogram(
            "hugging_voice_llm_prefix_prefill_tokens",
            "Prompt tokens reported by a successful session prefix prefill",
            registry=self.registry,
        )
        self.llm_prefix_prefill_failures = Counter(
            "hugging_voice_llm_prefix_prefill_failures_total",
            "Failed session prefix prefill requests",
            registry=self.registry,
        )
        self.llm_first_turn_wait_for_prefill_seconds = Histogram(
            "hugging_voice_llm_first_turn_wait_for_prefill_seconds",
            "First response generation wait for its existing prefix prefill future",
            registry=self.registry,
        )
        self.llm_duration_seconds = Histogram(
            "hugging_voice_llm_duration_seconds",
            "LLM streaming duration",
            registry=self.registry,
        )
        self.llm_tokens_per_second = Histogram(
            "hugging_voice_llm_tokens_per_second",
            "Service-reported visible text token throughput",
            registry=self.registry,
        )
        self.tts_queue_seconds = Histogram(
            "hugging_voice_tts_queue_seconds",
            "TTS scheduler queue wait",
            registry=self.registry,
        )
        self.tts_ttfa_seconds = Histogram(
            "hugging_voice_tts_ttfa_seconds",
            "TTS segment start to first audio",
            registry=self.registry,
        )
        self.tts_duration_seconds = Histogram(
            "hugging_voice_tts_duration_seconds",
            "TTS segment generation duration",
            registry=self.registry,
        )
        self.tts_audio_seconds = Histogram(
            "hugging_voice_tts_audio_seconds",
            "Generated TTS audio duration",
            registry=self.registry,
        )
        self.first_audio_latency_seconds = Histogram(
            "hugging_voice_first_audio_latency_seconds",
            "Speech stop to first response audio",
            registry=self.registry,
        )
        self.barge_in_stop_latency_seconds = Histogram(
            "hugging_voice_barge_in_stop_latency_seconds",
            "Barge-in detection to final stale audio suppression",
            registry=self.registry,
        )
        self.stale_chunks_dropped = Counter(
            "hugging_voice_stale_chunks_dropped",
            "Text/audio chunks rejected by generation tags",
            registry=self.registry,
        )
        self.websocket_errors = Counter(
            "hugging_voice_websocket_errors",
            "Structured WebSocket failures",
            registry=self.registry,
        )
        self.stt_jobs_active = Gauge(
            "hugging_voice_stt_jobs_active",
            "Currently executing shared STT jobs",
            registry=self.registry,
        )
        self.tts_jobs_active = Gauge(
            "hugging_voice_tts_jobs_active",
            "Currently executing shared TTS jobs",
            registry=self.registry,
        )
        self.tts_worker_busy = Gauge(
            "hugging_voice_tts_worker_busy",
            "Whether one bounded TTS worker is executing a segment",
            ("worker",),
            registry=self.registry,
        )
        self.tts_worker_jobs_total = Counter(
            "hugging_voice_tts_worker_jobs_total",
            "TTS jobs assigned to each bounded worker",
            ("worker",),
            registry=self.registry,
        )
        self.tts_job_worker = Counter(
            "hugging_voice_tts_job_worker_total",
            "Worker assignment count for TTS jobs",
            ("worker",),
            registry=self.registry,
        )
        self.tts_sessions_waiting = Gauge(
            "hugging_voice_tts_sessions_waiting",
            "Sessions with queued TTS jobs and no active segment",
            registry=self.registry,
        )
        self.tts_queue_depth = Gauge(
            "hugging_voice_tts_queue_depth",
            "Queued TTS jobs across all sessions",
            registry=self.registry,
        )
        self.tts_active_sessions = Gauge(
            "hugging_voice_tts_active_sessions",
            "Sessions currently executing one TTS segment",
            registry=self.registry,
        )
        self.tts_fairness_wait_seconds = Histogram(
            "hugging_voice_tts_fairness_wait_seconds",
            "Queue wait before a session receives a TTS worker",
            registry=self.registry,
        )
        self.llm_jobs_active = Gauge(
            "hugging_voice_llm_jobs_active",
            "Currently active Gemma streams",
            registry=self.registry,
        )
        self.llama_metrics_scrape_failures = Counter(
            "hugging_voice_llama_metrics_scrape_failures_total",
            "Failed scrapes of the loopback-only llama.cpp metrics endpoint",
            registry=self.registry,
        )
        self.gpu_memory_bytes = Gauge(
            "hugging_voice_gpu_memory_bytes",
            "Observed GPU memory use when NVML data is available",
            registry=self.registry,
        )
        self.ready.set(0)

    def render(self) -> bytes:
        return generate_latest(self.registry)
