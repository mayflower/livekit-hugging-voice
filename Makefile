.PHONY: sync lint format-check typecheck test coverage check models models-verify docker-build docker-up docker-down demo-agent gpu-test packages checksums benchmark-summary image-digests

MODEL_ROOT ?= .models
MODEL_MANIFEST ?= models/manifest.yaml
MODEL_LOCK ?= models/manifest.lock.json

sync:
	uv sync --all-packages --frozen

lint:
	uv run ruff check .

format-check:
	uv run ruff format --check .

typecheck:
	uv run mypy packages services examples

test:
	uv run pytest -q

coverage:
	uv run pytest -q \
		--cov=hugging_voice_protocol \
		--cov=livekit.plugins.hugging_voice \
		--cov=hugging_voice_service \
		--cov=examples/minimal-livekit-agent

check: lint format-check typecheck test

models:
	uv run hugging-voice-model-prefetch --manifest $(MODEL_MANIFEST) --output $(MODEL_ROOT) --lock $(MODEL_LOCK)

models-verify:
	HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 uv run hugging-voice-model-verify --lock $(MODEL_LOCK) --root $(MODEL_ROOT)

docker-build:
	docker compose -f deploy/docker/compose.yaml build

docker-up:
	docker compose -f deploy/docker/compose.yaml up -d

docker-down:
	docker compose -f deploy/docker/compose.yaml -f deploy/docker/compose.livekit.yaml down

demo-agent:
	docker compose -f deploy/docker/compose.yaml -f deploy/docker/compose.livekit.yaml up --build

gpu-test:
	HV_RUN_GPU_TESTS=1 uv run pytest -q -m gpu

packages:
	uv run python -m build packages/hugging-voice-protocol --outdir dist
	uv run python -m build packages/livekit-plugins-hugging-voice --outdir dist
	uv run python -m build services/gpu-service --outdir dist
	uv run python -m build examples/minimal-livekit-agent --outdir dist

checksums: packages
	cd dist && sha256sum *.whl *.tar.gz > SHA256SUMS

benchmark-summary:
	test -n "$(RAW_REPORT)"
	uv run python benchmarks/summarize.py $(RAW_REPORT)

image-digests:
	uv run python deploy/docker/resolve_image_digests.py --output dist/IMAGE_DIGESTS.json
