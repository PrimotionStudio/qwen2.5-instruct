import asyncio
import subprocess
import time

import httpx
import runpod

# Start vLLM server on localhost
proc = subprocess.Popen(
    [
        "python3.11",
        "-m",
        "vllm.entrypoints.openai.api_server",
        "--model",
        "/model",
        "--dtype",
        "bfloat16",
        "--max-model-len",
        "8192",
        "--gpu-memory-utilization",
        "0.90",
        "--port",
        "8000",
        "--host",
        "0.0.0.0",
    ]
)

# Wait for server to be ready
print("Waiting for vLLM server to start...")
for _ in range(120):  # wait up to 2 minutes
    try:
        r = httpx.get("http://localhost:8000/health", timeout=2)
        if r.status_code == 200:
            print("vLLM server is ready!")
            break
    except Exception:
        pass
    time.sleep(1)
else:
    raise RuntimeError("vLLM server failed to start in time")


async def handler(job):
    job_input = job["input"]

    # Health/ping check
    if job_input.get("ping"):
        return {"pong": True}

    # RunPod OpenAI-proxy format: input contains { openai_route, openai_input }
    if "openai_route" in job_input:
        route = job_input["openai_route"]
        openai_input = job_input.get("openai_input", {})

        if openai_input.get("model") in (None, "", "default"):
            openai_input["model"] = "/model"

        async with httpx.AsyncClient(timeout=180.0) as client:
            resp = await client.post(
                f"http://localhost:8000{route}",
                json=openai_input,
            )
            resp.raise_for_status()
            return resp.json()

    # Legacy flat format: { messages, max_tokens, temperature, ... }
    messages = job_input.get("messages", [])
    if not messages:
        return {"error": "messages array is required and cannot be empty"}

    payload = {
        "model": "/model",
        "messages": messages,
        "max_tokens": job_input.get("max_tokens", 512),
        "temperature": job_input.get("temperature", 0.7),
    }

    for key in (
        "stream",
        "top_p",
        "frequency_penalty",
        "presence_penalty",
        "stop",
        "seed",
    ):
        if key in job_input:
            payload[key] = job_input[key]

    async with httpx.AsyncClient(timeout=180.0) as client:
        resp = await client.post(
            "http://localhost:8000/v1/chat/completions",
            json=payload,
        )
        resp.raise_for_status()
        return resp.json()


runpod.serverless.start({"handler": handler})
