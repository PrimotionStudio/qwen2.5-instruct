import asyncio

# vLLM OpenAI-compatible server runs as a subprocess
import subprocess
import time

import httpx
import runpod
from openai import AsyncOpenAI

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

client = AsyncOpenAI(
    api_key="dummy",
    base_url="http://localhost:8000/v1",
)


async def handler(job):
    job_input = job["input"]

    if job_input.get("ping"):
        return {"pong": True}

    messages = job_input.get("messages", [])
    if not messages:
        return {"error": "messages array is required and cannot be empty"}
    max_tokens = job_input.get("max_tokens", 512)
    temperature = job_input.get("temperature", 0.7)

    response = await client.chat.completions.create(
        model="/model",
        messages=messages,
        max_tokens=max_tokens,
        temperature=temperature,
    )

    return {"choices": [{"message": {"content": response.choices[0].message.content}}]}


runpod.serverless.start({"handler": handler})
