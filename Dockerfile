FROM runpod/base:0.6.2-cuda12.4.1
# System deps
RUN apt-get update && apt-get install -y git && rm -rf /var/lib/apt/lists/*
# Python deps
RUN python3.11 -m pip install --no-cache-dir \
    vllm==0.7.3 \
    runpod \
    transformers==4.49.0 \
    accelerate \
    huggingface_hub \
    hf_transfer \
    openai \
    httpx
# Download model at BUILD time
ARG HF_TOKEN
RUN python3.11 -c "from huggingface_hub import snapshot_download; snapshot_download('Qwen/Qwen2.5-7B-Instruct', local_dir='/model', token='${HF_TOKEN}')"
# Patch conflicting rope_scaling config
RUN echo ">Config Before Patch"
RUN cat /model/config.json
RUN python3.11 -c "import json; path='/model/config.json'; cfg=json.load(open(path)); fix=lambda r: {k:v for k,v in r.items() if k!='type'} if isinstance(r,dict) and 'rope_type' in r and 'type' in r else r; cfg.update({k: fix(v) for k,v in cfg.items() if k=='rope_scaling'}); cfg.get('text_config',{}).update({k: fix(v) for k,v in cfg.get('text_config',{}).items() if k=='rope_scaling'}); json.dump(cfg,open(path,'w'),indent=2)"
RUN echo ">Config After Patch"
RUN cat /model/config.json
# Copy handler
COPY handler.py /handler.py
CMD ["python3.11", "-u", "/handler.py"]
