# edge-llm-server

A lightweight LLM inference server targeting fast, parallel, energy-aware inference on resource-constrained edge devices — specifically mid-range Android smartphones.

Built on [llama.cpp](https://github.com/ggml-org/llama.cpp), stripped to only the components relevant for edge deployment, and extended with scheduling and policy work for Aspects 5 and 6 of the research goals.

---

## Why

Running a shared LLM on a smartphone is fundamentally different from running one on a server. Battery is limited, heat builds up, and multiple apps compete for the same model simultaneously. A chat app needs a response in under 500ms. A background summarizer can wait. The OS has no concept of any of this — it just throttles the hardware when things go wrong.

This project addresses that gap by adding:

- **Energy-aware inference** — monitoring thermal and battery state and adjusting inference behavior proactively before the OS is forced to intervene
- **Application-aware QoS** — prioritizing foreground interactive requests over background batch jobs, with per-app resource limits

---

## Target

| Property | Value |
|----------|-------|
| Models | 2B parameters and below (Qwen2.5-1.5B, SmolLM2-1.7B) |
| Format | GGUF (Q4_K_M or Q5_K_M) |
| Device | Mid-range Android — Samsung Galaxy A06 class (4GB RAM, Mali-G52 GPU) |
| GPU backend | Vulkan |
| CPU backend | ARM NEON |

---

## Architecture

```
App request (chat / background task)
        ↓
  Policy layer          ← reads battery, thermal state, app priority
  (Aspects 5 & 6)       → decides: admit or defer? which queue? context cap?
        ↓
  llama-server          ← HTTP server with slot-based continuous batching
  (tools/server/)       → schedules requests across parallel slots
        ↓
  llama engine          ← loads GGUF model, runs transformer inference
  (src/)
        ↓
  ggml backends         ← Vulkan (GPU) or CPU (ARM NEON)
  (ggml/src/)
        ↓
     Response
```

---

## Folder Structure

```
edge-llm-server/
│
├── src/                          # Core llama inference engine
│   ├── llama.cpp                 # Main model loading and inference loop
│   ├── llama-kv-cache.cpp        # KV cache management (Aspect 3)
│   ├── llama-batch.cpp           # Batch processing (Aspect 2)
│   ├── llama-context.cpp         # Inference context and memory allocation
│   ├── llama-model.cpp           # Model architecture and weight loading
│   ├── llama-sampler.cpp         # Token sampling strategies
│   ├── llama-vocab.cpp           # Tokenizer
│   ├── llama-quant.cpp           # Quantization utilities (Aspect 1)
│   └── models/                   # One file per supported model architecture
│       ├── qwen3.cpp             # Qwen3 family (primary target)
│       ├── llama.cpp             # Llama family
│       └── ...                   # ~100 other model architectures
│
├── ggml/
│   ├── include/                  # Public headers for all backends
│   └── src/
│       ├── ggml.c / ggml.cpp     # Core tensor operations
│       ├── ggml-quants.c         # Quantization kernels
│       ├── ggml-backend.cpp      # Backend abstraction layer
│       ├── ggml-cpu/             # CPU backend (ARM NEON, x86 SIMD)
│       │   ├── arch/arm/         # ARM NEON optimizations
│       │   └── arch/x86/         # x86 AVX2 optimizations
│       ├── ggml-vulkan/          # Vulkan GPU backend for Android
│       │   └── vulkan-shaders/   # GLSL compute shaders
│       └── ggml-opencl/          # OpenCL backend (fallback for some Android GPUs)
│
├── tools/
│   ├── server/                   # HTTP inference server — primary modification target
│   │   ├── server.cpp            # Entry point, starts HTTP server
│   │   ├── server-context.cpp    # Scheduler loop, slot management
│   │   ├── server-queue.cpp      # Request queue (priority queues go here — Aspect 6)
│   │   ├── server-task.cpp       # Request representation (priority field goes here)
│   │   ├── server-http.cpp       # HTTP routing and request parsing
│   │   ├── server-chat.cpp       # Chat template formatting
│   │   └── tests/                # Python integration tests for the server
│   ├── llama-bench/              # Performance benchmarking tool
│   ├── batched-bench/            # Parallel request benchmarking (Aspect 2)
│   ├── quantize/                 # Convert models between GGUF quant formats
│   ├── mtmd/                     # Multimodal library (required by server)
│   └── ui/                       # Web chat interface served at localhost:8080
│
├── include/
│   └── llama.h                   # Public API — llama_batch, llama_decode, llama_kv_cache
│
├── common/                       # Shared utilities
│   ├── arg.cpp                   # Command-line flag parsing
│   ├── sampling.cpp              # Sampling utilities
│   ├── chat.cpp                  # Chat template handling
│   └── log.cpp                   # Logging
│
├── tests/                        # C++ unit tests
├── scripts/                      # Build helpers and benchmark scripts
├── cmake/                        # CMake build system modules
└── vendor/                       # Third-party libraries
    ├── cpp-httplib/              # HTTP server library
    ├── nlohmann/json.hpp         # JSON parsing
    ├── stb/stb_image.h           # Image processing (for mtmd)
    └── miniaudio/miniaudio.h     # Audio processing (for mtmd)
```

---

## Build

**Prerequisites**
```bash
sudo apt update
sudo apt install build-essential cmake git -y
```

**Build (Linux / WSL)**
```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --target llama-server -j$(nproc)
```

Binary is at `build/bin/llama-server`.

---

## Run

**Download a model**
```bash
curl -L "https://huggingface.co/bartowski/Qwen2.5-1.5B-Instruct-GGUF/resolve/main/Qwen2.5-1.5B-Instruct-Q4_K_M.gguf" \
  -o Qwen2.5-1.5B-Q4_K_M.gguf
```

**Start the server**
```bash
./build/bin/llama-server \
  -m Qwen2.5-1.5B-Q4_K_M.gguf \
  --port 8080 \
  --parallel 2 \
  --ctx-size 2048
```

Open `http://localhost:8080` for the web UI, or send requests via API:

```bash
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [{"role": "user", "content": "Hello"}],
    "max_tokens": 100
  }'
```

**Run benchmarks**
```bash
./build/bin/llama-bench \
  -m Qwen2.5-1.5B-Q4_K_M.gguf \
  -p 512 -n 128
```

---

## Research Goals

| Aspect | Description | Status |
|--------|-------------|--------|
| 1 | Quantization and model selection (GGUF format benchmarking) | Baseline established |
| 2 | Parallel request scheduling and continuous batching | Integrated via llama.cpp |
| 3 | KV cache quantization and memory management | Integrated via llama.cpp |
| 4 | Heterogeneous hardware offloading (Vulkan / ARM NEON) | Integrated via llama.cpp |
| 5 | Energy-aware inference and thermal governance | **In progress** |
| 6 | Application-aware quality of service | **In progress** |

---

## Based On

[llama.cpp](https://github.com/ggml-org/llama.cpp) — MIT License
