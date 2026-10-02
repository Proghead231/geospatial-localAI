# ScientificAssistant

A lightweight, local Python workflow for context-aware code review, paper analysis, and scientific problem-solving. Built to interface with **Ollama** and specifically optimized for **DeepSeek-R1** reasoning models running on mid-tier local hardware (e.g., RTX 3060 Laptop / 6 GB VRAM / 32 GB System RAM).

---

## ✨ Features

* **DeepSeek-R1 Native Support**: Direct handling of DeepSeek-R1's `<think>` reasoning tags, formatting thought processes and final answers distinctly in Jupyter Notebooks.
* **Multi-Format Ingestion**: Load and cross-reference `.py`, `.ipynb`, `.pdf`, and `.txt` files directly into conversation context.
* **Context Preservation Architecture**: Anchors system instructions and initial documents to avoid context bloat while supporting multi-turn dialogue.
* **Sliding Window Memory**: Automatically manages context growth by pruning older messages while archiving pre-trim chat history state atomically.
* **Jupyter Live Streaming**: Real-time rendering of streaming markdown output directly within interactive notebook outputs.
* **Hardware-Calibrated Controls**: Pre-configured parameter lookup table for optimizing `num_ctx`, `num_predict`, and `num_batch` settings on consumer GPUs.

---

## 📋 Requirements

* **Python**: 3.9 or higher
* **Ollama**: Running locally with your chosen model pulled (e.g., `ollama pull deepseek-r1:14b`)

### Python Packages

```bash
pip install ollama pypdf IPython
```

---

## 🚀 Quick Start

### 1. Basic Initialization

```python
from scientific_assistant import ScientificAssistant

# Initialize assistant with your local Ollama model
ai = ScientificAssistant(model_name="deepseek-r1:14b")
```

### 2. Asking a Quick Question

```python
ai.chat(
    prompt="What is Object-Based Image Analysis (OBIA)?",
    num_ctx=8192,
    num_predict=512
)
```

### 3. Reviewing Code or Notebooks

```python
# Pass single or multiple files to analyze
ai.chat(
    prompt="Inspect this script for bugs, CRS mismatches, and edge cases.",
    file_names=["process_raster.py"],
    num_ctx=32768,
    num_predict=4096
)
```

### 4. Cross-Referencing Multiple Documents

```python
ai.chat(
    prompt="Compare the methodologies described in these documents.",
    folder_path="D:/research/papers",
    file_names=["paper1.pdf", "paper2.pdf"],
    num_ctx=49152,
    num_predict=6144
)
```

---

## ⚙️ Hardware & Parameter Tuning Guide

When running large reasoning models locally, performance depends on balancing four main parameters (`num_ctx`, `num_predict`, `num_batch`, `temperature`).

| Parameter | Purpose | Recommendation |
| :--- | :--- | :--- |
| **`num_ctx`** | Context window size in tokens. Holds prompt, documents, thinking tags, and answer. | Scale based on file sizes. Stay below system RAM limits to prevent disk swap. |
| **`num_predict`** | Maximum tokens generated per turn (thinking + final response combined). | Raise if responses get cut off mid-sentence. |
| **`num_batch`** | Prompt ingestion batch size. | Keep at **512** on 6 GB VRAM GPUs to prevent VRAM allocation errors. |
| **`temperature`** | Randomness of model outputs. | Use **0.5** for math/code/logic; **0.6** for general analysis; **0.7–0.8** for creative tasks. |

---

## 📊 Recommended Task Benchmarks

Optimized specifically for **RTX 3060 Laptop (6 GB VRAM) + 32 GB RAM** running `deepseek-r1:14b`:

| Task | `num_ctx` | `num_predict` | `num_batch` | `temperature` |
| :--- | :---: | :---: | :---: | :---: |
| **Quick Chat / Fact Check** | 8,192 | 512 | 512 | 0.6 |
| **Simple Code (< 200 lines)** | 16,384 | 2,048 | 512 | 0.6 |
| **Moderate Script (~500 lines)** | 32,768 | 4,096 | 512 | 0.6 |
| **Large File / Complex Code** | 49,152 | 6,144 | 512 | 0.6 |
| **Single Scientific Paper** | 32,768 | 5,120 | 512 | 0.6 |
| **Multi-File Cross-Reference (2–3 files)** | 49,152 | 6,144 | 512 | 0.6 |
| **Deep Analysis (3–4 files)** | 65,536 | 8,192 | 512 | 0.6 |
| **Full Repository / Heavy Review (5+ files)** | 98,304 | 12,288 | 512 | 0.6 |
| **Math / Derivations / Formal Proofs** | 32,768 | 8,192 | 512 | 0.5 |

---

## 📁 Memory & State Management

* **Auto-saving**: Conversations and ingested documents automatically persist state to `outputs/chat_history/chat_history.json`.
* **Archiving**: Whenever the sliding window prunes old dialogue turns (>20 messages), a timestamped backup is saved automatically to preserve history.
* **Memory Management**:
  ```python
  # List all currently loaded documents
  ai.list_docs()

  # Remove a specific document from working memory
  ai.remove_doc("old_script.py")

  # Fully clear memory and conversation history
  ai.clear_memory()
  ```

---

## 💡 Troubleshooting & Best Practices

1. **Answer Cuts Off Mid-Sentence**: Increase `num_predict`. For DeepSeek-R1, reasoning (`<think>`) consumes generation quota.
2. **Model Claims It Can't See Files**: Increase `num_ctx` to fit the total character volume of loaded documents. (Approximate rule: $1 \text{ token} \approx 4 \text{ characters}$).
3. **Unexpected Slowdowns**: Check system RAM usage. If `num_ctx` is set too high for available memory, the KV cache will spill into disk swap space. Lower `num_ctx` to resolve.

---

## 📜 License

[MIT License](LICENSE)