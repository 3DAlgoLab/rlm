# RLM (Recursive Language Models) Experiments

This folder contains experiments demonstrating **Recursive Language Models (RLM)** implementations based on the article ["Going Beyond the Context Window: Recursive Language Models in Action"](https://towardsdatascience.com/going-beyond-the-context-window-recursive-language-models-in-action/) from Towards Data Science.

## Overview

RLM enables language models to handle tasks that exceed their context window by recursively breaking down problems and using tools like code execution. This is particularly useful for:

- Processing large documents that exceed token limits
- Multi-step reasoning tasks
- Analyzing data that requires iterative exploration

## Files

### DSPy-based Implementation (Article Examples)

These files demonstrate the DSPy version of RLM as discussed in the article:

- **`dspy_rlm.py`** - Full DSPy RLM implementation with MLflow logging
  - Uses `LocalPythonInterpreter` for code execution
  - Configured for LM Studio with Qwen models
  - Includes trajectory examination for debugging
  
- **`dspy_rlm_simple.py`** - Simplified DSPy RLM example
  - Basic setup for quick testing
  - Demonstrates simple question-answering with RLM
  
- **`custom_interpreter.py`** - Custom Python interpreter for DSPy RLM
  - Executes Python code in the current environment
  - Supports `SUBMIT()` function for returning final answers
  - Integrates with DSPy's RLM framework

### Native RLM Implementation (This Repository)

- **`rlm_read_big_article.py`** - Using the native `rlm` library from this repo
  - Demonstrates the core RLM implementation without DSPy
  - Uses the `RLM` class directly from `rlm` package
  - Configured with litellm backend and local environment
  - Shows how to use the library's logger and configuration

### Test Data

- **`articles.md`** - Large consolidated articles for testing
  - Contains multiple articles about AI trends, agentic AI, and data science
  - Used as input for RLM to demonstrate reading beyond context window

## Usage

### DSPy RLM Example

```bash
# Set up your environment
export LM_STUDIO_API_KEY=your_key
export LM_STUDIO_API_BASE=http://localhost:1234/v1

# Run the DSPy RLM example
uv run python experiment/dspy_rlm.py
```

### Native RLM Example

```bash
# Configure environment variables
export LM_STUDIO_API_KEY=your_key
export LM_STUDIO_API_BASE=http://localhost:1234/v1

# Run the native RLM example
uv run python experiment/rlm_read_big_article.py
```

## Key Concepts

### How RLM Works

1. **Problem Decomposition**: When faced with a large document or complex task, RLM breaks it into smaller, manageable pieces
2. **Tool Use**: RLM can execute Python code to process data, enabling it to:
   - Read files in chunks
   - Perform calculations
   - Store intermediate results
3. **Recursive Processing**: The model iteratively processes information until the task is complete

### Context Window Limitation

Traditional LLMs are limited by their context window (typically 4K-128K tokens). RLM overcomes this by:
- Reading and processing documents in chunks
- Using code execution to maintain state
- Recursively calling itself for sub-tasks

## Implementation Comparison

| Feature | DSPy RLM | Native RLM (this repo) |
|---------|----------|------------------------|
| Framework | DSPy | Custom implementation |
| Signature | Required | Uses prompts directly |
| Interpreter | Custom Python interpreter | Local/Modal/Prime environments |
| Max Iterations | Configurable | Configurable (`max_depth`) |
| Backend | LM Studio, OpenAI, etc. | litellm, portkey, etc. |
| Logging | MLflow integration | Built-in RLMLogger |

## Requirements

Install dependencies:

```bash
# For native RLM
uv pip install -e .

# For DSPy RLM
uv pip install dspy mlflow

# For Modal support (optional)
uv pip install -e ".[modal]"
```

## References

- [Original Article](https://towardsdatascience.com/going-beyond-the-context-window-recursive-language-models-in-action/)
- [DSPy Documentation](https://dspy.ai/)
- [Project README](../README.md)
