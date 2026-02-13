import os

from dotenv import load_dotenv

from rlm import RLM
from rlm.logger import RLMLogger

load_dotenv()

logger = RLMLogger(log_dir="./logs")

rlm = RLM(
    backend="litellm",  # or "portkey", etc.
    backend_kwargs={
        "model_name": "lm_studio/qwen/qwen3-coder-next",
        "api_key": os.getenv("LM_STUDIO_API_KEY"),
        "api_base": os.getenv("LM_STUDIO_API_BASE"), 
        "temperature": 0.5
    },
    environment="local",
    environment_kwargs={},
    max_depth=1,
    logger=logger,
    verbose=True,  # For printing to console with rich, disabled by default.
)

result = rlm.completion("Print me the first 5 powers of two, each on a newline.")

print(result)
