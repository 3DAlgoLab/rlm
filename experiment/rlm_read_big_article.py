# original src: https://towardsdatascience.com/going-beyond-the-context-window-recursive-language-models-in-action/
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
        "temperature": 0.5,
    },
    environment="local",
    environment_kwargs={},
    max_depth=1,
    logger=logger,
    verbose=True,  # For printing to console with rich, disabled by default.
)

# main_prompt = "Read from very big sized content from `./articles.md`. Then tell me what were the main AI trends of 2025 based on provided articles? Pay attention to the content not only the titles."

main_prompt = "Read the big file, `~/dev_sandbox/rlm/experiment/articles.md` Then tell me what were the main AI trends of 2025 based on provided articles? Pay attention to the content not only the titles."
result = rlm.completion(main_prompt)

print(result)
