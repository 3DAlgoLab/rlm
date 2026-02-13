# %%
# RLM introduction
# https://towardsdatascience.com/going-beyond-the-context-window-recursive-language-models-in-action/


print("Started ...")

# %%
import dspy
import warnings
from custom_interpreter import LocalPythonInterpreter

warnings.filterwarnings("ignore", category=UserWarning, module="pydantic")

from pathlib import Path
import mlflow


# local_model_id = "qwen/qwen3-coder-next"
# local_model_id = "mistralai/devstral-small-2-2512"
local_model_id = "qwen/qwen3-coder-next"
lm = dspy.LM(f"lm_studio/{local_model_id}", temperature=0.5)

mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment(f"{Path(__file__).stem}_{local_model_id.split('/')[-1]}")
mlflow.dspy.autolog()


# try x.ai model

dspy.configure_cache(enable_disk_cache=False, enable_memory_cache=False)
dspy.configure(lm=lm)


class ReadBigFileSignature(dspy.Signature):
    big_article_file_path: str = dspy.InputField(
        desc="Fie path for a text file. It's consolidated from several articles and big sized file. You need to consider this to read or to analyze it."
    )
    question = dspy.InputField()
    answer = dspy.OutputField()


# with open(
#     "data/articles.md",
# ) as f:
#     articles = f.read()

rlm = dspy.RLM(
    ReadBigFileSignature, max_iterations=50, interpreter=LocalPythonInterpreter()
)
output = rlm(
    big_article_file_path="data/articles.md",
    question="What were the main AI trends of 2025 based on provided articles? Pay attention to the content not only the titles.",
)

# %%
print("----- OUTPUT -----")
print(output.answer)
# print("\n\n".join(output.main_trends))

# %%
len(output.trajectory)
# %%
# Examine the full trajectory - this shows all REPL interactions
for i, step in enumerate(output.trajectory):
    print(f"\n{'=' * 60}")
    print(f"STEP {i + 1}")
    print(f"{'=' * 60}")
    print(f"\n📝 REASONING:\n{step['reasoning']}")
    print(f"\n💻 CODE:\n{step['code']}")
    print(
        f"\n📤 OUTPUT:\n{step['output'][:1000]}{'...' if len(step['output']) > 1000 else ''}"
    )
# %%
