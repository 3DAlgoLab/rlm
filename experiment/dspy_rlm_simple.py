from custom_interpreter import LocalPythonInterpreter
from dspy import RLM, Signature, InputField, OutputField
import dspy

# Configure LM
dspy.configure(lm=dspy.LM("lm_studio/qwen/qwen3-coder-next"))


class SimpleSignature(Signature):
    question = InputField()
    answer = OutputField()


rlm = RLM(
    SimpleSignature,
    max_iterations=2,  # Reduce iterations for testing
    interpreter=LocalPythonInterpreter(),
)

output = rlm(question="What is 2 + 2?")

print("Answer:", output.answer)
