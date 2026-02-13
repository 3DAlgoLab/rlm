import io
import sys
import traceback
from collections.abc import Callable
from typing import Any

from dspy.primitives import CodeInterpreter, FinalOutput


class LocalPythonInterpreter(CodeInterpreter):
    """A custom interpreter that runs Python code in the current Python environment.

    Note: This interpreter is designed to be used with dspy.RLM. When used with RLM,
    the llm_query and llm_query_batched tools will be automatically injected by RLM
    during execution. Do not call these tools directly outside of RLM execution.
    """

    def __init__(self, tools: dict[str, Callable[..., str]] | None = None):
        self._tools = tools or {}
        self._namespace = {}
        self._started = False
        self._lm = None

        # Note: llm_query and llm_query_batched will be injected by dspy.RLM
        # during forward() execution. We don't initialize them here because:
        # 1. RLM always provides its own implementations that call the configured LM
        # 2. Having stubs here would be confusing since they would be overridden
        # 3. Any attempt to use these tools before RLM execution should fail clearly

    @property
    def tools(self) -> dict[str, Callable[..., str]]:
        return self._tools

    @tools.setter
    def tools(self, value: dict[str, Callable[..., str]]) -> None:
        self._tools = value

    def start(self) -> None:
        """Initialize the interpreter."""
        self._namespace = {}
        self._started = True

    def execute(self, code: str, variables: dict[str, Any] | None = None) -> Any:
        """Execute Python code in the current environment."""
        # Initialize if not started
        if not self._started:
            self.start()

        # Add variables to namespace
        if variables:
            self._namespace.update(variables)

        # Capture stdout
        old_stdout = sys.stdout
        new_stdout = io.StringIO()
        sys.stdout = new_stdout

        try:
            # Make SUBMIT and tools available in the execution context
            exec_globals = {
                "__builtins__": __builtins__,
                "SUBMIT": SUBMIT,
                **self._tools,
                **self._namespace,
            }

            # Check for SUBMIT() call pattern
            if "SUBMIT(" in code:
                # Execute the entire code block first
                try:
                    exec(code, exec_globals)

                    # Update namespace with any new variables
                    self._namespace.update(exec_globals)

                    # Find the SUBMIT call
                    submit_start = code.index("SUBMIT(")
                    submit_end = code.index(")", submit_start)

                    # Get the code between SUBMIT( and )
                    submit_code = code[submit_start + 7 : submit_end].strip()

                    # Evaluate the SUBMIT code in the namespace
                    result = eval(submit_code, {"__builtins__": __builtins__}, exec_globals)
                    return FinalOutput(result)

                except Exception as e:
                    return f"Error processing SUBMIT(): {str(e)}"
            else:
                # Execute code normally
                try:
                    exec(code, exec_globals)

                    # Update namespace with any new variables
                    self._namespace.update(exec_globals)

                    # Capture output
                    output = new_stdout.getvalue()
                    return output or None

                except Exception as e:
                    # Format error with traceback
                    error_msg = f"Error: {str(e)}\n"
                    error_msg += traceback.format_exc()
                    return error_msg
        finally:
            sys.stdout = old_stdout

    def shutdown(self) -> None:
        """Clean up resources."""
        self._namespace = {}
        self._started = False


# Define SUBMIT function for the interpreter
SUBMIT = lambda x: FinalOutput(x)  # noqa: E731

if __name__ == "__main__":
    interpreter = LocalPythonInterpreter()
    result = interpreter.execute("print('Hello from custom interpreter!')")
    print("STDOUT:", result)

    # Test SUBMIT
    result2 = interpreter.execute("x = 10\ny = 20\nz = x + y\nSUBMIT(z)")
    print("SUBMIT result:", result2)

    interpreter.shutdown()
