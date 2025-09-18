"""local_llm: minimal client + programmatic server launcher.

Public API:
	get_response(prompt, model="local", ...)
	grammar(path)
	multiple_choice_grammar(choices, save_dir, name, thinking=True)
	start_server(...)

The start_server function launches a llama-server process and returns a ServerHandle
object that can be used to manage its lifecycle. Intended for local research use.
"""

from .client import get_response, grammar, multiple_choice_grammar, num_tokens
from .server import start_server, ServerHandle

__all__ = [
	"get_response",
    "num_tokens",
	"grammar",
	"multiple_choice_grammar",
	"start_server",
	"ServerHandle",
]

__version__ = "0.1.1"