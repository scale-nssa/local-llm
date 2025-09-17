"""Client and grammar utilities for local_llm.

Public functions:
    get_response: minimal OpenAI-compatible chat request
    grammar: load a .gbnf file into a string
    multiple_choice_grammar: generate and save a simple grammar enumerating choices
"""
from __future__ import annotations

from openai import OpenAI
import os, re
from pathlib import Path

_BASE_URL = os.getenv("LLAMA_BASE_URL", "http://127.0.0.1:8080/v1")
_API_KEY  = os.getenv("LLAMA_API_KEY", "none")  # value ignored by llama-server

################################
# respond and calculate tokens #
################################

def get_response(
    prompt: str,
    model: str = "local",
    system_prompt: str | None = None,
    max_tokens: int | None = None,
    temperature: float | None = None,
    grammar: str | None = None,
) -> str:
    """Send a chat completion request to local llama-server."""
    client = OpenAI(base_url=_BASE_URL, api_key=_API_KEY)
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})

    kwargs: dict = {}
    if max_tokens is not None:
        kwargs["max_tokens"] = max_tokens
    if temperature is not None:
        kwargs["temperature"] = temperature
    if grammar is not None:
        kwargs["extra_body"] = {"grammar": grammar}

    r = client.chat.completions.create(model=model, messages=messages, **kwargs)
    return r.choices[0].message.content.strip()

def num_tokens(
        prompt: str,
        model: str = "local"
) -> int:
    client = OpenAI(base_url=_BASE_URL, api_key=_API_KEY)
    messages = []
    messages.append({"role": "user", "content": prompt})
    r = client.chat.completions.create(
        model=model,
        messages=messages,
        max_tokens=0,       # prompt eval only
        temperature=0,      # irrelevant here, keeps it deterministic
    )
    return r.usage.prompt_tokens


#####################
# grammar functions #
#####################

def grammar(path: str | os.PathLike[str]) -> str:
    """Load a .gbnf grammar file into a string."""
    return Path(path).read_text(encoding="utf-8")

_IDENT_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")

def multiple_choice_grammar(
    choices, save_dir: str | os.PathLike[str], name: str, thinking: bool = True
) -> str:
    """Generate and save a grammar enumerating `choices`.

    Returns the grammar text; writes file into save_dir.
    """
    if not choices:
        raise ValueError("choices must be non-empty")
    if not _IDENT_RE.match(name):
        raise ValueError("name must be a valid rule identifier")

    def _esc(s: str) -> str:
        s = s.replace("\\", "\\\\").replace("\"", "\\\"")
        s = s.replace("\n", "\\n").replace("\t", "\\t")
        return s

    alts = " | ".join(f'"{_esc(str(c))}"' for c in choices)

    if thinking:
        content = (
            f"""root ::=  thinkingBlock {name}
thinkingBlock ::= thinkingStart anychar* thinkingEnd
thinkingStart ::= "<|channel|>analysis<|message|>" | "<think>"
thinkingEnd ::= "<|end|><|start|>assistant<|channel|>final<|message|>\\n" | "</think>\\n"
{name} ::= {alts}
anychar ::= [^<]
"""
        )
        filename = f"thinking_{name}.gbnf"
    else:
        content = f"""root ::=  {name}
{name} ::= {alts}
"""
        filename = f"{name}.gbnf"

    p = Path(save_dir)
    p.mkdir(parents=True, exist_ok=True)
    (p / filename).write_text(content, encoding="utf-8", newline="\n")
    return content