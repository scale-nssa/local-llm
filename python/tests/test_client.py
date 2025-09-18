from local_llm import (
    start_server,
    get_response,
    num_tokens,
    multiple_choice_grammar,
)

MODEL_PATH = "/Users/jubernado/models/llm/qwen/qwen3-8b-q4.gguf"

def test_response():
    with start_server(
        model_path=MODEL_PATH,
        n_ctx=4096,
        n_gpu_layers=99,
        port=8080,
        verbose=False,
        log_disable=True
    ):
        answer = get_response("Give one fun llama fact.")
        assert len(answer) > 0

def test_num_tokens():
    with start_server(
        model_path=MODEL_PATH,
        n_ctx=4096,
        n_gpu_layers=99,
        port=8080,
        verbose=False,
        log_disable=True
    ):
        assert num_tokens("") < num_tokens("a l c ap qwe fpv a sd gad sd fwpp1 qsd asd g4r4 qwqwe 121asda r12134")

def test_max_tokens(max_tokens: int):
    with start_server(
        model_path=MODEL_PATH,
        n_ctx=4096,
        n_gpu_layers=99,
        port=8080,
        verbose=False,
        log_disable=True
    ):
        boilerplate_token_number = num_tokens("")
        answer = get_response("Give one fun llama fact.", max_tokens = max_tokens)
        print(boilerplate_token_number)
        print(answer)
        print(num_tokens(answer))
        assert num_tokens(answer) <= max_tokens + boilerplate_token_number

def test_system_prompt():
    with start_server(
        model_path=MODEL_PATH,
        n_ctx=4096,
        n_gpu_layers=99,
        port=8080,
        verbose=False,
        log_disable=True
    ):
        answer = get_response("Give one fun llama fact. /nothink", system_prompt = "Your final response should be in ALL LOWERCASE LETTERS.")
        print(answer)
        assert answer.lower() == answer

if __name__ == "__main__":
    test_system_prompt()