import pandas as pd
from local_llm import start_server, annotate

MODEL_PATH = "/Users/jubernado/models/llm/qwen/qwen3-8b-q4.gguf"

def test_annotation():
    with start_server(
        model_path=MODEL_PATH,
        n_ctx=2048,
        n_gpu_layers=99,
        port=8080,
        verbose="Not question",
        log_disable=True
    ):
        person_row = ["Jim" if i % 2 == 0 else "Mary" for i in range(20)]
        dialogue_row = [
                "Hello",
                "Hi",
                "What are you doing today?",
                "Not too much",
                "You should come to my party.",
                "What time is the party?",
                "9pm",
                "Oh that's pretty late",
                "Yeah I don't get off work until 7 that day and I need to eat, y'know",
                "I see I see.",
                "But it's gonna be a ton of fun, you should roll through",
                "Okay okay, is it cool if I bring my buddy?",
                "Um, who are they?",
                "Ah a friend Nancy from college, she's cool I swear",
                "Haha okay okay yeah no that should be totally fine.",
                "Thanks. By the way did you hear about Madeline?",
                "Nah what happened",
                "She got into law school",
                "Woah that's super cool",
                "Yup, I guess we got a lawyer on the team now"
            ]
        dummy_df = pd.DataFrame({
            "person": person_row,
            "dialogue": dialogue_row,
        })
        new_df = annotate(
            df=dummy_df,
            schema="Your goal is to annotate whether a given utterance is a question or not. A question is any time that one person is asking something to somebody else.",
            labels=["Question", "Not question"],
            window=3,
            double_sided=True
        )
        assert new_df["label"].to_list() == [
            "Not question",
            "Not question",
            "Question",
            "Not question",
            "Not question",
            "Question",
            "Not question",
            "Not question",
            "Not question",
            "Not question",
            "Not question",
            "Question",
            "Not question",
            "Not question",
            "Question",
            "Not question",
            "Not question",
            "Not question",
            "Not question"
        ]
        for i, row in new_df.iterrows():
            print("-")*50
            print("Speaker:")
            print(row["person"])
            print("Dialogue")
            print(row["dialogue"])
            print("Annotation")
            print(row["label"])

def test_annotation2():
    with start_server(
        model_path=MODEL_PATH,
        n_ctx=2048,
        n_gpu_layers=99,
        port=8080,
        verbose="Not question",
        log_disable=True
    ):
        
        dummy_df = pd.DataFrame({
            "animal1": ["dog", "cat", "mammoth", "horse", "cow", "lizard"],
            "animal2": ["wolf", "dinosaur", "mouse", "donkey", "pig", "capybara"],
        })
        new_df = annotate(
            df=dummy_df,
            schema="Your goal is to annotate whether a given pair of animals could mate and produce an offspring or not",
            labels=["Could not mate", "Could mate"],
            window=2,
            double_sided=True
        )
        assert new_df["label"].to_list() == [
            "Could mate",
            "Could not mate",
            "Could not mate",
            "Could mate",
            "Could not mate",
            "Could not mate"
        ]
        for i, row in new_df.iterrows():
            print("-"*50)
            print(f"animals: {row["animal1"]}, {row["animal2"]}" )
            print("Annotation")
            print(row["raw_output"])
            print(row["label"])

if __name__ == "__main__":
    test_annotation2()