import pandas as pd
from .client import get_response, multiple_choice_grammar, strip_thinking

def row_str(row) -> str:
    return ", ".join([str(item) for item in row.to_list()])

def df_view(df: pd.DataFrame, target_mask: list[bool] = None) -> str:
    columns = row_str(df.columns)
    output_lines = ["[COLUMNS]", columns]
    if target_mask:
        output_lines.append("[PRE-TARGET CONTEXT ROWS]")
    for pos, (idx, row) in enumerate(df.iterrows()):
        if target_mask and target_mask[pos]:
            output_lines.append("\n [TARGET ROW]")
            output_lines.append(row_str(row))
            output_lines.append("\n [POST-TARGET CONTEXT ROWS]")
        else:
            output_lines.append(row_str(row))
    return "\n".join(output_lines)

def annotation_prompt(window_df: pd.DataFrame, schema: str, labels: list[str], target_mask: list[bool] = None) -> str:
    return f"""Your goal is to annotate a row in a dataset according to a schema.
The schema is:
{schema}

You should annotate the TARGET ROW below according to the schema above.
After your thinking process, you should output EXACTLY one of the following labels and nothing more: {str(labels)}
The row and corresponding context is given below:
{df_view(window_df, target_mask)}"""

def annotate(
        df: pd.DataFrame,
        schema: str,
        labels: list[str],
        window: int = 10,
        double_sided: bool = True,
        max_tokens: int = 1024
) -> pd.DataFrame:
    n = len(df)
    raw_annotations = []

    for pos, (idx, row) in enumerate(df.iterrows()):
        print(f"({pos+1}/{n})")
        # Generate the window around which we're going to annotate
        start = max(0, pos - window)
        stop = min(n, pos + window + 1) if double_sided else min(n, pos + 1)
        target_mask = [i == pos for i in range(start, stop)]
        window_df = df.iloc[start:stop]

        # Prepare to annotate with constrained grammar and dynamic prompt
        grammar = multiple_choice_grammar(labels, name="label", thinking=True)
        prompt = annotation_prompt(window_df, schema, labels, target_mask)

        # Now get annotation
        raw_annotations.append(get_response(prompt, max_tokens=max_tokens, grammar=grammar))
    annotated_df = df.copy()
    annotated_df["raw_output"] = raw_annotations
    annotated_df["label"] = [strip_thinking(response) for response in raw_annotations]
    return annotated_df