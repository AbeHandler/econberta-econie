import pandas as pd
import numpy as np
from jinja2 import Template
from os.path import join
from tenacity import retry, wait_exponential
from tqdm import tqdm
import logging

logging.basicConfig(level=logging.INFO)
logging.getLogger("httpx").setLevel(logging.WARNING)


def get_examples_from_annotation(df, n=1):
    """
    Generate a list of examples from a DataFrame.

    :param df: A DataFrame where each row represents a sentence and its entities.
    :param n: The number of examples to generate for each entity type.
    :return: A list of dictionaries where each dictionary represents an example.
    """

    texts = []
    final_dicts = []

    # Get example without entity
    count_columns = [col for col in df.columns if col.startswith("count_")]
    subset = df.loc[df[count_columns].sum(axis=1) == 0].copy()
    texts += subset["sentence"].sample(n=n).tolist()
    final_dicts += [{} for _ in range(n)]

    variables = [col.split("count_")[1] for col in count_columns]

    for variable in variables:
        subset = df.loc[df[f"count_{variable}"] > 0].sample(n=n).copy()
        texts += subset["sentence"].tolist()
        final_dicts += [get_final_dict(row, variables) for _, row in subset.iterrows()]

    examples = list(
        np.random.permutation(
            [
                {"text": text, "answer": answer}
                for text, answer in zip(texts, final_dicts)
            ]
        )
    )

    return examples


def get_final_dict(row, variables):
    """
    Generate a dictionary from a DataFrame row.

    :param row: A DataFrame row.
    :param variables: A list of variable names.
    :return: A dictionary where the keys are the variable names and the values are the corresponding values in the row.
    """

    return {
        _variable.capitalize(): row[_variable]
        for _variable in variables
        if pd.notnull(row[_variable])
    }


def load_prompts(
    path_data,
    instruction_prompt,
    examples_prompt,
    inference_prompt,
    load_examples=False,
):
    """
    Load prompts from files.

    :param path_data: The path to the directory containing the prompt files.
    :param instruction_prompt: The name of the instruction prompt file.
    :param examples_prompt: The name of the examples prompt file.
    :param inference_prompt: The name of the inference prompt file.
    :param load_examples: A boolean indicating whether to load examples.
    :return: A tuple containing the instruction prompt, examples template, and inference prompt.
    """

    with open(join(path_data, instruction_prompt), "r") as f:
        instruction_prompt = f.read()

    examples_template = None
    if load_examples:
        with open(join(path_data, examples_prompt), "r") as f:
            examples_template = f.read()
            examples_template = Template(examples_template)

    with open(join(path_data, inference_prompt), "r") as f:
        inference_prompt = f.read()
        inference_prompt = Template(inference_prompt)

    return instruction_prompt, examples_template, inference_prompt


def get_prompt_message(
    sentence, instruction_prompt, prompt, example_template, examples=None
):
    """
    Generate a list of prompt messages.

    :param sentence: The sentence to analyze.
    :param instruction_prompt: The instruction prompt.
    :param prompt: The prompt template.
    :param example_template: The example template.
    :param examples: A list of examples.
    :return: A list of dictionaries where each dictionary represents a message.
    """

    if examples is None:
        examples = []

    messages = [
        {
            "role": "system",
            "content": instruction_prompt
            or "You are an AI assistant helping me in analysis texts.",
        },
        *(
            {"role": "user", "content": example_template.render(example)}
            for example in examples
        ),
        {"role": "user", "content": prompt.render(text=sentence)},
    ]

    return messages


@retry(wait=wait_exponential(multiplier=1, min=2, max=6))
def api_call(messages, model, client):
    return client.chat.completions.create(
        model=model,
        messages=messages,
        stop=["\n\n"],
        max_tokens=250,
        temperature=0.0,
    )


def process_sentences(
    df,
    responses,
    tokens_info,
    client,
    INSTRUCTION_PROMPT,
    INFERENCE_PROMPT,
    EXAMPLES_TEMPLATE,
    EXAMPLES,
    model="gpt-3.5-turbo-0125",
    verbose=True,
):
    """
    Process sentences from a DataFrame and make API calls to generate responses.

    :param df: The DataFrame containing the sentences to process.
    :param responses: A list to store the generated responses.
    :param tokens_info: A dictionary to store tokens and cost information.
    :param client: The OpenAI client used to make API calls.
    :param INSTRUCTION_PROMPT: The instruction prompt.
    :param INFERENCE_PROMPT: The inference prompt.
    :param EXAMPLES_TEMPLATE: The examples template.
    :param EXAMPLES: A list of examples.
    :param model: The model to use for the API calls.
    :param verbose: A boolean indicating whether to print logging information.
    :return: A tuple containing the updated responses list and a dictionary with tokens and cost information.
    """

    start_index = len(responses)

    for i, sentence in tqdm(
        enumerate(df.sentence.values[start_index:]),
        total=len(df.sentence.values[start_index:]),
    ):
        messages = get_prompt_message(
            sentence, INSTRUCTION_PROMPT, INFERENCE_PROMPT, EXAMPLES_TEMPLATE, EXAMPLES
        )
        response = api_call(messages, model, client)
        responses.append(
            {
                "text": sentence,
                "response": response.choices[0].message.content,
                "compleption_token": response.usage.completion_tokens,
                "prompt_token": response.usage.prompt_tokens,
            }
        )
        tokens_info["total_token_prompts"] += response.usage.prompt_tokens
        tokens_info["total_prompt_price"] = (
            tokens_info["total_token_prompts"] * 0.5 / 1e6
        )
        tokens_info["total_token_completion"] += response.usage.completion_tokens
        tokens_info["total_completion_price"] = (
            tokens_info["total_completion_price"] * 1.5 / 1e6
        )
        if verbose and (i + start_index) % 200 == 0:
            logging.info(
                "Total token prompts: {}".format(tokens_info["total_token_prompts"])
            )
            logging.info(
                "Total token completion: {}".format(
                    tokens_info["total_token_completion"]
                )
            )
            logging.info(
                "Total cost completion: {:.2f}€".format(
                    tokens_info["total_completion_price"]
                )
            )
            logging.info(
                "Total cost prompt: {:.2f}€".format(tokens_info["total_prompt_price"])
            )
            logging.info(
                "Total cost: {:.2f}€".format(
                    tokens_info["total_completion_price"]
                    + tokens_info["total_prompt_price"]
                )
            )

    return responses, tokens_info


def log_prompts(INSTRUCTION_PROMPT, EXAMPLES_TEMPLATE, EXAMPLES, INFERENCE_PROMPT):

    logging.info("INSTRUCTION")
    logging.info(INSTRUCTION_PROMPT)
    logging.info("-------")

    if EXAMPLES_TEMPLATE:
        logging.info(EXAMPLES_TEMPLATE.render(EXAMPLES[1]))
        logging.info("-------")
    logging.info(INFERENCE_PROMPT.render(EXAMPLES[1]))
