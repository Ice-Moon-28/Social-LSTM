
from openai import OpenAI
from sklearn.metrics import roc_auc_score

from model.model_choices import ModelChoices
from util.transformIds import transformIdIntoSentence
from util.load_data import load_data
import constants
import random

def get_answer_from_gpt_4o_mini(prompt):
    client = OpenAI(
        api_key=""
    )

    completion = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {
                "role": "user",
                "content": prompt,
            }
        ]
    )

    return completion

def get_answer_from_gpt_4o(prompt):
    client = OpenAI(
        api_key=""
    )

    completion = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {
                "role": "user",
                "content": prompt,
            }
        ]
    )

    return completion



# print(completion.choices[0].message)

def generate_classification_prompt(task_description, examples, input_text):
    """
    生成分类任务的 Prompt 模板。

    参数:
        task_description (str): 任务描述。
        examples (list): 示例列表，每个示例是一个字典，包含 'input' 和 'output'。
        input_text (str): 要分类的输入文本。

    返回:
        str: 生成的 Prompt。
    """
    prompt = f"Task: {task_description}\n\nExamples:\n"
    for idx, example in enumerate(examples, start=1):
        prompt += f"{idx}. Title: {example['title']}\n  Sentence: \"{example['input']}\"\n   Category: {example['output']}\n\n"
    
    prompt += f"Now, given the following input:\n"
    prompt += f"Sentence: \"{input_text}\"\nCategory:"
    return prompt

def load_examples(max_len=512):
    thread_to_sub = {}

    # this is cross_link in the post
    with open(constants.POST_INFO) as fp:
        for line in fp:
            info = line.split()
            source_sub = info[0]
            target_sub = info[1]
            source_post = info[2].split("T")[0].strip()
            target_post = info[6].split("T")[0].strip()
            thread_to_sub[source_post] = source_sub
            thread_to_sub[target_post] = target_sub
    
    label_map = {}
    source_to_dest_sub = {}

    # this is the crosslink's attribution
    with open(constants.LABEL_INFO) as fp:
        for line in fp:
            info = line.split("\t")
            source = info[0].split(",")[0].split("\'")[1]
            dest = info[0].split(",")[1].split("\'")[1]
            label_map[source] = 1 if info[1].strip() == "burst" else 0
            try:
                source_to_dest_sub[source] = thread_to_sub[dest]
            except KeyError:
                continue

    examples = []

        
    # source_sub, dest_sub, user, time, title, body
    with open(constants.PREPROCESSED_DATA) as fp:
        words, users, subreddits, lengths, labels, ids = [], [], [], [], [], []
        for i, line in enumerate(fp):
            info = line.split("\t")

            if info[1] in label_map and info[1] in source_to_dest_sub:
                title_words = info[-2].split(":")[1].strip().split(",")
                title_words = title_words[:min(len(title_words), max_len)]
                title_words = transformIdIntoSentence(title_words)
                if len(title_words) == 0 or title_words[0] == '':
                    continue
                
                body_words = info[-1].split(":")[1].strip().split(",")
                body_words = transformIdIntoSentence(body_words)
             
                label = label_map[info[1]]

                examples.append({
                    "title": title_words,
                    "input": body_words,
                    "output": 'positive' if label == 0 else 'negative'
                })
    with open("examples.txt", "w") as fp:
        for example in examples:
            fp.write(" ".join([example['title'], example['input'], example['output']]) + "\n")

    return examples

def transfrom_label_to_number(label):
    if 'positive' in label or 'Positive' in label:
        return 0
    elif 'negative' in label or 'Negative' in label:
        return 1
    else:
        return 2

def prompt_gpt(max_len=1024, choice=ModelChoices.GPT_4o_mini):

    examples = load_examples(max_len=max_len)

    gold_labels = []

    predictions = []

    real_predictions = []

    # random generate valid examples
    random_examples = random.sample(examples, 100)

    print(random_examples)

    import pdb; pdb.set_trace()

    for example in random_examples:
        prompt = generate_classification_prompt(
            task_description="Classify the sentiment of a given post into one of two categories: Positive or Negative. Neutral is not a valid category. When you want to predict neutral, please predict positive.",
            examples=[
                examples[11],
                examples[131],
                examples[133],
            ],
            input_text=example['input'],
        )

        gold_labels.append(transfrom_label_to_number(example['output']))

        if choice == ModelChoices.GPT_4o_mini:
            content = get_answer_from_gpt_4o_mini(prompt).choices[0].message.content
        elif choice == ModelChoices.GPT_4o:
            content = get_answer_from_gpt_4o(prompt).choices[0].message.content

        predictions.append(
            transfrom_label_to_number(
                content,
            )
        )

        real_predictions.append(
            content,
        )

    import pdb; pdb.set_trace()

    score = roc_auc_score(gold_labels, predictions)

    print("Val AUC", score)

    

def random_guess():
    examples = load_examples()

    gold_labels = []

    predictions = []

    for example in examples:

        gold_labels.append(
            transfrom_label_to_number(example['output'])
        )

        predictions.append(
            random.randint(0, 1)
        )

    score = roc_auc_score(gold_labels, predictions)

    print("Val AUC", score)



def all_positive():
    examples = load_examples()

    gold_labels = []

    predictions = []

    for example in examples:

        gold_labels.append(
            transfrom_label_to_number(example['output'])
        )

        predictions.append(
            0
        )

    score = roc_auc_score(gold_labels, predictions)

    print("Val AUC", score)
