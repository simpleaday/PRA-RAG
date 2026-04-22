MULTIPLE_PROMPT = 'You are a helpful assistant, below is a query from a user and some relevant contexts. \
Answer the question given the information in those contexts. Your answer should be short and concise. \
If you cannot find the answer to the question, just say "I don\'t know". \
\n\nContexts: [context] \n\nQuery: [question] \n\nAnswer:'


RESPONSE_CHECK_PROMPT = 'You are given a question and two answers. Determine whether the two answers are semantically consistent with each other — i.e., \
whether they express the same stance, conclusion, or outcome regarding the question. If they are consistent, output "yes". \
If they are inconsistent or contradictory, output "no". Do not explain your answer. Only output "yes" or "no". \
\nQuestion: [question]  \n Answer 1: [answer_one]. \n Answer 2: [answer_two]'

TEXT_OPTIMIZE_PROMPT = 'You are a professional text restorer. Your task is to rewrite semantically corrupted or noisy text into a fluent and meaningful English sentence or paragraph, while preserving the original information. Here is the corrupted text:[text]. Please restore it into a clean, coherent sentence or paragraph.'


def wrap_prompt(question, context, prompt_id=1) -> str:
    if prompt_id == 4:
        assert type(context) == list
        context_str = "\n".join(context)
        input_prompt = MULTIPLE_PROMPT.replace('[question]', question).replace('[context]', context_str)
    else:
        input_prompt = MULTIPLE_PROMPT.replace('[question]', question).replace('[context]', context)
    return input_prompt

def response_check_prompt(question, answer_one, answer_two) -> str:
    response_check_prompt = RESPONSE_CHECK_PROMPT.replace('[question]', question).replace('[answer_one]', answer_one).replace('[answer_two]', answer_two)
    return response_check_prompt

def text_optimize_prompt(context) -> str:
    text_optimize_prompt = TEXT_OPTIMIZE_PROMPT.replace('[text]', context)
    return text_optimize_prompt