from transformers import AutoTokenizer
from transformers import AutoModelForSeq2SeqLM


def load_model(path, device="cuda", fp16=False):
    """
    Loads the model from the specified path.

    :param path: the path load the model/tokenizer from
    :type path: str
    :param device: the device to use (cuda|cpu)
    :type device: str
    :param fp16: whether to use half-precision floats
    :type fp16: bool
    :return: the tuple of tokenizer, model
    :rtype: tuple
    """
    tokenizer = AutoTokenizer.from_pretrained(path)
    model = AutoModelForSeq2SeqLM.from_pretrained(path)
    if fp16:
        model.half()
    model.to(device)
    return tokenizer, model


def translate(tokenizer, model, prompt, text, device="cuda", max_new_tokens=40, do_sample=True,
              top_k=30, top_p=0.95):
    """
    Translates the text

    :param tokenizer: the tokenizer to use
    :param model: the model to use
    :param prompt: the prompt string to use (was used for training, e.g., 'translate English to French: ')
    :type prompt: str
    :param text: the text to translate
    :type text: str
    :param device: the device to use (cuda|cpu)
    :type device: str
    :param max_new_tokens: the maximum number of tokens
    :type max_new_tokens: int
    :param do_sample: whether to sample
    :type do_sample: bool
    :param top_k: the top X predictions
    :type top_k: int
    :param top_p: the probability
    :type top_p: float
    :return: the translation
    :rtype: str
    """
    full_prompt = prompt + " " + text
    inputs = tokenizer(full_prompt, return_tensors="pt").input_ids.to(device)
    outputs = model.generate(inputs, max_new_tokens=max_new_tokens, do_sample=do_sample,
                             top_k=top_k, top_p=top_p)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)
