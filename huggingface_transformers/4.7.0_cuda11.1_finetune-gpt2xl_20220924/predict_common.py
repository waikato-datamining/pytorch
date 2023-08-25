# Copyright 2018 Google AI, Google Brain and Carnegie Mellon University Authors and the HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
# Copyright (c) 2023 University of Waikato, Hamilton, NZ
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import numpy as np
import torch
from transformers import GPTNeoForCausalLM, AutoTokenizer
from transformers import GPT2LMHeadModel, GPT2Tokenizer

MAX_LENGTH = int(10000)  # Hardcoded max length to avoid infinite loop

MODEL_TYPE_GPT2XL = "gpt2xl"
MODEL_TYPE_GPTNEO = "gptneo"
MODEL_TYPES = [
    MODEL_TYPE_GPT2XL,
    MODEL_TYPE_GPTNEO,
]


def set_seed(seed, device="cuda"):
    """
    Sets the specified seed in numpy, torch and cuda.

    :param seed: the seed to use
    :type seed: int
    :param device: the device that is being used for inference (cuda|cpu)
    :type device: str
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    if (device == "cuda") and torch.cuda.is_available() and (torch.cuda.device_count() > 0):
        torch.cuda.manual_seed_all(seed)


def adjust_length_to_model(length, max_sequence_length):
    """
    Adjusts the maximum length to use.

    :param length: the requested length
    :type length: int
    :param max_sequence_length: the maximum sequence length from the model
    :type max_sequence_length: int
    :return: the adjusted length
    :rtype: int
    """
    if (length < 0) and (max_sequence_length > 0):
        length = max_sequence_length
    elif 0 < max_sequence_length < length:
        length = max_sequence_length  # No generation bigger than model size
    elif length < 0:
        length = MAX_LENGTH  # avoid infinite loop
    return length


def load_gpt2xl(model_path, device="cuda", fp16=False):
    """
    Loads the GPT2-XL model and tokenizer from the model path.

    :param model_path: the directory with the model
    :type model_path: str
    :param device: the device to use (cuda|cpu)
    :type device: str
    :param fp16: whether to use half-precision floats
    :type fp16: bool
    :return: tuple of model and tokenizer
    :rtype: tuple
    """
    model = GPT2LMHeadModel.from_pretrained(model_path)
    if fp16:
        model.half()
    model.to(device)
    tokenizer = GPT2Tokenizer.from_pretrained(model_path)
    tokenizer.padding_side = "left"
    tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer


def load_gptneo(model_path, device="cuda", fp16=False):
    """
    Loads the GPT-Neo model and tokenizer from the model path.

    :param model_path: the directory with the model
    :type model_path: str
    :param device: the device to use (cuda|cpu)
    :type device: str
    :param fp16: whether to use half-precision floats
    :type fp16: bool
    :return: tuple of model and tokenizer
    :rtype: tuple
    """
    model = GPTNeoForCausalLM.from_pretrained(model_path)
    if fp16:
        model.half()
    model.to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    return model, tokenizer


def predict_gpt2xl(prompt, model, tokenizer, length=400, device="cuda", temperature=0.9, top_p=0.9, top_k=0,
                   use_cache=True, do_sample=True):
    """
    Makes a prediction with a GPT2-XL model.

    :param prompt: the prompts string to use
    :type prompt: str
    :param model: the GPT2-XL model
    :param tokenizer: the GPT2 tokenizer
    :param length: the sequence length
    :type length: int
    :param device: the device to run in the inference on (cuda|cpu)
    :type device: str
    :param temperature: the temperature to use for inference
    :type temperature: float
    :param top_p: the top p sampling parameter
    :type top_p: float
    :param top_k: the top k sampling parameter
    :type top_k: int
    :param use_cache: whether to use the cache
    :type use_cache: bool
    :param do_sample: whether to do sampling
    :type do_sample: bool
    :return: the generated text, None if failed to do so eg when prompt is empty
    :rtype: str
    """
    if len(prompt) == 0:
        return None
    ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)

    length = adjust_length_to_model(length, max_sequence_length=model.config.max_position_embeddings)
    max_length = length + ids.shape[1]

    gen_tokens = model.generate(
        ids,
        do_sample=do_sample,
        max_length=max_length,
        temperature=temperature,
        use_cache=use_cache,
        top_p=top_p,
        top_k=top_k,
    )
    return tokenizer.batch_decode(gen_tokens)[0]


def predict_gptneo(prompt, model, tokenizer, length=400, device="cuda", temperature=0.9, top_p=0.9, top_k=0,
                   use_cache=True, do_sample=True):
    """
    Makes a prediction with a GPT-Neo model.

    :param prompt: the prompts string to use
    :type prompt: str
    :param model: the GPT-Neo model
    :param tokenizer: the GPT-Neo tokenizer
    :param length: the sequence length
    :type length: int
    :param device: the device to run in the inference on (cuda|cpu)
    :type device: str
    :param temperature: the temperature to use for inference
    :type temperature: float
    :param top_p: the top p sampling parameter
    :type top_p: float
    :param top_k: the top k sampling parameter
    :type top_k: int
    :param use_cache: whether to use the cache
    :type use_cache: bool
    :param do_sample: whether to do sampling
    :type do_sample: bool
    :return: the generated text, None if failed to do so eg when prompt is empty
    :rtype: str
    """
    if len(prompt) == 0:
        return None
    ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)

    length = adjust_length_to_model(length, max_sequence_length=model.config.max_position_embeddings)
    max_length = length + ids.shape[1]

    gen_tokens = model.generate(
        ids,
        do_sample=do_sample,
        max_length=max_length,
        temperature=temperature,
        use_cache=use_cache,
        top_p=top_p,
        top_k=top_k,
    )
    return tokenizer.batch_decode(gen_tokens)[0]
