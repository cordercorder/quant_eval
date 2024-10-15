import os
import time
import json
import warnings
import argparse

import torch
import shortuuid
from tqdm import tqdm
from torch import nn

import torch.distributed as dist

from functools import partial
from auto_gptq import AutoGPTQForCausalLM
from fastchat.llm_judge.common import temperature_config
from transformers import AutoTokenizer
from transformers.trainer_utils import set_seed
from transformers.generation import GenerationConfig
from transformers.generation.streamers import BaseStreamer
from transformers.generation.logits_process import LogitsProcessorList
from transformers.generation.stopping_criteria import StoppingCriteriaList, validate_stopping_criteria
from transformers.generation.utils import SampleOutput, SampleEncoderDecoderOutput, SampleDecoderOnlyOutput
from typing import List, Optional, Union



def load_questions(question_file: str):
    questions = []
    with open(question_file, "r") as ques_file:
        for line in ques_file:
            if line:
                questions.append(json.loads(line))
    return questions


def load_models_tokenizer(args):
    tokenizer = AutoTokenizer.from_pretrained(
        args.checkpoint_path, trust_remote_code=True
    )

    model = AutoGPTQForCausalLM.from_quantized(
        args.quant_checkpoint_path,
        device_map={"": "cuda"},
        low_cpu_mem_usage=True,
        trust_remote_code=True
    )
    model = model.model
    model.generation_config = GenerationConfig.from_pretrained(
        args.checkpoint_path, trust_remote_code=True
    )

    print(f"GenerationConfig: {model.generation_config}")
    return model, tokenizer


def sample(
    self,
    input_ids: torch.LongTensor,
    logits_processor: Optional[LogitsProcessorList] = None,
    stopping_criteria: Optional[StoppingCriteriaList] = None,
    logits_warper: Optional[LogitsProcessorList] = None,
    max_length: Optional[int] = None,
    pad_token_id: Optional[int] = None,
    eos_token_id: Optional[Union[int, List[int]]] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    output_scores: Optional[bool] = None,
    return_dict_in_generate: Optional[bool] = None,
    synced_gpus: bool = False,
    streamer: Optional["BaseStreamer"] = None,
    **model_kwargs,
) -> Union[SampleOutput, torch.LongTensor]:

    # init values
    logits_processor = logits_processor if logits_processor is not None else LogitsProcessorList()
    stopping_criteria = stopping_criteria if stopping_criteria is not None else StoppingCriteriaList()
    if max_length is not None:
        warnings.warn(
            "`max_length` is deprecated in this function, use"
            " `stopping_criteria=StoppingCriteriaList([MaxLengthCriteria(max_length=max_length)])` instead.",
            UserWarning,
        )
        stopping_criteria = validate_stopping_criteria(stopping_criteria, max_length)
    logits_warper = logits_warper if logits_warper is not None else LogitsProcessorList()
    pad_token_id = pad_token_id if pad_token_id is not None else self.generation_config.pad_token_id
    eos_token_id = eos_token_id if eos_token_id is not None else self.generation_config.eos_token_id
    if isinstance(eos_token_id, int):
        eos_token_id = [eos_token_id]
    eos_token_id_tensor = torch.tensor(eos_token_id).to(input_ids.device) if eos_token_id is not None else None
    output_scores = output_scores if output_scores is not None else self.generation_config.output_scores
    output_attentions = (
        output_attentions if output_attentions is not None else self.generation_config.output_attentions
    )
    output_hidden_states = (
        output_hidden_states if output_hidden_states is not None else self.generation_config.output_hidden_states
    )
    return_dict_in_generate = (
        return_dict_in_generate
        if return_dict_in_generate is not None
        else self.generation_config.return_dict_in_generate
    )

    # init attention / hidden states / scores tuples
    scores = () if (return_dict_in_generate and output_scores) else None
    decoder_attentions = () if (return_dict_in_generate and output_attentions) else None
    cross_attentions = () if (return_dict_in_generate and output_attentions) else None
    decoder_hidden_states = () if (return_dict_in_generate and output_hidden_states) else None

    # if model is an encoder-decoder, retrieve encoder attention weights and hidden states
    if return_dict_in_generate and self.config.is_encoder_decoder:
        encoder_attentions = model_kwargs["encoder_outputs"].get("attentions") if output_attentions else None
        encoder_hidden_states = (
            model_kwargs["encoder_outputs"].get("hidden_states") if output_hidden_states else None
        )

    # keep track of which sequences are already finished
    unfinished_sequences = torch.ones(input_ids.shape[0], dtype=torch.long, device=input_ids.device)

    this_peer_finished = False  # used by synced_gpus only
    # auto-regressive generation
    while True:
        if synced_gpus:
            # Under synced_gpus the `forward` call must continue until all gpus complete their sequence.
            # The following logic allows an early break if all peers finished generating their sequence
            this_peer_finished_flag = torch.tensor(0.0 if this_peer_finished else 1.0).to(input_ids.device)
            # send 0.0 if we finished, 1.0 otherwise
            dist.all_reduce(this_peer_finished_flag, op=dist.ReduceOp.SUM)
            # did all peers finish? the reduced sum will be 0.0 then
            if this_peer_finished_flag.item() == 0.0:
                break

        # prepare model inputs
        model_inputs = self.prepare_inputs_for_generation(input_ids, **model_kwargs)

        # forward pass to get next token
        outputs = self(
            **model_inputs,
            return_dict=True,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )

        if synced_gpus and this_peer_finished:
            continue  # don't waste resources running the code we don't need

        next_token_logits = outputs.logits[:, -1, :]

        # modify
        previous_dtype = next_token_logits.dtype
        
        # modify
        next_token_logits = next_token_logits.float()

        # pre-process distribution
        next_token_scores = logits_processor(input_ids, next_token_logits)
        next_token_scores = logits_warper(input_ids, next_token_scores)

        # Store scores, attentions and hidden_states when required
        if return_dict_in_generate:
            if output_scores:
                # modify
                scores += (next_token_scores.to(previous_dtype),)
            if output_attentions:
                decoder_attentions += (
                    (outputs.decoder_attentions,) if self.config.is_encoder_decoder else (outputs.attentions,)
                )
                if self.config.is_encoder_decoder:
                    cross_attentions += (outputs.cross_attentions,)

            if output_hidden_states:
                decoder_hidden_states += (
                    (outputs.decoder_hidden_states,)
                    if self.config.is_encoder_decoder
                    else (outputs.hidden_states,)
                )

        # sample
        probs = nn.functional.softmax(next_token_scores, dim=-1)
        next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)

        # finished sentences should have their next token be a padding token
        if eos_token_id is not None:
            if pad_token_id is None:
                raise ValueError("If `eos_token_id` is defined, make sure that `pad_token_id` is defined.")
            next_tokens = next_tokens * unfinished_sequences + pad_token_id * (1 - unfinished_sequences)

        # update generated ids, model inputs, and length for next step
        input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)
        if streamer is not None:
            streamer.put(next_tokens.cpu())
        model_kwargs = self._update_model_kwargs_for_generation(
            outputs, model_kwargs, is_encoder_decoder=self.config.is_encoder_decoder
        )

        # if eos_token was found in one sentence, set sentence to finished
        if eos_token_id_tensor is not None:
            unfinished_sequences = unfinished_sequences.mul(
                next_tokens.tile(eos_token_id_tensor.shape[0], 1).ne(eos_token_id_tensor.unsqueeze(1)).prod(dim=0)
            )

            # stop when each sentence is finished
            if unfinished_sequences.max() == 0:
                this_peer_finished = True

        # stop if we exceed the maximum length
        if stopping_criteria(input_ids, scores):
            this_peer_finished = True

        if this_peer_finished and not synced_gpus:
            break

    if streamer is not None:
        streamer.end()

    if return_dict_in_generate:
        if self.config.is_encoder_decoder:
            return SampleEncoderDecoderOutput(
                sequences=input_ids,
                scores=scores,
                encoder_attentions=encoder_attentions,
                encoder_hidden_states=encoder_hidden_states,
                decoder_attentions=decoder_attentions,
                cross_attentions=cross_attentions,
                decoder_hidden_states=decoder_hidden_states,
                past_key_values=model_kwargs.get("past_key_values"),
            )
        else:
            return SampleDecoderOnlyOutput(
                sequences=input_ids,
                scores=scores,
                attentions=decoder_attentions,
                hidden_states=decoder_hidden_states,
                past_key_values=model_kwargs.get("past_key_values"),
            )
    else:
        return input_ids


def cli_main(args):
    print("loading model weights")
    model, tokenizer = load_models_tokenizer(args)

    model.sample = partial(sample, model)
    print("model loaded")

    model_id = os.path.basename(args.checkpoint_path)

    questions = load_questions(args.question_file)

    os.makedirs(os.path.dirname(args.answer_file), exist_ok=True)

    if os.path.isfile(args.answer_file) and not args.overwrite:
        mode = "a"
        num_skiped_examples = 0
        for _ in open(args.answer_file, mode="r", encoding="utf-8"):
            num_skiped_examples += 1
    else:
        mode = "w"
        num_skiped_examples = 0

    with open(args.answer_file, mode=mode, encoding="utf-8") as fout:
        
        question_id = 0

        for question in tqdm(questions):
            
            if question_id < num_skiped_examples:
                question_id += 1
                continue
            else:
                question_id += 1

            if question["category"] in temperature_config:
                temperature = temperature_config[question["category"]]
            else:
                temperature = 0.7

            choices = []

            for i in range(args.num_choices):
                torch.manual_seed(i)

                turns = []
                history = []

                for j in range(len(question["turns"])):
                    question_turn_j = question["turns"][j]

                    if temperature < 1e-4:
                        do_sample = False
                    else:
                        do_sample = True
                    
                    response, history = model.chat(
                        tokenizer, 
                        question_turn_j, 
                        history=history,
                        do_sample=do_sample,
                        temperature=temperature,
                        max_new_tokens=args.max_new_token,
                        remove_invalid_values=True,
                    )

                    turns.append(response)
                
                choices.append({"index": i, "turns": turns})

            ans_json = {
                "question_id": question["question_id"],
                "answer_id": shortuuid.uuid(),
                "model_id": model_id,
                "choices": choices,
                "tstamp": time.time(),
            }
            fout.write(json.dumps(ans_json, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-s", 
        "--seed", 
        type=int, 
        default=1234, 
        help="Random seed"
    )
    parser.add_argument(
        "-c",
        "--checkpoint_path",
        type=str,
        help="Checkpoint path",
        default="Qwen/Qwen-7B-Chat",
    )
    parser.add_argument(
        "--quant_checkpoint_path",
        type=str,
        help="Checkpoint path of quantized model",
        required=True,
    )

    parser.add_argument(
        "--question_file", 
        type=str,
        required=True,
    )

    parser.add_argument(
        "--answer_file", 
        type=str, 
        required=True,
        help="The output answer file."
    )

    parser.add_argument(
        "--num_choices",
        type=int,
        default=1,
        help="How many completion choices to generate.",
    )
    parser.add_argument(
        "--max_new_token",
        type=int,
        default=1024,
        help="The maximum number of new generated tokens.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        default=False
    )

    args = parser.parse_args()

    print(f"Args: {args}")

    set_seed(args.seed)

    cli_main(args)
