import argparse
import json
import os
import pprint

from typing import List

import numpy as np
import pandas as pd
import torch
import tqdm

from evaluate import load as load_metric
from peft import AutoPeftModelForCausalLM, AutoPeftModelForSeq2SeqLM, get_peft_model, LoraConfig, PeftModel
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import trange
from transformers import AutoModelForSeq2SeqLM, AutoModelForCausalLM, AutoTokenizer, AdamW, Adafactor

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

torch.manual_seed(0)
np.random.seed(0)

class QueryDataset(torch.utils.data.Dataset):
    def __init__(self, inputs, targets):
        self.inputs = inputs
        self.targets = targets

    def __getitem__(self, idx):

        input_ids = torch.tensor(self.inputs["input_ids"][idx]).squeeze()
        input_attn_mask = torch.tensor(self.inputs["attention_mask"][idx]).squeeze()
        
        target_ids = torch.tensor(self.targets["input_ids"][idx]).squeeze()
        target_attn_mask = torch.tensor(self.targets["attention_mask"][idx]).squeeze()
        
        return {"input_ids": input_ids, "input_mask":input_attn_mask,
                "labels": target_ids, "labels_mask": target_attn_mask}
    
    def __len__(self):
        return len(self.targets.data['input_ids'])

# ChatModels specifically
class CausalLMQueryDataset(torch.utils.data.Dataset):
    
    def __init__(self, tokenizer, inputs, targets, encoded_instances, chat_template_tokens):
        self.tokenizer = tokenizer
        self.inputs = inputs
        self.targets = targets
        self.encoded_instances = encoded_instances
        self.chat_template_tokens = chat_template_tokens

    def from_inputs(tokenizer, inputs: List[str], targets: List[str]):
        assert hasattr(tokenizer, 'apply_chat_template')

        # all of this is to automatically identify tokens to mask spans
        # assuming that "the" is one token long
        chat_template_span_token = 'the'
        chat_template_span_token_id = tokenizer(chat_template_span_token, add_special_tokens=False)['input_ids']
        assert len(chat_template_span_token_id) == 1
        chat_template_span_token_instance = tokenizer.apply_chat_template([{'role': 'user', 'content': chat_template_span_token}])
        chat_template_span_token_instance = np.array(chat_template_span_token_instance)
        (placeholder_token_positions,) = np.where(chat_template_span_token_instance == chat_template_span_token_id[0])
        #(start_token_pos,) = np.where(chat_template_span_token_instance == tokenizer.bos_token_id)[0]
        #start_chat_template_token_ids = chat_template_span_token_instance[start_token_pos+1:placeholder_token_positions.item()]
        end_chat_template_token_ids = chat_template_span_token_instance[placeholder_token_positions.item() + 1:]
        #assert len(start_chat_template_token_ids) > 0
        assert len(end_chat_template_token_ids) > 0

        encoded_instances = []
        for inp, target in zip(inputs, targets):
            messages = [
                {
                    'role': 'user',
                    'content': inp,
                },
                {
                    'role': 'assistant',
                    # the space " " here is very important; the end instruction template '[/INST]' can be incorrectly tokenized when the target begins with one or more open parentheses - to avoid this this we force a token boundary via a space.
                    'content': ' ' + target.strip(),
                },
            ]
            encodeds = tokenizer.apply_chat_template(messages, return_tensors="np")
            #labels = np.array([np.concatenate([i, [tokenizer.eos_token_id]]) for i in encodeds])
            labels = encodeds
            # apparently this shouldn't be computed in advance
            # TODO
            #labels = encodeds[0, 1:] + [[tokenizer.eos_token_id]]
            prediction_mask_start_position = np.where(np.all(np.lib.stride_tricks.sliding_window_view(labels[0], len(end_chat_template_token_ids)) == end_chat_template_token_ids, axis=1))
            assert len(prediction_mask_start_position) == 1
            prediction_mask_start_position = prediction_mask_start_position[0].item()
            #prediction_mask = np.zeros(encodeds.size, dtype=int)
            #prediction_mask[0][:prediction_mask_start_position] = -100
            prediction_mask = np.ones(encodeds.size, dtype=int)
            prediction_mask[:prediction_mask_start_position] = 0
            input_mask = np.triu(np.full(encodeds[0].size, 1))
            input_mask[input_mask == 0] = -100
            input_mask[input_mask == 1] = 0
            encoded_instances.append({
                'input_ids': torch.LongTensor(encodeds).squeeze(),
                'input_mask': torch.LongTensor(input_mask),
                'labels': torch.LongTensor(labels).squeeze(),
                'labels_mask': torch.LongTensor(prediction_mask).squeeze(),
            })
            
        return CausalLMQueryDataset(tokenizer, inputs, targets, encoded_instances, torch.Tensor(end_chat_template_token_ids))

    def completion_start_index(self, data: torch.Tensor) -> int:
        strides = data[0].unfold(0, len(self.chat_template_tokens), 1)
        matching = strides == self.chat_template_tokens.to(strides.device)
        chat_template_start = torch.where(torch.all(matching, dim=-1))
        assert len(chat_template_start) == 1
        (chat_template_start,) = chat_template_start
        return chat_template_start.item() + len(self.chat_template_tokens)


    def __getitem__(self, idx):
        return self.encoded_instances[idx]
    
    def __len__(self):
        return len(self.inputs)


def train(
        model,
        tokenizer,
        optimizer,
        epochs,
        batch_size,
        output_dir,
        writer,
        train_dataset,
        gradient_accumulation_steps=None,
        #clip_grad_norm=None,
        val_dataset=None,
        val_df=None,
        generation_config=None,
        restore_from_epoch=None,
    ):
    assert gradient_accumulation_steps is None or gradient_accumulation_steps > 0
    model.train()

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)


    best_model_output = None
    best_val_epoch = None
    best_val_loss = None
    best_val_rouge1 = None
    if val_dataset is not None:
        val_loss, decoded, rouge_scores = eval(model, tokenizer, val_dataset, print_outputs=False, generation_config=generation_config)
        print(decoded[0])
        writer.add_scalar("loss/val/epoch", val_loss, 0)
        for k, v in rouge_scores.items():
            writer.add_scalar(f"loss/val/{k}", v, 0)
        best_val_epoch = -1
        best_val_loss = val_loss
        best_val_rouge1 = rouge_scores['rouge1']
        with open(os.path.join(output_dir, f'val_generated_pre.txt'), 'w') as of:
            for g in decoded:
                of.write(g)
                of.write('\n')

    # TODO this should really be formatted as a dataclass
    epoch_info_file = os.path.join(output_dir, 'epoch_info.json')
    if restore_from_epoch is not None and os.path.exists(epoch_info_file):
        with open(epoch_info_file, 'r') as inf:
            epoch_info = json.loads(inf.read())
        for i in range(len(epoch_info['epoch'])):
            if epoch_info['val_losses'][i] < best_val_loss:
                best_val_epoch = epoch_info['epoch'][i]
                best_val_loss = epoch_info['val_losses'][i]
                best_model_output = epoch_info['model_outputs'][i]
    else:
        epoch_info = {
            'epoch': [],
            'train_losses': [],
            'val_losses': [],
            'train_batch_losses': [],
            'model_outputs': [],
            'optimizer_outputs': [],
        }


    loss_fn = torch.nn.CrossEntropyLoss(reduction='none')
    # skips training batches to restore training sampling
    start_epoch = restore_from_epoch + 1 if restore_from_epoch else 0
    print(f'Skipping the first {start_epoch} epochs, resuming from previous training')
    with trange(0, start_epoch, desc='epoch') as epoch_iterator:
        with tqdm.tqdm(train_loader, desc='batch') as batch_iter:
            for batch in batch_iter:
                continue
    with trange(start_epoch, epochs, desc='epoch') as epoch_iterator:
        model.train()
        for epoch in epoch_iterator:
            epoch_info['epoch'].append(epoch)
            epoch_info['train_batch_losses'].append([])
            epoch_loss = 0.0
            epoch_iterator.set_postfix(
                val_loss=val_loss,
                best_val_loss=best_val_loss,
                best_val_epoch=best_val_epoch,
                best_rouge1=best_val_rouge1,
            )
            with tqdm.tqdm(train_loader, desc='batch') as batch_iter:
                batches = 0
                for batch in batch_iter:
                    batches += 1
                    input_ids = batch['input_ids'].to(device)
                    input_attn_mask = batch['input_mask'].to(device)

                    labels = batch['labels'].to(device)
                    if model.config.is_encoder_decoder:
                        # mask the loss for pad tokens
                        labels = torch.where(labels != tokenizer.pad_token_id, labels, -100)
                        outputs = model(input_ids, attention_mask=input_attn_mask, labels=labels)
                        loss = outputs[0]
                    else:
                        # TODO make the attention mask cooperate here...the right mask gets created in the model but this is dissatisfying
                        outputs = model(input_ids)
                        # we predict the next token; we don't care about predicting anything beyond <eos>
                        logits = outputs.logits[..., :-1, :]
                        # the true label for the above is this token rightward shifted.
                        labels = input_ids[..., 1:]
                        labels_mask = batch['labels_mask'].to(logits.device)
                        labels_mask = labels_mask[..., 1:]
                        # manually perform a mean
                        loss = loss_fn(logits.squeeze(dim=0), labels.squeeze(dim=0)) * labels_mask
                        loss = loss.sum() / labels_mask.sum()

                    assert loss == loss, "Found a nan!"
                    writer.add_scalar("loss/train/batch", loss.detach().cpu().item(), batches)
                    epoch_info['train_batch_losses'][-1].append(loss.detach().cpu().item())
                    batch_iter.set_postfix(batch_loss='{:.3f}'.format(loss.detach().cpu().item()))

                    epoch_loss += loss.detach().item()
                    if gradient_accumulation_steps is not None:
                        loss = loss / gradient_accumulation_steps
                    loss.backward()
                    # TODO: add grad norm, or grad value clipping?
                    # TODO: is https://stackoverflow.com/a/54816498 better?
                    #if clip_grad_norm is not None:
                    #    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_grad_norm, error_if_nonfinite=True, foreach=True)
                    if gradient_accumulation_steps is None \
                        or (gradient_accumulation_steps is not None and batches % gradient_accumulation_steps == 0) \
                        or (gradient_accumulation_steps is not None and batches == len(train_dataset)):
                        optimizer.step()
                        optimizer.zero_grad()
                    epoch_iterator.set_postfix(epoch_loss=epoch_loss/batches)
                epoch_info['train_losses'].append(epoch_loss/batches)
                writer.add_scalar("loss/train/epoch", np.mean(epoch_info['train_losses']), batches)
            print("-"*25)
            print("train loss on epoch {}: {:.2f}\n".format(epoch, epoch_loss / batches))
            if val_dataset is not None:
                val_loss, decoded, rouge_scores = eval(
                    model,
                    tokenizer,
                    val_dataset,
                    print_outputs=(epoch == epochs -1),
                    generation_config=generation_config)
                epoch_info['val_losses'].append(val_loss)
                writer.add_scalar("loss/val/epoch", val_loss, epoch)
                for k, v in rouge_scores.items():
                    writer.add_scalar(f"loss/val/{k}", v, epoch)
                print(decoded[0])
                print(f'val loss: {val_loss}')
                if val_df is not None:
                    val_df['generated'] = decoded
                    val_df.to_csv(os.path.join(output_dir, f'val_generated_epoch{epoch}.csv'))
                with open(os.path.join(output_dir, f'val_generated_epoch{epoch}.txt'), 'w') as of:
                    for g in decoded:
                        of.write(g)
                        of.write('\n')
            else:
                val_loss = None
            if val_dataset is not None and (best_val_loss is None or val_loss < best_val_loss):
                assert epoch >= 0
                output = os.path.join(os.path.abspath(output_dir), f'checkpoint_epoch_{epoch:04d}-best')
                epoch_info['model_outputs'].append(output)
                epoch_info['optimizer_outputs'].append(output + '.optimizer')
                print("\n **** \nnew best loss; dumping model to disk @ {}\n **** \n".format(output))
                if isinstance(model, PeftModel):
                    os.makedirs(output, exist_ok=True)
                    model.save_pretrained(output)
                else:
                    torch.save(model.state_dict(), output)
                torch.save(optimizer.state_dict(), output + '.optimizer')
                best_val_loss = val_loss
                best_val_epoch = epoch
                best_val_rouge1 = rouge_scores['rouge1']
                best_model_output = output
            else:
                assert epoch >= 0
                output = os.path.join(os.path.abspath(output_dir), f'checkpoint_epoch_{epoch:04d}')
                epoch_info['model_outputs'].append(output)
                epoch_info['optimizer_outputs'].append(output + '.optimizer')
                print("new val loss not better than previous, dumping to disk @ {}".format(output))
                torch.save(optimizer.state_dict(), output + '.optimizer')
                if isinstance(model, PeftModel):
                    os.makedirs(output, exist_ok=True)
                    model.save_pretrained(output)
                else:
                    torch.save(model.state_dict(), output)

            epoch_iterator.set_postfix(
                epoch_loss=epoch_loss/batches,
                val_loss=val_loss,
                best_val_loss=best_val_loss,
                best_val_epoch=best_val_epoch,
                best_rouge1=best_val_rouge1,
            )
            with open(epoch_info_file, 'w') as of:
                of.write(json.dumps(epoch_info))
            print("-"*25)
    return epoch_info, best_model_output, best_val_epoch


def eval(model, tokenizer, val_dataset, max_tokens=200, print_outputs=True, generation_config=None):
    model.eval()
    batch_size = 1
    # make the batch size 1 for simplicity of decoding and decoding args (and hushing a warning).
    val_loader = DataLoader(val_dataset, batch_size=1)
    val_loss = 0
    decoded = []
    targets = []
    batches = 0
    loss_fn = torch.nn.CrossEntropyLoss(reduction='none')

    if generation_config is None:
        generation_config = {}
    if 'max_new_tokens' in generation_config:
        max_tokens = generation_config.get('max_new_tokens', max_tokens)
        del generation_config["max_new_tokens"]

    for batch in tqdm.tqdm(val_loader, desc='evaluate'):
        batches += 1
        input_ids = batch['input_ids'].to(device)
        input_attn_mask = batch['input_mask'].to(device)
        labels = batch['labels'].to(device)
        if model.config.is_encoder_decoder:
            if 'labels_mask' in batch:
                labels_mask = batch['labels_mask'].to(torch.bool).to(device)
                labels[~labels_mask] = -100
            outputs = model(input_ids, attention_mask=input_attn_mask, labels=labels)   
            loss = outputs[0]
        else:
            labels_mask = batch['labels_mask'].to(torch.bool).to(device)
            labels_mask = labels_mask[..., 1:]
            outputs = model(
                input_ids=input_ids,
                #attention_mask=input_attn_mask
            )
            logits = outputs.logits[..., :-1, :].contiguous()
            labels = input_ids[..., 1:].contiguous()
            # manually perform a mean
            loss = loss_fn(logits.squeeze(dim=0), labels.squeeze(dim=0))
            loss = loss * labels_mask
            loss = loss.sum() / labels_mask.sum()
        val_loss += loss.item()
        
        if model.config.is_encoder_decoder:
            search_strs = model.generate(input_ids, attention_mask=input_attn_mask, max_new_tokens=max_tokens, **generation_config)
            generated = tokenizer.batch_decode(search_strs, skip_special_tokens=True)
            labels = torch.where(labels != -100, labels, tokenizer.pad_token_id)
        else:
            assert input_ids.shape[0] == 1
            completion_start = val_dataset.completion_start_index(input_ids)
            search_inputs = input_ids[..., :completion_start]
            search_strs = model.generate(
                # HF bug:
                # PeftModel* do not pass *args, only **kwargs, so everything needs to passed as kwargs
                input_ids=search_inputs,
                max_new_tokens=max_tokens,
                # specify pad token to shush a warning, this is the default behavior.
                # nothing to see here since generation is done one at a time.
                pad_token_id=tokenizer.eos_token_id,
                **generation_config,
            )
            search_strs = search_strs[..., completion_start:]
            labels = input_ids[..., completion_start:]
            generated = tokenizer.batch_decode(search_strs, skip_special_tokens=True)
        generated = [x.replace('\n', ' ' ) for x in generated]
        decoded.extend(generated)
        # this is gross, but now I swap back from -100 to pad token ids to print out
        target_strs = tokenizer.batch_decode(labels, skip_special_tokens=True)
        targets.extend(target_strs)
        if print_outputs:
            print("input:", tokenizer.batch_decode(input_ids[..., :completion_start]))
            print("generated:")
            print(generated)
            print("targets:")
            print(target_strs)
            print("\n\n")
    rouge_scorer = load_metric('rouge')
    rouge_score = rouge_scorer.compute(predictions=decoded, references=[[x] for x in targets])
    print('rouge:')
    pprint.pprint(rouge_score)
    print("\n\nloss on val: {:.2f}\n".format(val_loss/batches))
    return val_loss/batches, decoded, rouge_score


def prep_seq2seq(tokenizer, inputs, targets):
    x = tokenizer(list(inputs), truncation=True, pad_to_max_length=True, padding="max_length")
    y = tokenizer(list(targets), truncation=True, pad_to_max_length=True, padding="max_length")
    return QueryDataset(x, y)


def load_model(args, resume_from_checkpoint=False, init_weights_path=None, init_lora_path=None, optimizer_path=None):
    # enforce arg consistency
    if not resume_from_checkpoint:
        assert init_weights_path is None
        assert init_lora_path is None
    if init_weights_path is not None:
        assert resume_from_checkpoint
        assert init_lora_path is None
    if init_lora_path is not None:
        assert resume_from_checkpoint
        assert init_weights_path is None

    is_peft = False
    if args.peft_r is not None or args.peft_alpha is not None or args.peft_dropout is not None:
        assert args.peft_r is not None
        assert args.peft_alpha is not None
        assert args.peft_dropout is not None
        is_peft = True

    def get_epoch_from_checkpoint(ckpt):
        # strip '-best' from the epoch strings
        if ckpt[-1] == '/':
            ckpt = ckpt[:-1]
        nobest = os.path.basename(ckpt).split('-')[0]
        # the format is checkpoint_epoch_num
        checkpoint_num = nobest.split('_')[-1]
        checkpoint_num = int(checkpoint_num)
        return checkpoint_num

    if init_weights_path is None and init_lora_path is None and resume_from_checkpoint:
        epoch_info_file = os.path.join(args.output_dir, 'epoch_info.json')
        if os.path.exists(epoch_info_file):
            with open(epoch_info_file, 'r') as inf:
                epoch_info = json.loads(inf.read())
            restore_from_epoch = epoch_info['epoch'][-1]
            model_path = epoch_info['model_outputs'][-1]
            optimizer_path = epoch_info['optimizer_outputs'][-1]
            if os.path.isfile(model_path):
                init_weights_path = model_path
            elif os.path.isdir(model_path):
                init_lora_path = model_path
            else:
                assert False, "animal, vegetable, or mineral?"
        else:
            restore_from_epoch = -1

    elif resume_from_checkpoint and init_weights_path is not None:
        # TODO verify this works
        assert False
        restore_from_epoch = get_epoch_from_checkpoint(init_weights_path)
    elif resume_from_checkpoint and init_lora_path is not None:
        restore_from_epoch = get_epoch_from_checkpoint(init_lora_path)
    else:
        restore_from_epoch = -1

    assert not (args.fp16 and args.bf16), 'pick either fp16 or bf16, not both'
    if args.fp16:
        model_type = torch.float16
    elif args.bf16:
        model_type = torch.bfloat16
    else:
        model_type = 'auto'
    # casual models:
    # peft: restore from training
    if args.is_causal and is_peft and init_lora_path is not None and resume_from_checkpoint:
        # TODO verify peft args are compatible
        print(f'Resuming peft model for causal lm from {init_lora_path}')
        model = AutoPeftModelForCausalLM.from_pretrained(init_lora_path, is_trainable=True, attn_implementation="flash_attention_2", torch_dtype=model_type)
    # peft: initialization, ignoring any existing training
    elif args.is_causal and is_peft and (init_lora_path is None or not resume_from_checkpoint):
        model = AutoModelForCausalLM.from_pretrained(args.model_name, attn_implementation="flash_attention_2", torch_dtype=model_type)
        lora_config = LoraConfig(
            r=args.peft_r,
            target_modules=["q_proj", "k_proj", "v_proj", "out_proj", "fc_in", "fc_out", "wte"],
            lora_alpha=args.peft_alpha,
            lora_dropout=args.peft_dropout,
            bias='all',
        )
        model = get_peft_model(model, lora_config)
    # not peft: restore from training or not
    elif args.is_causal and not is_peft:
        assert init_lora_path is None
        model = AutoModelForCausalLM.from_pretrained(args.model_name, attn_implementation="flash_attention_2", torch_dtype=model_type)
        if init_weights_path is not None and resume_from_checkpoint:
            print(f'Resuming model for causal lm from {init_model_path}')
            model.load_state_dict(torch.load(init_weights_path))
    # seq2seq models:
    # peft: restore from training
    elif not args.is_causal and is_peft and init_lora_path is not None and resume_from_checkpoint:
        print(f'Resuming peft model for seq2seq lm from {init_lora_path}')
        assert False
        # TODO verify peft args are compatible
        # TODO, does this work?
        model = AutoPeftModelForSeq2SeqLM.from_pretrained(init_lora_path, is_trainable=True)
    # peft: initialization, ignoring any existing training
    elif not args.is_causal and is_peft and (init_lora_path is None or not resume_from_checkpoint):
        assert False
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        lora_config = LoraConfig(
            r=args.peft_r,
            target_modules=["q_proj", "k_proj", "v_proj", "out_proj", "fc_in", "fc_out", "wte"],
            lora_alpha=args.peft_alpha,
            lora_dropout=args.peft_dropout,
            bias='all',
        )
        model = get_peft_model(model, lora_config)
    # not peft: restore from training or not
    elif not args.is_causal and not is_peft:
        assert False
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        if init_weights_path is not None and resume_from_checkpoint:
            print(f'Resuming model for seq2seq lm from {init_model_path}')
            model.load_state_dict(torch.load(init_weights_path))
    else:
        assert False

    model.to(device=device)
    # TODO torch.compile?
    # https://stackoverflow.com/questions/75886125/how-should-i-use-torch-compile-properly
    # TODO What interactions does this have with training, evaluation, saving, and loading?
    # training: https://discuss.pytorch.org/t/how-exactly-to-use-torch-compile/190486/2
    # saving: https://github.com/pytorch/pytorch/issues/101107
    # how does this interact with PEFT?

    optimizer = load_optimizer(model, args)
    if resume_from_checkpoint and restore_from_epoch > -1:
        if optimizer_path is not None:
            optimizer.load_state_dict(torch.load(optimizer_path))
        else:
            print('warning, not restoring optimizer from previous state')

    return model, optimizer, restore_from_epoch


def load_optimizer(model, args):
    # TODO this should support more optimizers (if I care); consider using HF optimizer set up
    if args.optimizer == 'adafactor':
        optim = Adafactor(
            model.parameters(),
            lr=args.learning_rate,
            scale_parameter=False, relative_step=False, warmup_init=False
        )
    elif args.optimizer == 'adamw':
        optim = AdamW(
            model.parameters(),
            lr=args.learning_rate,
        )
    else:
        assert False
    return optim


def main(args):
    os.makedirs(args.output_dir, exist_ok=True)
    args_file = os.path.join(args.output_dir, 'args.json')
    if os.path.exists(args_file):
        with open(args_file, 'r') as inf:
            old_args_dict = json.loads(inf.read())
        current_args_dict = args.__dict__
        all_keys = set(old_args_dict) | set(current_args_dict)
        matched, unmatched, new_args, missing_args = set(), set(), set(), set()
        for k in all_keys:
            if k in current_args_dict and k in old_args_dict:
                if current_args_dict[k] == old_args_dict[k]:
                    matched.add(k)
                else:
                    unmatched.add(k)
            if k in current_args_dict and k not in old_args_dict:
                new_args.add(k)
            if k not in current_args_dict and k in old_args_dict:
                missing_args.add(k)
        if len(matched) == len(all_keys):
            print(f'Found existing arguments {args_file}, all keys matched')
        else:
            os.rename(args_file, args_file.replace('.json', '.old.json'))
            print(f'Found existing arguments {args_file}')
            print(f'Matching keys: {matched}')
            if len(unmatched) > 0:
                print(f'Found unmatched keys: {len(unmatched)}')
                for k in unmatched:
                    print(f'Unmatched {k}, old: {old_args_dict[k]},\tnew: {current_args_dict[k]}')
            if len(new_args) > 0:
                print(f'Found new args: {len(new_args)}')
                for k in new_args:
                    print(f'{k}, {current_args_dict[k]}')
            if len(new_args) > 0:
                print(f'Found missing args: {len(missing_args)}')
                for k in missing_args:
                    print(f'{k}, {old_args_dict[k]}')

    with open(args_file, 'w') as of:
        of.write(json.dumps(args.__dict__, indent=4))
    print("model: {}, epochs: {}; batch size: {}; learning rate: {}".format(
        args.model_name, args.epochs, args.batch_size, args.learning_rate))
    writer = SummaryWriter(log_dir=os.path.join(args.output_dir, 'runs'))
    if args.generation_config:
        with open(args.generation_config, 'r') as inf:
            generation_config = json.loads(inf.read())
    else:
        generation_config = None

    df = pd.read_csv(args.data_path)
    # cannot do this, training data does not have ground truth for included studies so this would delete all training
    #df = df.dropna()
    df = df[df[args.target_field].notna()]
    df = df[df[args.input_field].notna()]

    assert 'input' not in df.columns
    #df['inputs'] = "Translate the following into a boolean search query to find relevant studies in PubMed. Do not add any explanation. Do not repeat terms. Do not add MeSH terms. Prefer shorter queries.\n" + df[args.input_field].astype(str)
    df['inputs'] = "Translate the following into a boolean search query to find relevant studies in PubMed. Do not add any explanation. Do not repeat terms. Prefer shorter queries.\n" + df[args.input_field].astype(str)
    #df['inputs'] = "Translate the following into a search query.\n " + df[args.input_field].astype(str)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    # this happens by default anyway. Ostensibly this is used to construct masks, but:
    # 1. models without this set (i.e. encoder/decoder models) already fit multiple instances & construct this explicitly
    # 2. models without this set (i.e. causal decoder-only models) don't have this, so we set it here to shush the (irrelevant)
    #    warning because we train one instance at a time here.
    if not hasattr(tokenizer, 'pad_token_id') or tokenizer.pad_token_id is None:
        print('setting pad token id!')
        tokenizer.pad_token_id = tokenizer.eos_token_id

    train_df = df[df['Split'] == 'train'][:args.limit_train_size]
    val_df = df[df['Split'].isin(['val', 'dev'])][:args.limit_val_size]
    test_df = df[df['Split'] == 'test'][:args.limit_test_size]
    print(f'Have {len(train_df)} train instances, {len(val_df)} val instances, {len(test_df)} test instances')
    if args.is_causal:
        train_dataset = CausalLMQueryDataset.from_inputs(tokenizer, train_df['inputs'].values, train_df[args.target_field].values)    
        val_dataset = CausalLMQueryDataset.from_inputs(tokenizer, val_df['inputs'].values, val_df[args.target_field].values)    
        test_dataset = CausalLMQueryDataset.from_inputs(tokenizer, test_df['inputs'].values, test_df[args.target_field].values)    
    else:
        train_inputs = train_df['inputs'].values
        train_targets = train_df[args.target_field].values

        val_inputs = val_df['inputs'].values
        val_targets = val_df[args.target_field].values

        test_inputs = test_df['inputs'].values
        test_targets = test_df[args.target_field].values
        
        train_dataset = prep_seq2seq(train_inputs, train_targets)
        val_dataset = prep_seq2seq(val_inputs, val_targets)
        test_dataset = prep_seq2seq(test_inputs, test_targets)

    model, optimizer, restore_from_epoch = load_model(
        args,
        resume_from_checkpoint=args.resume_from_training,
        init_weights_path=args.init_weights_path,
        init_lora_path=args.init_lora_path,
    )

    if not args.do_train and args.init_weights_path is None and args.init_lora_path is None:
        print(f'Warning: you are not training, and no initialization paths are specified!')

    # TODO:
    # - verify that restarting training works vs. continuing all the way through
    if args.do_train:
        _, best_model_path, best_model_epoch = train(
            tokenizer=tokenizer,
            model=model,
            optimizer=optimizer,
            epochs=args.epochs,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            #clip_grad_norm=args.clip_grad_norm,
            batch_size=args.batch_size,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            val_df=val_df,
            output_dir=args.output_dir,
            writer=writer,
            generation_config=generation_config,
            restore_from_epoch=restore_from_epoch,
        )

        if best_model_epoch == -1:
            print(f'No improvement found from training, loading original model')
            model, _, restored_from = load_model(args, resume_from_checkpoint=False)
            assert restored_from == -1
        elif isinstance(model, PeftModel):
            print(f'Loading best PeftModel model from {os.path.abspath(best_model_path)}')
            model, _, restored_from = load_model(args, resume_from_checkpoint=True, init_lora_path=best_model_path)
            assert restored_from == best_model_epoch
        else:
            print(f'Loading best full model from {os.path.abspath(best_model_path)}')
            best_model_state_dict = torch.load(best_model_path)
            model.load_state_dict(best_model_state_dict)

    val_loss, val_decoded, val_rouge_scores = eval(model, tokenizer, val_dataset, print_outputs=True, generation_config=generation_config)
    print('val loss', val_loss)
    for k, v in val_rouge_scores.items():
        print(f'val {k}', v)
    with open(os.path.join(args.output_dir, 'val_generated_predictions.txt'), 'w') as of:
        for generated in val_decoded:
            of.write(generated)
            of.write('\n')
    val_df['generated'] = val_decoded
    val_df_output = os.path.join(args.output_dir, 'val_generated_predictions.csv')
    if os.path.exists(val_df_output):
        print(f'Existing val output found at {val_df_output}, copying to {val_df_output + ".bak"}')
        os.rename(val_df_output, val_df_output + '.bak')
    val_df.to_csv(val_df_output)
    test_loss, test_decoded, test_rouge_scores = eval(model, tokenizer, test_dataset, print_outputs=False, generation_config=generation_config)
    print('test loss', test_loss)
    for k, v in test_rouge_scores.items():
        print(f'test {k}', v)
    with open(os.path.join(args.output_dir, 'test_generated_predictions.txt'), 'w') as of:
        for generated in test_decoded:
            of.write(generated)
            of.write('\n')
    test_df['generated'] = test_decoded
    test_df_output = os.path.join(args.output_dir, 'test_generated_predictions.csv')
    if os.path.exists(test_df_output):
        os.rename(test_df_output, test_df_output + '.bak')
        print(f'Existing test output found at {test_df_output}, copying to {test_df_output +".bak"}')
    test_df.to_csv(test_df_output)

    writer.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Learning arguments.')
 
    parser.add_argument('--model_name', dest='model_name', type=str, default="google/flan-t5-small",
                        help='name of model')

    parser.add_argument('--optimizer', default='adafactor', type=str, help='Optimizer (currently only adafactor and adamw allowed')
    parser.add_argument('--is_causal', action='store_true', default=False, help='is this a causal model (instead of seq2seq)?')
    parser.add_argument('--do_train', action='store_true', default=False, help='do we train this?')
    parser.add_argument('--resume_from_training', action='store_true', default=False, help='do we resume from our last training?')
    # lora parameters
    parser.add_argument('--peft_r', default=None, type=int, help='specify peft parameters (e.g. r=64)')
    parser.add_argument('--peft_alpha', default=None, type=int, help='specify peft parameters (e.g. alpha=16)')
    parser.add_argument('--peft_dropout', default=None, type=float, help='specify peft parameters (e.g. dropout=0.1)')

    parser.add_argument('--epochs', dest='epochs', type=int, default=10,
                        help='number of epochs')

    parser.add_argument('--lr', dest='learning_rate', type=float, default=1e-3,
                        help='learning rate for optimizer')

    parser.add_argument('--fp16', default=False, action='store_true',
                        help='Use fp16 or not?')
    parser.add_argument('--bf16', default=False, action='store_true',
                        help='Use bf16 or not?')
 
    parser.add_argument('--batch_size', dest='batch_size', type=int, default=16,
                        help='batch size (per device, not that this supports more than one device)')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=None, help='gradient accumulation')
    #parser.add_argument('--clip_grad_norm', type=float, default=None, help='gradient clipping')
    
    parser.add_argument('--limit_train_size', type=int, default=None,
                        help='use the first n of the training data')
    parser.add_argument('--limit_val_size', type=int, default=None,
                        help='use the first n of the val data')
    parser.add_argument('--limit_test_size', type=int, default=None,
                        help='use the first n of the test data')
 
    parser.add_argument('--init_weights', dest='init_weights_path', type=str, default=None,
                        help='path to initial weights')
    parser.add_argument('--init_lora', dest='init_lora_path', type=str, default=None,
                        help='path to initial weights')

    parser.add_argument('--input_field', required=True, type=str, help="What csv field to use for the input")

    parser.add_argument('--target_field', required=True, type=str, help="What csv field to use for the target")
    
    parser.add_argument('--data_path', required=True, type=str, help="Path to CSV ")
    parser.add_argument('--generation_config', required=False, type=str, default=None, help="Path to json")

    parser.add_argument('--output_dir', required=True, type=str, help="Path to outputs")

    args = parser.parse_args()

    main(args)
