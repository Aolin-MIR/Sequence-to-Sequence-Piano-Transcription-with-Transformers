from torch.nn import parameter
from torch.optim import lr_scheduler, optimizer
from tfrecord.torch.dataset import TFRecordDataset
import math
import torch
from base_src.trainer_utils import EvalPrediction, EvalLoopOutput, denumpify_detensorize
from base_src.trainer_pt_utils import find_batch_size, nested_concat, nested_numpify, nested_truncate, IterableDatasetShard
from base_src.file_utils import is_torch_tpu_available
from base_src.deepspeed import deepspeed_init
from model_ours import MyModel
# from torch_preprocess import tokenize
import base_src as transformers
from base_src.trainer_seq2seq import Seq2SeqTrainer
from base_src import T5Model, T5Config, DataCollatorForSeq2Seq, PreTrainedTokenizer, BatchEncoding, Seq2SeqTrainingArguments
import numpy as np
from torch.utils.data import DataLoader
from typing import TYPE_CHECKING, Any, Dict, List, NamedTuple, Optional, Sequence, Tuple, Union
# import tensorflow as tf
# total_examples = 4345295
total_examples = 4435818
total_examples = 2388022
# total_examples = 3805390
train_path = "train.tfrecord"
valid_path = 'small_valid.tfrecord'
batch_size = 128
# seg_width=16
seg_width = 127
epochs = 40
nbins = 88*3
beams=32

class MyConfig(T5Config):
    def __init__(self,
                 use_dense=False,
                 use_position_embed=False,
                 nbins=88*3,
                 input_length=31,
                 **kwargs
                 ):
        self.use_dense=use_dense
        self.use_position_embed = use_position_embed
        self.input_length=input_length
        self.nbins=nbins
        super().__init__(
                         **kwargs)


def note_4f1_loss(evalPrediction):

    '''Calculate note-F1 score.  
    Returns
    -------
    dict    
    '''
    # print(24, evalPrediction.predictions.shape)
    total_pred=0
    total_true=0
    count =0
    for y_pred,y_true in zip(evalPrediction.predictions, evalPrediction.label_ids):
    
        assert y_true.ndim == 1
        assert y_pred.ndim == 1 or y_pred.ndim == 2

        if y_pred.ndim == 2:
            # print(64)
            y_pred = y_pred.argmax(dim=1)
        
        y_pred=y_pred.tolist()
        
        try:
            y_pred = y_pred[:y_pred.index(0)]
        except ValueError as e:
            pass
        
        y_pred=[((x-1)//88,(x-1)%88) for x in y_pred if x !=seg_width*88+1]

        total_pred+=len(y_pred)
        y_true=y_true.tolist()
        y_true = [x for x in y_true if x not in [0, seg_width*88+1]]
        total_true+=len(y_true)
        for i in y_true: 
            # if i ==1:
            #     break
            # if i == seg_width*88+1:
            #     continue
            relt_true = (i-1)//88
            note_true = (i-1)%88
            for j in y_pred[:]:
                if j[1] == note_true and j[0] in range(max(0,relt_true-5), relt_true+5):
                    count+=1
                    y_pred.remove(j)
                    break
    epsilon = 1e-7
    # print(57,total_true,total_pred)
    r = count/(total_true+epsilon)
    p = count/(total_pred+epsilon)

    f1 = 2 * (p*r) / (r + p + epsilon)
    print({'note_f1': f1, 'precision': p, 'recall': r})
    return {'note_f1':f1,'precision':p,'recall':r}



def parse_fn(features):
    # print(60, features['inputs_embeds'].shape,type(features['inputs_embeds']))
   
    features['inputs_embeds'] =np.frombuffer(
        features['inputs_embeds'], dtype=np.float32).reshape(seg_width, -1)
    if features['inputs_embeds'].shape[-1] not in [512, nbins]:
        print(features['inputs_embeds'].shape, features['labels'])
        assert 1==2
    # features['labels'] = torch.from_numpy(features['labels'])
    # features['inputs_embeds'] = torch.from_numpy(features['inputs_embeds'])
    # print(62, features['inputs_embeds'], features['inputs_embeds'].shape)
    return features


index_path = None
description = {"inputs_embeds": "byte", "labels": "int"}
train_dataset = TFRecordDataset(train_path, None, description,
                                shuffle_queue_size=512, transform=parse_fn)
valid_dataset = TFRecordDataset(valid_path, None, description,
                           transform=parse_fn)
# dataset = tf.data.TFRecordDataset(tfrecord_path)
# dataset =dataset.map(parse_fn)


class MyDataCollatorForSeq2Seq(DataCollatorForSeq2Seq):
    
    def __call__(self, features, return_tensors=None):

        if return_tensors is None:
            return_tensors = self.return_tensors
        labels = [feature["labels"]
                  for feature in features] if "labels" in features[0].keys() else None
        # We have to pad the labels before calling `tokenizer.pad` as this method won't pad them and needs them of the
        # same length to return tensors.
        if labels is not None:
            max_label_length = max(len(l) for l in labels)
            if self.pad_to_multiple_of is not None:
                max_label_length = (
                    (max_label_length + self.pad_to_multiple_of - 1)
                    // self.pad_to_multiple_of
                    * self.pad_to_multiple_of
                )

            padding_side = self.tokenizer.padding_side
            for feature in features:
                remainder = [self.label_pad_token_id] * \
                    (max_label_length - len(feature["labels"]))
                if isinstance(feature["labels"], list):
                    feature["labels"] = (
                        feature["labels"] +
                        remainder if padding_side == "right" else remainder +
                        feature["labels"]
                    )
                elif padding_side == "right":
                    feature["labels"] = np.concatenate(
                        [feature["labels"], remainder]).astype(np.int64)
                else:
                    feature["labels"] = np.concatenate(
                        [remainder, feature["labels"]]).astype(np.int64)

        features = self.tokenizer.pad(
            features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=return_tensors,
        )
        features['inputs_embeds'] = torch.from_numpy(
            np.array(features['inputs_embeds']))
        features['labels'] = torch.from_numpy(
            np.array(features['labels']))
        # print(126, features.keys(), type(features['inputs_embeds']), type(features['labels']))
        # prepare decoder_input_ids

        if self.model is not None and hasattr(self.model, "prepare_decoder_input_ids_from_labels"):
            decoder_input_ids = self.model.prepare_decoder_input_ids_from_labels(
                labels=features["labels"])
            features["decoder_input_ids"] = decoder_input_ids

        return features


class MyTokenizer(PreTrainedTokenizer):
    def pad(self,
            encoded_inputs,
            padding= True,
            max_length: Optional[int] = None,
            pad_to_multiple_of: Optional[int] = None,
            return_attention_mask: Optional[bool] = None,
            return_tensors= None,
            verbose: bool = True,):

        if isinstance(encoded_inputs, (list, tuple)) and isinstance(encoded_inputs[0], (dict, BatchEncoding)):
            encoded_inputs = {key: [
                example[key] for example in encoded_inputs] for key in encoded_inputs[0].keys()}

        return encoded_inputs
        

class MyTrainer(Seq2SeqTrainer):

    def prediction_step(
        self,
        model,
        inputs,
        prediction_loss_only,
        ignore_keys = None,
        ):
        if not self.args.predict_with_generate or prediction_loss_only:
            return super().prediction_step(
                model, inputs, prediction_loss_only=prediction_loss_only, ignore_keys=ignore_keys
            )
        # print(212,inputs.keys())
        has_labels = "labels" in inputs
        inputs = self._prepare_inputs(inputs)

        # XXX: adapt synced_gpus for fairscale as well
        gen_kwargs = {
            "max_length": self._max_length if self._max_length is not None else self.model.config.max_length,
            "num_beams": self._num_beams if self._num_beams is not None else self.model.config.num_beams,
            "synced_gpus": False,
        }

        # if self.tokenizer is not None:
        #     generation_inputs = {k: v for k, v in inputs.items(
        #     ) if k in self.tokenizer.model_input_names}
        #     # very ugly hack to make it work
        #     generation_inputs["input_ids"] = generation_inputs.pop(
        #         self.tokenizer.model_input_names[0])
        # else:
        generation_inputs = {"input_ids": None,"inputs_embeds":inputs['inputs_embeds']}

        generated_tokens = self.model.generate(
            **generation_inputs,
            **gen_kwargs,
        )
        # in case the batch is shorter than max length, the output should be padded
        if generated_tokens.shape[-1] < gen_kwargs["max_length"]:
            generated_tokens = self._pad_tensors_to_max_len(
                generated_tokens, gen_kwargs["max_length"])

        with torch.no_grad():
            with self.autocast_smart_context_manager():
                outputs = model(**inputs)
            if has_labels:
                if self.label_smoother is not None:
                    loss = self.label_smoother(
                        outputs, inputs["labels"]).mean().detach()
                else:
                    loss = (outputs["loss"] if isinstance(
                        outputs, dict) else outputs[0]).mean().detach()
            else:
                loss = None

        if self.args.prediction_loss_only:
            return (loss, None, None)

        if has_labels:
            labels = inputs["labels"]
            if labels.shape[-1] < gen_kwargs["max_length"]:
                labels = self._pad_tensors_to_max_len(
                    labels, gen_kwargs["max_length"])
        else:
            labels = None

        return (loss, generated_tokens, labels)
    
    def evaluation_loop(
        self,
        dataloader: DataLoader,
        description: str,
        prediction_loss_only: Optional[bool] = None,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "eval",
    ) :
        """
        Prediction/evaluation loop, shared by :obj:`Trainer.evaluate()` and :obj:`Trainer.predict()`.

        Works both with or without labels.
        """
        args = self.args

        prediction_loss_only = prediction_loss_only if prediction_loss_only is not None else args.prediction_loss_only

        # if eval is called w/o train init deepspeed here
        if args.deepspeed and not self.deepspeed:

            # XXX: eval doesn't have `resume_from_checkpoint` arg but we should be able to do eval
            # from the checkpoint eventually
            deepspeed_engine, _, _ = deepspeed_init(
                self, num_training_steps=0, resume_from_checkpoint=None, inference=True
            )
            self.model = deepspeed_engine.module
            self.model_wrapped = deepspeed_engine
            self.deepspeed = deepspeed_engine

        model = self._wrap_model(self.model, training=False)

        # if full fp16 or bf16 eval is wanted and this ``evaluation`` or ``predict`` isn't called
        # while ``train`` is running, cast it to the right dtype first and then put on device
        if not self.is_in_train:
            if args.fp16_full_eval:
                model = model.to(dtype=torch.float16, device=args.device)
            elif args.bf16_full_eval:
                model = model.to(dtype=torch.bfloat16, device=args.device)

        batch_size = dataloader.batch_size



        model.eval()

        self.callback_handler.eval_dataloader = dataloader
        # Do this before wrapping.
        eval_dataset = dataloader.dataset

        if is_torch_tpu_available():
            import torch_xla.distributed.parallel_loader as pl
            dataloader = pl.ParallelLoader(
                dataloader, [args.device]).per_device_loader(args.device)

        if args.past_index >= 0:
            self._past = None

        # Initialize containers
        # losses/preds/labels on GPU/TPU (accumulated for eval_accumulation_steps)
        losses_host = None
        preds_host = None
        labels_host = None
        # losses/preds/labels on CPU (final containers)
        all_losses = None
        all_preds = None
        all_labels = None
        # Will be useful when we have an iterable dataset so don't know its length.

        observed_num_examples = 0
        # Main evaluation loop
        for step, inputs in enumerate(dataloader):
            # Update the observed num examples
            observed_batch_size = find_batch_size(inputs)
            if observed_batch_size is not None:
                observed_num_examples += observed_batch_size
                # For batch samplers, batch_size is not known by the dataloader in advance.
                if batch_size is None:
                    batch_size = observed_batch_size

            # Prediction step
            loss, logits, labels = self.prediction_step(
                model, inputs, prediction_loss_only, ignore_keys=ignore_keys)
            # print(249,type(logits),logits.shape,logits)
            if type (logits)==type((1,0)):
                logits=torch.argmax(logits[0],-1)
            # print(logits.shape,logits)
            # Update containers on host
            if loss is not None:

                losses = self._nested_gather(loss.repeat(batch_size))
                losses_host = losses if losses_host is None else torch.cat(
                    (losses_host, losses), dim=0)
            if logits is not None:
                # print(360,logits)
                logits = self._pad_across_processes(logits)
                logits = self._nested_gather(logits)
                # print(363,logits.shape)
                #len(logits)=2
                # print(2337,logits[0].shape,logits[1].shape)
                preds_host = logits if preds_host is None else nested_concat(
                    preds_host, logits, padding_index=-100)
                # print(2338,len(preds_host),preds_host[0].device)
            if labels is not None:
                labels = self._pad_across_processes(labels)
                labels = self._nested_gather(labels)
                labels_host = labels if labels_host is None else nested_concat(
                    labels_host, labels, padding_index=-100)
            self.control = self.callback_handler.on_prediction_step(
                args, self.state, self.control)

            # Gather all tensors and put them back on the CPU if we have done enough accumulation steps.
            if args.eval_accumulation_steps is not None and (step + 1) % args.eval_accumulation_steps == 0:
                if losses_host is not None:
                    losses = nested_numpify(losses_host)
                    all_losses = losses if all_losses is None else np.concatenate(
                        (all_losses, losses), axis=0)
                if preds_host is not None:
                    logits = nested_numpify(preds_host)
                    all_preds = logits if all_preds is None else nested_concat(
                        all_preds, logits, padding_index=-100)
                if labels_host is not None:
                    labels = nested_numpify(labels_host)
                    all_labels = (
                        labels if all_labels is None else nested_concat(
                            all_labels, labels, padding_index=-100)
                    )

                # Set back to None to begin a new accumulation
                losses_host, preds_host, labels_host = None, None, None

        if args.past_index and hasattr(self, "_past"):
            # Clean the state at the end of the evaluation loop
            delattr(self, "_past")

        # Gather all remaining tensors and put them back on the CPU
        if losses_host is not None:
            losses = nested_numpify(losses_host)
            all_losses = losses if all_losses is None else np.concatenate(
                (all_losses, losses), axis=0)
        if preds_host is not None:
            logits = nested_numpify(preds_host)
            all_preds = logits if all_preds is None else nested_concat(
                all_preds, logits, padding_index=-100)
        if labels_host is not None:
            labels = nested_numpify(labels_host)
            all_labels = labels if all_labels is None else nested_concat(
                all_labels, labels, padding_index=-100)

        # Number of samples
        if not isinstance(eval_dataset, torch.utils.data.IterableDataset):
            num_samples = len(eval_dataset)
        # The instance check is weird and does not actually check for the type, but whether the dataset has the right
        # methods. Therefore we need to make sure it also has the attribute.
        elif isinstance(eval_dataset, IterableDatasetShard) and hasattr(eval_dataset, "num_examples"):
            num_samples = eval_dataset.num_examples
        else:
            num_samples = observed_num_examples

        # Number of losses has been rounded to a multiple of batch_size and in a distributed training, the number of
        # samplers has been rounded to a multiple of batch_size, so we truncate.
        # print(2390)
        if all_losses is not None:
            all_losses = all_losses[:num_samples]
        if all_preds is not None:
            all_preds = nested_truncate(all_preds, num_samples)
        if all_labels is not None:
            all_labels = nested_truncate(all_labels, num_samples)

        # Metrics!
        if self.compute_metrics is not None and all_preds is not None and all_labels is not None:
            metrics = self.compute_metrics(EvalPrediction(
                predictions=all_preds, label_ids=all_labels))
        else:
            metrics = {}

        # To be JSON-serializable, we need to remove numpy types or zero-d tensors
        metrics = denumpify_detensorize(metrics)

        if all_losses is not None:
            metrics[f"{metric_key_prefix}_loss"] = all_losses.mean().item()

        # Prefix all keys with metric_key_prefix + '_'
        for key in list(metrics.keys()):
            if not key.startswith(f"{metric_key_prefix}_"):
                metrics[f"{metric_key_prefix}_{key}"] = metrics.pop(key)

        return EvalLoopOutput(predictions=all_preds, label_ids=all_labels, metrics=metrics, num_samples=num_samples)

    def get_train_dataloader(self) -> DataLoader:
        """
        Returns the training :class:`~torch.utils.data.DataLoader`.
        Will use no sampler if :obj:`self.train_dataset` does not implement :obj:`__len__`, a random sampler (adapted
        to distributed training if necessary) otherwise.
        Subclass and override this method if you want to inject some custom behavior.
        """
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")

        train_dataset = self.train_dataset

        train_sampler = self._get_train_sampler()

        return DataLoader(
            train_dataset,
            # shuffle=True,
            batch_size=self.args.train_batch_size,
            sampler=train_sampler,
            collate_fn=self.data_collator,
            drop_last=self.args.dataloader_drop_last,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory,
        )

    def compute_loss(self, model, inputs, return_outputs=False):
        """
        How the loss is computed by Trainer. By default, all models return the loss in the first element.
        Subclass and override for custom behavior.
        """
        # print(150, type(inputs['inputs_embeds']))
        if self.label_smoother is not None and "labels" in inputs:
            labels = inputs.pop("labels")

        else:
            labels = None


        # for x in inputs['labels']:
        #     for y in x:
        #         if y>= 16*88:
        #             print('%#%Z@#%@R%@#$@#$@#$@#$@#$@#$@#$',x)

        outputs = model(**inputs)
        # Save past state if it exists
        # TODO: this needs to be fixed and made cleaner later.
        if self.args.past_index >= 0:
            self._past = outputs[self.args.past_index]

        if labels is not None:
            loss = self.label_smoother(outputs, labels)
        else:
            # We don't use .loss here since the model may return tuples instead of ModelOutput.
            loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]

        return (loss, outputs) if return_outputs else loss


config = MyConfig(vocab_size=seg_width*88+2,input_length=seg_width,use_position_embed=True, use_dense=False,d_model=512, d_kv=64, d_ff=1024, num_layers=8, num_decoder_layers=None, num_heads=8, relative_attention_num_buckets=32, dropout_rate=0.1,
                  layer_norm_epsilon=1e-06, initializer_factor=1.0, feed_forward_proj='relu', is_encoder_decoder=True, use_cache=True, 
                  bos_token_id=seg_width*88+1,
                  pad_token_id=seg_width*88+1, eos_token_id=0, decoder_start_token_id=seg_width*88+1)
model = MyModel(config)

args = Seq2SeqTrainingArguments(output_dir='./output', overwrite_output_dir=False, do_train=True, do_eval=True, do_predict=False, evaluation_strategy='steps', prediction_loss_only=False, per_device_train_batch_size=batch_size, per_device_eval_batch_size=int(batch_size/beams), per_gpu_train_batch_size=batch_size, per_gpu_eval_batch_size=int(batch_size/beams), gradient_accumulation_steps=1, eval_accumulation_steps=200, max_grad_norm=1.0,  max_steps=math.ceil(total_examples*0.8/batch_size)*epochs,  log_level='passive', log_level_replica='passive', log_on_each_node=True,
                                logging_dir=None, logging_strategy='steps', logging_first_step=False, 
                                logging_steps=math.ceil(total_examples*0.8/batch_size), 
                                logging_nan_inf_filter=True, save_strategy='steps', save_steps=math.ceil(total_examples*0.8/batch_size), save_total_limit=10, save_on_each_node=False, no_cuda=False, seed=42, fp16=False, fp16_opt_level='O1', fp16_backend='auto', fp16_full_eval=False, local_rank=- 1, xpu_backend=None, tpu_num_cores=None, tpu_metrics_debug=False, dataloader_drop_last=False, 
                                eval_steps=math.ceil(total_examples*0.8/batch_size),
                                 dataloader_num_workers=0, generation_max_length=seg_width*2, 
                                 predict_with_generate=True, 
                                 generation_num_beams=beams
                                 )

optimizer=transformers.AdamW(model.parameters(), 1e-4)
lr_scheduler = transformers.get_constant_schedule(optimizer)

trainer = MyTrainer(model, train_dataset=train_dataset,
                  eval_dataset=valid_dataset,    
                         args=args,
                    data_collator=MyDataCollatorForSeq2Seq(tokenizer=MyTokenizer(padding_side='right') ), tokenizer=None,
                    compute_metrics=note_4f1_loss,
                    optimizers=(optimizer, lr_scheduler)
)
                   
trainer.train()
