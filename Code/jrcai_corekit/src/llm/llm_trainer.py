from os.path import exists
from typing import *
from peft import *
import shutil
from transformers import *
from datasets import *

from simpleio import write_to_file
from .llm_loader import LLMLoader


class _MyTrainerCallback(TrainerCallback):

    def __init__(self, patience=1, save_best_model=True):
        self.patience = patience
        self.patience_counter = 0
        self.best_loss = float('inf')
        self.save_best_model = save_best_model

    def on_evaluate(self, args, state, control, **kwargs):
        print(kwargs['metrics'])
        eval_loss = kwargs['metrics']['eval_loss']
        if eval_loss <= self.best_loss:
            self.best_loss = eval_loss
            self.patience_counter = 0
            if self.save_best_model:
                shutil.rmtree(args.output_dir, ignore_errors=True)
                kwargs["model"].save_pretrained(args.output_dir, safe_serialization=False)
        else:
            self.patience_counter += 1
        if self.patience_counter >= self.patience:
            control.should_training_stop = True


def _to_causal_lm_dataset(
        samples: List[Tuple[str, str]],
        tokenizer: PreTrainedTokenizer,
):
    mapped_samples = []
    for source, target in samples:
        prompt_tokens = tokenizer.encode(
            source,
            add_special_tokens=False,
        )
        output_tokens = tokenizer.encode(
            target + tokenizer.eos_token,
            add_special_tokens=False,
        )
        mapped_samples.append({
            'input_ids': prompt_tokens + output_tokens,
            'attention_mask': [1] * (len(prompt_tokens) + len(output_tokens)),
            'labels': [-100] * len(prompt_tokens) + output_tokens
        })
    return Dataset.from_list(mapped_samples)


def train_llm(
        model,
        tokenizer,
        train_samples: List[Tuple[str, str]],
        eval_samples: List[Tuple[str, str]],
        peft_config: Optional[PeftConfig],
        epochs_count,
        learning_rate: float,
        train_batch_size: int,
        eval_batch_size: int,
        output_dir: str,
        save_best_model: bool = True,
):
    if peft_config is not None:
        model = get_peft_model(model, peft_config)
    train_dataset, eval_dataset = (
        _to_causal_lm_dataset(train_samples, tokenizer),
        _to_causal_lm_dataset(eval_samples, tokenizer),
    )
    callback = _MyTrainerCallback(
        patience=5,
        save_best_model=save_best_model,
    )
    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        args=TrainingArguments(
            output_dir=output_dir,
            do_eval=True,
            eval_steps=10_000,
            evaluation_strategy='steps',
            save_strategy='no',
            learning_rate=learning_rate,
            per_device_train_batch_size=train_batch_size,
            per_device_eval_batch_size=eval_batch_size,
            num_train_epochs=epochs_count,
            disable_tqdm=False,
            weight_decay=0.01,
            remove_unused_columns=False,
            bf16=True,
            report_to="none",
        ),
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=DataCollatorForSeq2Seq(tokenizer),
        callbacks=[callback],
        # compute_metrics=compute_metrics,
    )
    trainer.train()
    return callback.best_loss


class LLMTrainer:

    def __init__(
            self,
            llm_loader: LLMLoader,
            train_samples: List[Tuple[str, str]],
            eval_samples: List[Tuple[str, str]],
            peft_config: Optional[PeftConfig],
            epochs_count,
            learning_rate: float,
            train_batch_size: int,
            eval_batch_size: int,
            output_dir: str,
            save_best_model: bool = True,
    ):
        self.llm_loader = llm_loader
        self.train_samples = train_samples
        self.eval_samples = eval_samples
        self.peft_config = peft_config
        self.epochs_count = epochs_count
        self.learning_rate = learning_rate
        self.train_batch_size = train_batch_size
        self.eval_batch_size = eval_batch_size
        self.output_dir = output_dir
        self.save_best_model = save_best_model

    def __call__(self):
        done_file_path = f"{self.output_dir}/done"
        if exists(done_file_path):
            return
        model, tokenizer, _ = self.llm_loader()
        print(model)
        train_llm(
            model=model,
            tokenizer=tokenizer,
            train_samples=self.train_samples,
            eval_samples=self.eval_samples,
            peft_config=self.peft_config,
            epochs_count=self.epochs_count,
            learning_rate=self.learning_rate,
            train_batch_size=self.train_batch_size,
            eval_batch_size=self.eval_batch_size,
            output_dir=self.output_dir,
            save_best_model=self.save_best_model,
        )
        write_to_file(file_path=done_file_path, content="")
