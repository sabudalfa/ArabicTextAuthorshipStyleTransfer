import os.path
from abc import ABC, abstractmethod
import pandas as pd
from llm import *
from llm.llm_trainer import LLMTrainer
from simpleio import read_from_json_file, write_to_json_file, make_parent_dirs, append_to_json_list_file
from typing import *

GPUS_COUNT = {
    'gemma-2-9b-it': 1,
    'Phi-3.5-mini-instruct': 1,
    'Qwen2-7B-Instruct': 1,
    'Llama-2-7b-chat-hf': 1,
    'Meta-Llama-3-8B-Instruct': 1,
    'Meta-Llama-3.1-8B-Instruct': 1,
    'jais-30b-chat-v1': 2,
    'c4ai-command-r-v01': 2,
    'Phi-3.5-MoE-instruct': 2,
    'gemma-2-27b-it': 2,
    'Meta-Llama-3-70B-Instruct': 3,
    'Meta-Llama-3.1-70B-Instruct': 3,
    'Qwen2-72B-Instruct': 3,
}


class LLMJob(ABC):

    def __init__(self, gpus_count):
        self.gpus_count = gpus_count

    @abstractmethod
    def __call__(self):
        pass


class LLMTrainingJob(LLMJob):

    def __init__(self, llm_trainer: LLMTrainer, gpus_count=None):
        if not gpus_count:
            gpus_count = GPUS_COUNT[llm_trainer.llm_loader.name]
        self.llm_trainer = llm_trainer
        super().__init__(gpus_count)

    def __call__(self):
        self.llm_trainer()


class ExperimentDirInitializationJob(LLMJob):

    def __init__(
            self,
            job_information: Dict,
            test_samples: List[Tuple[str, str]],
            experiment_path: str,
    ):
        super().__init__(gpus_count=0)
        self.job_information = job_information
        self.test_samples = test_samples
        self.experiment_path = experiment_path

    def __call__(self):
        if os.path.exists(f'{self.experiment_path}/job.json'):
            return
        write_to_json_file(
            file_path=f'{self.experiment_path}/job.json',
            content=self.job_information,
        )
        write_to_json_file(
            file_path=f'{self.experiment_path}/prompts.json',
            content=[
                prompt
                for prompt, _ in self.test_samples
            ],
        )
        write_to_json_file(
            file_path=f'{self.experiment_path}/ground_truth_outputs.json',
            content=[
                output
                for _, output in self.test_samples
            ],
        )


class TextGenerationJob(LLMJob):

    def __init__(
            self,
            llm_loader: LLMLoader,
            experiment_path: str,
            use_chat_format: bool = False,
            gpus_count=None,
    ):
        if not gpus_count:
            gpus_count = GPUS_COUNT[llm_loader.name]
        super().__init__(gpus_count)
        self.llm_loader = llm_loader
        self.experiment_path = experiment_path
        self.use_chat_format = use_chat_format

    def __call__(self):
        output_file_path = f'{self.experiment_path}/predicted_outputs.json'
        if os.path.exists(output_file_path):
            return
        prompts = read_from_json_file(f'{self.experiment_path}/prompts.json')
        text_generator = TextGenerator.from_llm_loader(
            self.llm_loader,
            max_tokens_per_batch='auto',
        )
        if self.use_chat_format:
            predicted_outputs = text_generator.generate_chat_message(
                conversations=[
                    [
                        {
                            'role': 'user',
                            'content': prompt,
                        },
                    ]
                    for prompt in prompts
                ],
            )
        else:
            predicted_outputs = text_generator.generate(prompts)
        write_to_json_file(output_file_path, predicted_outputs)


class LLMEvaluationJob(LLMJob):

    def __init__(self, llm_evaluator: LLMEvaluator, experiment_path: str, gpus_count=1):
        super().__init__(gpus_count)
        self.experiment_path = experiment_path
        self.llm_evaluator = llm_evaluator

    def __call__(self):
        prompts = read_from_json_file(f'{self.experiment_path}/prompts.json')
        ground_truth_outputs = read_from_json_file(f'{self.experiment_path}/ground_truth_outputs.json')
        predicted_outputs = read_from_json_file(f'{self.experiment_path}/predicted_outputs.json')
        evaluation_dict = self.llm_evaluator(
            ground_truth=ground_truth_outputs,
            predictions=predicted_outputs,
            sources=prompts,
        )
        write_to_json_file(
            file_path=f'{self.experiment_path}/evaluation.json',
            content=evaluation_dict,
        )


class ResultsAggregationJob:

    def __init__(self, output_dir: str, job_paths: List[str]):
        self.output_dir = output_dir
        self.job_paths = job_paths

    def __call__(self):
        summary_dataframe = pd.DataFrame.from_records([
            {
                **read_from_json_file(f'{path}/job.json'),
                **read_from_json_file(f'{path}/evaluation.json'),
            }
            for path in self.job_paths
        ])
        output_file = f'{self.output_dir}/summary.xlsx'
        make_parent_dirs(output_file)
        summary_dataframe.to_excel(output_file)

        experiment_dataframe = pd.concat([
            pd.DataFrame({
                **read_from_json_file(f'{path}/job.json'),
                **{
                    'prompt': read_from_json_file(f'{path}/prompts.json'),
                    'ground_truth_output': read_from_json_file(f'{path}/ground_truth_outputs.json'),
                    'predicted_output': read_from_json_file(f'{path}/predicted_outputs.json'),
                },
            })
            for path in self.job_paths
        ])
        experiment_dataframe.to_excel(f'{self.output_dir}/experiment.xlsx')
