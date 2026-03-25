import ray
from .llm_job import LLMJob
from typing import *

@ray.remote
def run_job_0_gpu(job):
    job()


@ray.remote(num_gpus=1)
def run_job_1_gpu(job):
    job()


@ray.remote(num_gpus=2)
def run_job_2_gpu(job):
    job()


@ray.remote(num_gpus=3)
def run_job_3_gpu(job):
    job()


@ray.remote(num_gpus=4)
def run_job_4_gpu(job):
    job()


@ray.remote(num_gpus=5)
def run_job_5_gpu(job):
    job()


@ray.remote(num_gpus=6)
def run_job_6_gpu(job):
    job()


@ray.remote(num_gpus=7)
def run_job_7_gpu(job):
    job()


@ray.remote(num_gpus=8)
def run_job_8_gpu(job):
    job()


def _get_ray_job(job, gpus_count):
    if gpus_count == 0:
        return run_job_0_gpu.remote(job)
    if gpus_count == 1:
        return run_job_1_gpu.remote(job)
    if gpus_count == 2:
        return run_job_2_gpu.remote(job)
    if gpus_count == 3:
        return run_job_3_gpu.remote(job)
    if gpus_count == 4:
        return run_job_4_gpu.remote(job)
    if gpus_count == 5:
        return run_job_5_gpu.remote(job)
    if gpus_count == 6:
        return run_job_6_gpu.remote(job)
    if gpus_count == 7:
        return run_job_7_gpu.remote(job)
    if gpus_count == 8:
        return run_job_8_gpu.remote(job)


def run_in_parallel(job_and_gpus_count_tuples):
    ray.get([
        _get_ray_job(job, gpus_count)
        for job, gpus_count in job_and_gpus_count_tuples
    ])

def run_llm_jobs_in_parallel(llm_jobs: List[LLMJob]):
    run_in_parallel([
        (llm_job, llm_job.gpus_count)
        for llm_job in llm_jobs
    ])
