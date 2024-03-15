"""
API REQUEST PARALLEL PROCESSOR

Using the OpenAI API to process lots of text quickly takes some care.
If you trickle in a million API requests one by one, they'll take days to complete.
If you flood a million API requests in parallel, they'll exceed the rate limits and fail with errors.
To maximize throughput, parallel requests need to be throttled to stay under rate limits.

This script parallelizes requests to the OpenAI API while throttling to stay under rate limits.

Features:
- Streams requests from file, to avoid running out of memory for giant jobs
- Makes requests concurrently, to maximize throughput
- Throttles request and token usage, to stay under rate limits
- Retries failed requests up to {max_attempts} times, to avoid missing data
- Logs errors, to diagnose problems with requests

Example command to call script:
```
python examples/api_request_parallel_processor.py \
  --requests_filepath examples/data/example_requests_to_parallel_process.jsonl \
  --save_filepath examples/data/example_requests_to_parallel_process_results.jsonl \
  --request_url https://api.openai.com/v1/embeddings \
  --max_requests_per_minute 1500 \
  --max_tokens_per_minute 6250000 \
  --token_encoding_name cl100k_base \
  --max_attempts 5 \
  --logging_level 20
```

Inputs:
- requests_filepath : str
    - path to the file containing the requests to be processed
    - file should be a jsonl file, where each line is a json object with API parameters and an optional metadata field
    - e.g., {"model": "text-embedding-ada-002", "input": "embed me", "metadata": {"row_id": 1}}
    - as with all jsonl files, take care that newlines in the content are properly escaped (json.dumps does this automatically)
    - an example file is provided at examples/data/example_requests_to_parallel_process.jsonl
    - the code to generate the example file is appended to the bottom of this script
- save_filepath : str, optional
    - path to the file where the results will be saved
    - file will be a jsonl file, where each line is an array with the original request plus the API response
    - e.g., [{"model": "text-embedding-ada-002", "input": "embed me"}, {...}]
    - if omitted, results will be saved to {requests_filename}_results.jsonl
- request_url : str, optional
    - URL of the API endpoint to call
    - if omitted, will default to "https://api.openai.com/v1/embeddings"
- api_key : str, optional
    - API key to use
    - if omitted, the script will attempt to read it from an environment variable {os.getenv("OPENAI_API_KEY")}
- max_requests_per_minute : float, optional
    - target number of requests to make per minute (will make less if limited by tokens)
    - leave headroom by setting this to 50% or 75% of your limit
    - if requests are limiting you, try batching multiple embeddings or completions into one request
    - if omitted, will default to 1,500
- max_tokens_per_minute : float, optional
    - target number of tokens to use per minute (will use less if limited by requests)
    - leave headroom by setting this to 50% or 75% of your limit
    - if omitted, will default to 125,000
- token_encoding_name : str, optional
    - name of the token encoding used, as defined in the `tiktoken` package
    - if omitted, will default to "cl100k_base" (used by `text-embedding-ada-002`)
- max_attempts : int, optional
    - number of times to retry a failed request before giving up
    - if omitted, will default to 5
- logging_level : int, optional
    - level of logging to use; higher numbers will log fewer messages
    - 40 = ERROR; will log only when requests fail after all retries
    - 30 = WARNING; will log when requests his rate limits or other errors
    - 20 = INFO; will log when requests start and the status at finish
    - 10 = DEBUG; will log various things as the loop runs to see when they occur
    - if omitted, will default to 20 (INFO).

The script is structured as follows:
    - Imports
    - Define main()
        - Initialize things
        - In main loop:
            - Get next request if one is not already waiting for capacity
            - Update available token & request capacity
            - If enough capacity available, call API
            - The loop pauses if a rate limit error is hit
            - The loop breaks when no tasks remain
    - Define dataclasses
        - StatusTracker (stores script metadata counters; only one instance is created)
        - APIRequest (stores API inputs, outputs, metadata; one method to call API)
    - Define functions
        - api_endpoint_from_url (extracts API endpoint from request URL)
        - append_to_jsonl (writes to results file)
        - num_tokens_consumed_from_request (bigger function to infer token usage from request)
        - task_id_generator_function (yields 1, 2, 3, ...)
    - Run main()
"""

# imports
import aiohttp  # for making API calls concurrently
import argparse  # for running script from command line
import asyncio  # for running API calls concurrently
import json  # for saving results to a jsonl file
import logging  # for logging rate limit warnings and other messages
import os  # for reading API key
import re  # for matching endpoint from request URL
import tiktoken  # for counting tokens
import time  # for sleeping after rate limit is hit
from dataclasses import (
    dataclass,
    field,
)  # for storing API inputs, outputs, and metadata


# ========================== Begin of added ==========================

from typing import Dict, List, Any
from collections import Counter

import openai
from tqdm import tqdm

from stefutil import get_logger, pl, fmt_delta, describe, Timer, stem

__all__ = ['parallel_api_call']

# List of modifications
# replace logging with custom logger, custom logging messages

_logger = get_logger('OpenAI Parallel Req')


# **Update Jan. 1st. 2024**
# somehow the original way of API calls in this script via `post` produced
#   inconsistent results compared to OpenAI playground,
# I will modify using the most recent `openai` v1 package,
#   which from standalone usage is consistent with OpenAI playground.

# USE_OPENAI = False
USE_OPENAI = True
# ========================== End of added ==========================


async def parallel_api_call(
        # ========================== Begin of modified ==========================
        # requests_filepath: str,
        # save_filepath: str,
        # request_url: str,
        # api_key: str,
        # max_requests_per_minute: float,
        # max_tokens_per_minute: float,
        # token_encoding_name: str,
        # max_attempts: int,
        # logging_level: int,
        requests_filepath: str = None,
        save_filepath: str = None,
        output_directory: str = None,
        request_url: str = 'https://api.openai.com/v1/embeddings',
        api_key: str = None,
        max_requests_per_minute: float = 3_000 * 0.5,
        max_tokens_per_minute: float = 250_000 * 0.5,
        token_encoding_name: str = 'cl100k_base',
        max_attempts: int = 5,
        logging_level: int = logging.INFO,
        logger: logging.Logger = _logger,
        n_prompt: int = None,
        completion_filenames: List[str] = None,
        is_chat_model: bool = None,
        with_tqdm: bool = True,
        timeout: int = None,
        logprobs: bool = False,
        # ========================== End of modified ==========================
):
    """Processes API requests in parallel, throttling to stay under rate limits."""
    # constants
    seconds_to_pause_after_rate_limit_error = 15
    seconds_to_sleep_each_loop = (
        0.001  # 1 ms limits max throughput to 1,000 requests per second
    )

    # ========================== Begin of modified ==========================
    if save_filepath is None:
        save_filepath = requests_filepath.replace('.jsonl', '_results.jsonl')
    d_log = {
        'requests-filepath': requests_filepath, 'save-filepath': save_filepath, 'output-directory': output_directory,
        'request-url': request_url, 'token-encoding-name': token_encoding_name,
        'max-requests-per-minute': round(max_requests_per_minute), 'max-tokens-per-minute': round(max_tokens_per_minute),
        'max-attempts': 'inf' if max_attempts == 2 ** 31 - 1 else max_attempts, 'timeout': timeout,
        'is-chat-model': is_chat_model, '#-prompt': n_prompt, 'completion-filenames': completion_filenames,
    }
    logger.info(f'Parallel processing {pl.i(n_prompt)} requests w/ {pl.i(d_log, indent=1)}... ')

    if output_directory:
        requests_filepath = os.path.join(output_directory, requests_filepath)
        save_filepath = os.path.join(output_directory, save_filepath)

    # for separately saving completions to individual files
    base_dir_nm = os.path.dirname(save_filepath)
    completion_filenames = [os.path.join(base_dir_nm, fnm) for fnm in completion_filenames]
    pbar = tqdm(desc='Parallel processing', unit='req', total=n_prompt) if with_tqdm else None

    client = None
    if USE_OPENAI:
        client = openai.AsyncClient(api_key=api_key)
        httpx_logger: logging.Logger = logging.getLogger('httpx')
        httpx_logger.setLevel(logging.WARNING)  # hides many `HTTP Request: POST “HTTP/1.1 200 OK”` logging output that floods the terminal
    # ========================== End of modified ==========================

    # initialize logging
    logging.basicConfig(level=logging_level)
    logger.setLevel(logging_level)
    logger.debug(f"Logging initialized at level {logging_level}")

    # infer API endpoint and construct request header
    api_endpoint = api_endpoint_from_url(request_url)
    request_header = {"Authorization": f"Bearer {api_key}"}
    # use api-key header for Azure deployments
    if '/deployments' in request_url:
        request_header = {"api-key": f"{api_key}"}

    # initialize trackers
    queue_of_requests_to_retry = asyncio.Queue()
    task_id_generator = (
        task_id_generator_function()
    )  # generates integer IDs of 1, 2, 3, ...
    status_tracker = (
        StatusTracker()
    )  # single instance to track a collection of variables
    stats_tracker = StatsTracker()
    next_request = None  # variable to hold the next request to call

    # initialize available capacity counts
    available_request_capacity = max_requests_per_minute
    available_token_capacity = max_tokens_per_minute
    last_update_time = time.time()

    # initialize flags
    file_not_finished = True  # after file is empty, we'll skip reading it
    logger.debug(f"Initialization complete.")

    # initialize file reading
    with open(requests_filepath) as file:
        # `requests` will provide requests one at a time
        requests = file.__iter__()
        logger.debug(f"File opened. Entering main loop")
        async with aiohttp.ClientSession() as session:  # Initialize ClientSession here
            while True:
                # get next request (if one is not already waiting for capacity)
                if next_request is None:
                    if not queue_of_requests_to_retry.empty():
                        next_request = queue_of_requests_to_retry.get_nowait()
                        logger.debug(
                            f"Retrying request {next_request.task_id}: {next_request}"
                        )
                    elif file_not_finished:
                        try:
                            # get new request
                            request_json = json.loads(next(requests))
                            next_request = APIRequest(
                                task_id=next(task_id_generator),
                                request_json=request_json,
                                token_consumption=num_tokens_consumed_from_request(
                                    request_json, api_endpoint, token_encoding_name
                                ),
                                attempts_left=max_attempts,
                                metadata=request_json.pop("metadata", None),
                                client=client,
                                total_n_request=n_prompt,
                                is_chat_model=is_chat_model,
                                logprobs=logprobs,
                                pbar=pbar
                            )
                            status_tracker.num_tasks_started += 1
                            status_tracker.num_tasks_in_progress += 1
                            logger.debug(
                                f"Reading request {next_request.task_id}: {next_request}"
                            )
                        except StopIteration:
                            # if file runs out, set flag to stop reading it
                            logger.debug("Read file exhausted")
                            file_not_finished = False

                # update available capacity
                current_time = time.time()
                seconds_since_update = current_time - last_update_time
                available_request_capacity = min(
                    available_request_capacity
                    + max_requests_per_minute * seconds_since_update / 60.0,
                    max_requests_per_minute,
                )
                available_token_capacity = min(
                    available_token_capacity
                    + max_tokens_per_minute * seconds_since_update / 60.0,
                    max_tokens_per_minute,
                )
                last_update_time = current_time

                # if enough capacity available, call API
                if next_request:
                    next_request_tokens = next_request.token_consumption
                    if (
                        available_request_capacity >= 1
                        and available_token_capacity >= next_request_tokens
                    ):
                        # update counters
                        available_request_capacity -= 1
                        available_token_capacity -= next_request_tokens
                        next_request.attempts_left -= 1

                        # call API
                        d_complete = asyncio.create_task(
                            next_request.call_api(
                                session=session,
                                request_url=request_url,
                                request_header=request_header,
                                retry_queue=queue_of_requests_to_retry,
                                save_filepath=save_filepath,
                                status_tracker=status_tracker,
                                stats_tracker=stats_tracker,
                                logger=logger,
                                # ========================== Begin of added ==========================
                                completion_filepath=completion_filenames[next_request.task_id] if completion_filenames else None,
                                timeout=timeout
                                # ========================== End of added ==========================
                            )
                        )
                        # ========================== Begin of added ==========================
                        # complete_log['#res-token'].append((await d_complete)['#res-token'])
                        # complete_log['finish-reason'].append((await d_complete)['finish-reason'])
                        # ========================== End of added ==========================
                        next_request = None  # reset next_request to empty

                # if all tasks are finished, break
                if status_tracker.num_tasks_in_progress == 0:
                    break

                # main loop sleeps briefly so concurrent tasks can run
                await asyncio.sleep(seconds_to_sleep_each_loop)

                # if a rate limit error was hit recently, pause to cool down
                seconds_since_rate_limit_error = (
                    time.time() - status_tracker.time_of_last_rate_limit_error
                )
                if (
                    seconds_since_rate_limit_error
                    < seconds_to_pause_after_rate_limit_error
                ):
                    remaining_seconds_to_pause = (
                        seconds_to_pause_after_rate_limit_error
                        - seconds_since_rate_limit_error
                    )
                    await asyncio.sleep(remaining_seconds_to_pause)
                    # ^e.g., if pause is 15 seconds and final limit was hit 5 seconds ago
                    tm = time.ctime(status_tracker.time_of_last_rate_limit_error + seconds_to_pause_after_rate_limit_error)
                    logger.warning(f"Pausing to cool down until {pl.i(tm)}")

        # after finishing, log final status
        logger.info(
            f"""Parallel processing on {pl.i(n_prompt)} requests complete. Results saved to {pl.i(stem(save_filepath))}"""
        )
        if status_tracker.num_tasks_failed > 0:
            logger.warning(f"{pl.i(status_tracker.num_tasks_failed)} / {pl.i(status_tracker.num_tasks_started)} requests failed. "
                           f"Errors logged to {pl.i(save_filepath)}.")
        if status_tracker.num_rate_limit_errors > 0:
            logger.warning(f"{pl.i(status_tracker.num_rate_limit_errors)} rate limit errors received. Consider running at a lower rate.")
        # ========================== Begin of added ==========================
        sum_ = stats_tracker.to_summary()
        logger.info(f'Completion statistics: {pl.i(sum_, indent=1)}')
        end_for_len = sum_['finish-reason'].get('length', 0)
        if end_for_len > 0:
            logger.warning(f'{pl.i(end_for_len)} completion(s) finished due to token limit, consider increasing {"max-tokens"}')
        # return sum_
        # ========================== End of added ==========================


# dataclasses


@dataclass
class StatusTracker:
    """Stores metadata about the script's progress. Only one instance is created."""

    num_tasks_started: int = 0
    num_tasks_in_progress: int = 0  # script ends when this reaches 0
    num_tasks_succeeded: int = 0
    num_tasks_failed: int = 0
    num_rate_limit_errors: int = 0
    num_api_errors: int = 0  # excluding rate limit errors, counted above
    num_other_errors: int = 0
    time_of_last_rate_limit_error: int = 0  # used to cool off after hitting rate limits


# ========================== Begin of added ==========================
@dataclass
class StatsTracker:
    n_res_token: List[int] = field(default_factory=list)
    finish_reason: List[str] = field(default_factory=list)

    def log_completion(self, res: Dict[str, Any]):
        self.n_res_token.append(res['#res-token'])
        self.finish_reason.append(res['finish-reason'])

    def to_summary(self) -> Dict[str, Any]:
        return {
            '#res-token': describe(self.n_res_token, round_dec=1),
            'finish-reason': Counter(self.finish_reason)
        }
# ========================== End of added ==========================


@dataclass
class APIRequest:
    """Stores an API request's inputs, outputs, and other metadata. Contains a method to make an API call."""

    task_id: int
    request_json: dict
    token_consumption: int
    attempts_left: int
    metadata: dict
    client: openai.AsyncClient = None
    result: list = field(default_factory=list)
    total_n_request: int = None
    is_chat_model: bool = None
    logprobs: bool = False
    pbar: tqdm = None
    n_failed: int = 0

    @property
    def log_ordinal(self) -> str:  # 0-indexing => 1-indexing
        return f'#{pl.i(self.task_id+1)}/{pl.i(self.total_n_request)}'

    async def call_api(
            self,
            session: aiohttp.ClientSession,
            request_url: str,
            request_header: dict,
            retry_queue: asyncio.Queue,
            save_filepath: str,
            status_tracker: StatusTracker,
            stats_tracker: StatsTracker,
            logger: logging.Logger = _logger,
            # ========================== Begin of added ==========================
            completion_filepath: str = None,
            timeout: int = None
            # ========================== End of added ==========================
    ):
        """Calls the OpenAI API and saves results."""
        msg = f"Starting request {self.log_ordinal}"
        # exponential backoff based on `n_failed`
        timeout = timeout or 120
        timeout = round(timeout * (1.25 ** self.n_failed))
        timeout = min(timeout, 600)  # max timeout 10 min
        if self.n_failed:
            d_log = {'#failed': self.n_failed, 'timeout (s)': timeout, 'timeout (readable)': fmt_delta(timeout)}
            msg += f' w/ {pl.i(d_log)}'
        logger.info(msg)
        error = None
        tm = Timer()
        try:
            # ========================== Begin of modified ==========================
            if not USE_OPENAI:
                assert self.client is None  # sanity check
                async with session.post(
                    url=request_url, headers=request_header, json=self.request_json, timeout=timeout
                ) as response:
                    response = await response.json()
            else:
                assert self.client is not None  # sanity check
                cln = self.client.with_options(timeout=timeout)
                response = await cln.chat.completions.create(**self.request_json)
                response = response.model_dump()  # `ChatCompletion` => `dict`
            # ========================== End of modified ==========================

            if "error" in response:
                d_log = dict(error=response['error'], duration=tm.end())
                logger.warning(f"Request {self.log_ordinal} failed w/ {pl.i(d_log)}")
                status_tracker.num_api_errors += 1
                error = response
                if "Rate limit" in response["error"].get("message", ""):
                    status_tracker.time_of_last_rate_limit_error = time.time()
                    status_tracker.num_rate_limit_errors += 1
                    status_tracker.num_api_errors -= (
                        1  # rate limit errors are counted separately
                    )

        except (
            Exception
        ) as e:  # catching naked exceptions is bad practice, but in this case we'll log & save them
            # if isinstance(e, asyncio.TimeoutError):  # check if the exception is TimeoutError
            logger.warning(f"Request {self.log_ordinal} failed with {pl.i(e.__class__.__qualname__)} Exception [{pl.i(e)}] "
                           f"& duration {pl.i(tm.end())}")
            status_tracker.num_other_errors += 1
            error = e
        if error:
            self.result.append(error)
            if self.attempts_left:
                retry_queue.put_nowait(self)
            else:
                logger.error(f"Request {self.log_ordinal} w/ json {pl.i(self.request_json)} failed after all attempts. "
                             f"Saving errors: {pl.i(self.result)}")
                data = (
                    [self.request_json, [str(e) for e in self.result], self.metadata]
                    if self.metadata
                    else [self.request_json, [str(e) for e in self.result]]
                )
                append_to_jsonl(data, save_filepath)
                status_tracker.num_tasks_in_progress -= 1
                status_tracker.num_tasks_failed += 1
            self.n_failed += 1
        else:
            data = (
                [self.request_json, response, self.metadata]
                if self.metadata
                else [self.request_json, response]
            )
            append_to_jsonl(data, save_filepath)
            # ========================== Begin of added ==========================
            choice = response['choices'][0]
            finish = choice['finish_reason']
            if finish != 'stop':
                logger.warning(f'Completion {self.log_ordinal} did not finish normally w/ reason {pl.i(finish)}')
            if completion_filepath:
                if self.is_chat_model:
                    completion = choice['message']['content']
                    with open(completion_filepath, 'w') as fl:
                        fl.write(completion)
                    if self.logprobs:  # write the log probs to a separate file
                        logprobs = choice['logprobs']['content']
                        base_dir_nm, fnm = os.path.dirname(completion_filepath), stem(completion_filepath)
                        base_dir_nm = os.path.join(base_dir_nm, 'logprobs')
                        os.makedirs(base_dir_nm, exist_ok=True)
                        with open(os.path.join(base_dir_nm, f'{fnm}.json'), 'w') as fl:
                            json.dump(logprobs, fl)  # no indent to save file size
                else:
                    raise NotImplementedError
            # ========================== End of added ==========================
            status_tracker.num_tasks_in_progress -= 1
            status_tracker.num_tasks_succeeded += 1
            n_res_tok = response['usage']['completion_tokens']
            d_log = {'#res-token': n_res_tok, 'duration': tm.end()}
            stats_tracker.log_completion({'#res-token': n_res_tok, 'finish-reason': finish})
            logger.info(f"Request {self.log_ordinal} w/ {pl.i(d_log)} saved")
            if self.pbar:
                self.pbar.update(1)
                self.pbar.set_postfix(req=self.log_ordinal)
            # ========================== Begin of added ==========================


# functions


def api_endpoint_from_url(request_url):
    """Extract the API endpoint from the request URL."""
    match = re.search("^https://[^/]+/v\\d+/(.+)$", request_url)
    if match is None:
        # for Azure OpenAI deployment urls
        match = re.search(r"^https://[^/]+/openai/deployments/[^/]+/(.+?)(\?|$)", request_url)
    return match[1]


def append_to_jsonl(data, filename: str) -> None:
    """Append a json payload to the end of a jsonl file."""
    json_string = json.dumps(data)
    with open(filename, "a") as f:
        f.write(json_string + "\n")


def num_tokens_consumed_from_request(
    request_json: dict,
    api_endpoint: str,
    token_encoding_name: str,
):
    """Count the number of tokens in the request. Only supports completion and embedding requests."""
    encoding = tiktoken.get_encoding(token_encoding_name)
    # if completions request, tokens = prompt + n * max_tokens
    if api_endpoint.endswith("completions"):
        max_tokens = request_json.get("max_tokens", 15)
        n = request_json.get("n", 1)
        completion_tokens = n * max_tokens

        # chat completions
        if api_endpoint.startswith("chat/"):
            num_tokens = 0
            for message in request_json["messages"]:
                num_tokens += 4  # every message follows <im_start>{role/name}\n{content}<im_end>\n
                for key, value in message.items():
                    num_tokens += len(encoding.encode(value))
                    if key == "name":  # if there's a name, the role is omitted
                        num_tokens -= 1  # role is always required and always 1 token
            num_tokens += 2  # every reply is primed with <im_start>assistant
            return num_tokens + completion_tokens
        # normal completions
        else:
            prompt = request_json["prompt"]
            if isinstance(prompt, str):  # single prompt
                prompt_tokens = len(encoding.encode(prompt))
                num_tokens = prompt_tokens + completion_tokens
                return num_tokens
            elif isinstance(prompt, list):  # multiple prompts
                prompt_tokens = sum([len(encoding.encode(p)) for p in prompt])
                num_tokens = prompt_tokens + completion_tokens * len(prompt)
                return num_tokens
            else:
                raise TypeError(
                    'Expecting either string or list of strings for "prompt" field in completion request'
                )
    # if embeddings request, tokens = input tokens
    elif api_endpoint == "embeddings":
        input = request_json["input"]
        if isinstance(input, str):  # single input
            num_tokens = len(encoding.encode(input))
            return num_tokens
        elif isinstance(input, list):  # multiple inputs
            num_tokens = sum([len(encoding.encode(i)) for i in input])
            return num_tokens
        else:
            raise TypeError(
                'Expecting either string or list of strings for "inputs" field in embedding request'
            )
    # more logic needed to support other API calls (e.g., edits, inserts, DALL-E)
    else:
        raise NotImplementedError(
            f'API endpoint "{api_endpoint}" not implemented in this script'
        )


def task_id_generator_function():
    """Generate integers 0, 1, 2, and so on."""
    task_id = 0
    while True:
        yield task_id
        task_id += 1


# run script


if __name__ == "__main__":
    # parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--requests_filepath")
    parser.add_argument("--save_filepath", default=None)
    parser.add_argument("--request_url", default="https://api.openai.com/v1/embeddings")
    parser.add_argument("--api_key", default=os.getenv("OPENAI_API_KEY"))
    parser.add_argument("--max_requests_per_minute", type=int, default=3_000 * 0.5)
    parser.add_argument("--max_tokens_per_minute", type=int, default=250_000 * 0.5)
    parser.add_argument("--token_encoding_name", default="cl100k_base")
    parser.add_argument("--max_attempts", type=int, default=5)
    parser.add_argument("--logging_level", default=logging.INFO)
    args = parser.parse_args()

    if args.save_filepath is None:
        args.save_filepath = args.requests_filepath.replace(".jsonl", "_results.jsonl")

    # run script
    asyncio.run(
        parallel_api_call(
            requests_filepath=args.requests_filepath,
            save_filepath=args.save_filepath,
            request_url=args.request_url,
            api_key=args.api_key,
            max_requests_per_minute=float(args.max_requests_per_minute),
            max_tokens_per_minute=float(args.max_tokens_per_minute),
            token_encoding_name=args.token_encoding_name,
            max_attempts=int(args.max_attempts),
            logging_level=int(args.logging_level),
        )
    )


"""
APPENDIX

The example requests file at openai-cookbook/examples/data/example_requests_to_parallel_process.jsonl contains 10,000 requests to text-embedding-ada-002.

It was generated with the following code:

```python
import json

filename = "data/example_requests_to_parallel_process.jsonl"
n_requests = 10_000
jobs = [{"model": "text-embedding-ada-002", "input": str(x) + "\n"} for x in range(n_requests)]
with open(filename, "w") as f:
    for job in jobs:
        json_string = json.dumps(job)
        f.write(json_string + "\n")
```

As with all jsonl files, take care that newlines in the content are properly escaped (json.dumps does this automatically).
"""
