import os
import openai
from tenacity import (
    retry,
    stop_after_attempt, # type: ignore
    wait_random_exponential, # type: ignore
)

from typing import Optional, List, Union

import atexit
import hashlib
import logging
import os
import time
from typing import Any, List, Mapping, Optional, Tuple, Union

import openai
import pandas as pd
from yaspin import yaspin
from yaspin.spinners import Spinners

openai.api_key = os.getenv('OPENAI_API_KEY')

# Jared's code below

# Default directory to store cached values such as API calls and embedding vectors.
CACHE_DIR = "~/.openai_cache/"

# Default arguments to pass to OpenAI GPT chat completion API calls.
DEFAULT_GPT_API_ARGS = {
    "model": "gpt-3.5-turbo",
    "temperature": 0.0,
}


logger = logging.getLogger(__name__)


class OpenAICache:
    """
    Caches OpenAI API requests and responses to improve runtime performance and save on
    costs.

    Arguments:
        file: Local file to store the cache in. If not specified, the cache will be
           stored in the default cache directory.
    """

    def __init__(self, file: Optional[str] = None, max_tokens=256):
        self.file = os.path.expanduser(
            file or os.path.join(CACHE_DIR, "openai_cache.json")
        )

        atexit.register(self.cleanup)

        if os.path.exists(self.file) and os.stat(self.file).st_size > 0:
            self.store = {
                row.hash: (row.request, row.result)
                for row in pd.read_json(self.file, lines=True).itertuples()
            }

            logger.debug(
                f"Loaded {len(self.store)} cached openAI API calls from file '{self.file}'"
            )
        else:
            logger.debug("Setup new openAI API cache")
            self.store = {}

        self.max_tokens = max_tokens

    def call_gpt(
        self,
        messages: Union[List[Mapping[str, str]], List[Tuple[str, str]]],
        api_args: Optional[Mapping[str, Any]] = None,
        retry_on_rate_limit_error: bool = True,
    ) -> Tuple[str, str]:
        """
        Takes an API requests and tries to retrieve the result from the cache. If the
        result is not in the cache, the request is forwarded to the OpenAI API and the
        result is cached.

        Arguments:
            messages: A list of messages to send to the OpenAI API. Each message is
                either a dictionary with a "role" and "content" key or a tuple with a
                role and content.
            api_args: An optional dictionary of arguments to be passed to the OpenAI API.
            retry_on_rate_limit_error: Whether to retry the API call if the rate limit
                is exceeded.

        Returns:
            A tuple with the role and content of the response from the OpenAI API.
        """
        args = DEFAULT_GPT_API_ARGS.copy()

        if api_args:
            args.update(api_args)

        messages = [
            {"role": m[0], "content": m[1]} if isinstance(m, tuple) else m
            for m in messages
        ]

        args["messages"] = messages

        args_hash = self._compute_md5_hash(str(args.items()))

        if args["temperature"] == 0.0 and args_hash in self.store:
            logger.debug("Avoided OpenAI API call by retrieving from cache")
            return tuple(self.store[args_hash][1])

        if retry_on_rate_limit_error:
            for i in range(11):
                try:
                    text = "" if i == 0 else f" (retry {i}/10)"
                    with yaspin(
                        spinner=Spinners.bouncingBar,
                        text=f"Calling OpenAI API... {text}",
                    ):
                        response = openai.ChatCompletion.create(max_tokens=self.max_tokens, **args)
                        break
                except openai.error.RateLimitError:
                    logger.warning(
                        "Rate limit exceeded. Retrying in 1 second(s)...",
                        exc_info=True,
                    )
                    time.sleep(1)
                except openai.error.ServiceUnavailableError:
                    logger.warning(
                        "Service Error. Retrying in 1 second(s)...",
                        exc_info=True,
                    )
                    time.sleep(1)
        else:
            with yaspin(spinner=Spinners.bouncingBar, text="Calling OpenAI API..."):
                response = openai.ChatCompletion.create(**args)

        logger.debug("Called OpenAI API")

        assert isinstance(response, dict)
        result = tuple(response["choices"][0]["message"].values())

        if args["temperature"] == 0.0:
            self.store[args_hash] = (args, result)

        return result

    def cleanup(self) -> None:
        """
        Saves the cache to the file specified in the constructor.
        """
        if len(self.store) > 0:
            requests, results = zip(*self.store.values())

            df = pd.DataFrame(
                {
                    "hash": list(self.store.keys()),
                    "request": requests,
                    "result": results,
                }
            )
            df.to_json(self.file, lines=True, orient="records")

            logger.debug(
                f"Saved {len(self.store)} cached openAI API calls to file '{self.file}'"
            )

    def _compute_md5_hash(self, string: str) -> str:
        """
        Internal method.
        """
        m = hashlib.md5()
        m.update(string.encode("utf-8"))
        return m.hexdigest()


# This creates a package-level global variable that can be used to access the cache so
# that we don't have to pass it around everywhere.
global_openai_cache = OpenAICache()
global_openai_cache_reflection = OpenAICache(max_tokens=256)



@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
def get_completion(prompt: Union[str, List[str]], max_tokens: int = 256, stop_strs: Optional[List[str]] = None, is_batched: bool = False) -> Union[str, List[str]]:
    assert (not is_batched and isinstance(prompt, str)) or (is_batched and isinstance(prompt, list))
    return global_openai_cache_reflection.call_gpt(
            [('system', "You are a helpful assistant."),
             ('user', prompt)],
            api_args=dict(stop=["\n"])
            )[1]

    # messages = [
    #     {"role": "system", "content": "You are a helpful assistant."},
    #     {"role": "user", "content": prompt}
    # ]

    # response = openai.ChatCompletion.create(
    #     model='gpt-3.5-turbo',
    #     messages=messages,
    #     temperature=0.0,
    #     max_tokens=max_tokens,
    #     stop=stop_strs,
    # )
    # if is_batched:
    #     res: List[str] = [""] * len(prompt)
    #     for choice in response.choices:
    #         res[choice.index] = choice.text
    #     return rek      
    # return response.choices[0].message["content"]



