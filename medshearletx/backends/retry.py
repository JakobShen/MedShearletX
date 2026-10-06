"""Bounded retries of transport failures; never resample invalid classifications."""

import math
from numbers import Real
from threading import Lock
import time

from .base import BackendRequestError


class RetryingBackend:
    """Wrap RecordedBackend so every physical attempt remains logged and capped.

    The allowance is shared across the entire run, including concurrent calls.
    Scorers still request the same number of usable classification draws.
    """

    def __init__(self, backend, *, retry_budget=0, max_attempts=3,
                 backoff_seconds=0.5, sleep=time.sleep):
        if type(retry_budget) is not int or retry_budget < 0:
            raise ValueError("retry_budget must be a nonnegative integer")
        if type(max_attempts) is not int or not 1 <= max_attempts <= 10:
            raise ValueError("max_attempts must be an integer in [1,10]")
        if (isinstance(backoff_seconds, bool) or not isinstance(backoff_seconds, Real)
                or not math.isfinite(backoff_seconds) or backoff_seconds < 0):
            raise ValueError("backoff_seconds must be finite and nonnegative")
        if not callable(sleep):
            raise ValueError("sleep must be callable")
        self.backend = backend
        self.retry_budget = retry_budget
        self.max_attempts = max_attempts
        self.backoff_seconds = float(backoff_seconds)
        self._sleep = sleep
        self._retry_count = 0
        self._lock = Lock()

    def __getattr__(self, name):
        return getattr(self.backend, name)

    @property
    def retry_count(self):
        with self._lock:
            return self._retry_count

    def predict(self, image, task, *, require_logprobs, temperature=1.0):
        for attempt in range(self.max_attempts):
            try:
                return self.backend.predict(image, task, require_logprobs=require_logprobs,
                                            temperature=temperature)
            except BackendRequestError as error:
                if not error.retryable or attempt + 1 == self.max_attempts:
                    raise
                with self._lock:
                    if self._retry_count >= self.retry_budget:
                        raise
                    self._retry_count += 1
                self._sleep(min(2.0, self.backoff_seconds * 2 ** attempt))
