from __future__ import annotations

from groq import Groq
from groq import BadRequestError
from tenacity import retry, retry_if_not_exception_type, stop_after_attempt, wait_exponential

from biogenesis.config import settings
from biogenesis.logging_utils import get_logger

logger = get_logger(__name__)


def _is_json_validation_error(exc: BaseException) -> bool:
    """Return True only for Groq's structured-output JSON validation error."""
    if not isinstance(exc, BadRequestError):
        return False

    # Groq may expose the error code directly or only inside the response/body.
    error_code = getattr(exc, "code", None)
    if error_code == "json_validate_failed":
        return True

    body = getattr(exc, "body", None)
    if isinstance(body, dict):
        error = body.get("error")
        if isinstance(error, dict) and error.get("code") == "json_validate_failed":
            return True

    message = str(exc)
    return "json_validate_failed" in message


class LLMClient:
    def __init__(self) -> None:
        if not settings.groq_api_key:
            logger.warning(
                "GROQ_API_KEY is not set. Add it to your .env file "
                "(see .env.example) before calling the LLM."
            )

        self.client = Groq(api_key=settings.groq_api_key)
        self.model = settings.groq_model

    def _complete_once(
        self,
        prompt: str,
        system: str = "",
        temperature: float = 0.3,
        max_tokens: int = 1024,
        json_mode: bool = False,
    ) -> str:
        messages = []

        if system:
            messages.append({"role": "system", "content": system})

        messages.append({"role": "user", "content": prompt})

        kwargs = {}
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}

        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs,
        )

        return response.choices[0].message.content or ""

    @retry(
        retry=retry_if_not_exception_type(BadRequestError),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
    )
    def complete(
        self,
        prompt: str,
        system: str = "",
        temperature: float = 0.3,
        max_tokens: int = 1024,
        json_mode: bool = False,
    ) -> str:
        """
        Complete an LLM request.

        Transient/non-BadRequest errors retain the existing retry behavior.
        BadRequestError is not retried blindly because a 400 generally
        indicates a request/content problem rather than a transient failure.

        In particular, evidence extraction handles Groq's
        'json_validate_failed' case explicitly in EvidenceExtractor.
        """
        try:
            return self._complete_once(
                prompt=prompt,
                system=system,
                temperature=temperature,
                max_tokens=max_tokens,
                json_mode=json_mode,
            )
        except BadRequestError as exc:
            if _is_json_validation_error(exc):
                logger.warning(
                    "Groq JSON validation failed for model=%s. "
                    "Raising without retry so the caller can perform its "
                    "fallback extraction path: %s",
                    self.model,
                    exc,
                )
            else:
                logger.warning(
                    "Groq returned a non-retryable BadRequestError: %s",
                    exc,
                )
            raise