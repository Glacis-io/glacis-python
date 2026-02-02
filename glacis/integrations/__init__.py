"""
GLACIS integrations for AI providers.

These integrations provide drop-in wrappers that automatically attest
all API calls to the GLACIS transparency log with optional PII/PHI redaction.

Available integrations:
- OpenAI: `from glacis.integrations.openai import attested_openai`
- Anthropic: `from glacis.integrations.anthropic import attested_anthropic`

Example (OpenAI):
    >>> from glacis.integrations.openai import attested_openai, get_last_receipt
    >>> client = attested_openai(
    ...     openai_api_key="sk-xxx",
    ...     offline=True,
    ...     signing_seed=os.urandom(32),
    ...     redaction="fast",  # Enable PII redaction
    ... )
    >>> response = client.chat.completions.create(
    ...     model="gpt-4o",
    ...     messages=[{"role": "user", "content": "Hello!"}]
    ... )
    >>> receipt = get_last_receipt()

Example (Anthropic):
    >>> from glacis.integrations.anthropic import attested_anthropic, get_last_receipt
    >>> client = attested_anthropic(
    ...     anthropic_api_key="sk-ant-xxx",
    ...     offline=True,
    ...     signing_seed=os.urandom(32),
    ...     redaction="fast",
    ... )
    >>> response = client.messages.create(
    ...     model="claude-3-5-sonnet-20241022",
    ...     max_tokens=1024,
    ...     messages=[{"role": "user", "content": "Hello!"}]
    ... )

Note: For Azure OpenAI, use the openai package with azure endpoint configuration.
"""

from glacis.integrations.anthropic import attested_anthropic
from glacis.integrations.base import (
    GlacisBlockedError,
    get_evidence,
    get_last_receipt,
)
from glacis.integrations.openai import attested_openai

# Backwards compatible aliases
get_last_openai_receipt = get_last_receipt
get_last_anthropic_receipt = get_last_receipt

__all__ = [
    "attested_openai",
    "attested_anthropic",
    "get_last_receipt",
    "get_last_openai_receipt",
    "get_last_anthropic_receipt",
    "get_evidence",
    "GlacisBlockedError",
]
