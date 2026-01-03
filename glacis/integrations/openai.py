"""
GLACIS integration for OpenAI.

Provides an attested OpenAI client wrapper that automatically logs all
completions to the GLACIS transparency log. Supports both online (server-witnessed)
and offline (locally-signed) modes.

Example (online):
    >>> from glacis.integrations.openai import attested_openai
    >>> client = attested_openai(glacis_api_key="glsk_live_xxx", openai_api_key="sk-xxx")
    >>> response = client.chat.completions.create(
    ...     model="gpt-4",
    ...     messages=[{"role": "user", "content": "Hello!"}]
    ... )
    # Response is automatically attested to GLACIS

Example (offline):
    >>> client = attested_openai(openai_api_key="sk-xxx", offline=True, signing_seed=seed)
    >>> response = client.chat.completions.create(
    ...     model="gpt-4o",
    ...     messages=[{"role": "user", "content": "Hello!"}],
    ... )
    >>> receipt = get_last_receipt()
"""

from __future__ import annotations

import threading
from typing import TYPE_CHECKING, Any, Optional, Union

if TYPE_CHECKING:
    from openai import OpenAI

    from glacis.models import AttestReceipt, OfflineAttestReceipt


# Thread-local storage for the last receipt
_thread_local = threading.local()


def get_last_receipt() -> Optional[Union["AttestReceipt", "OfflineAttestReceipt"]]:
    """
    Get the last attestation receipt from the current thread.

    Returns:
        The last AttestReceipt or OfflineAttestReceipt, or None if no attestation
        has been made in this thread.
    """
    return getattr(_thread_local, "last_receipt", None)


def attested_openai(
    glacis_api_key: Optional[str] = None,
    openai_api_key: Optional[str] = None,
    glacis_base_url: str = "https://api.glacis.io",
    service_id: str = "openai",
    debug: bool = False,
    offline: bool = False,
    signing_seed: Optional[bytes] = None,
    **openai_kwargs: Any,
) -> "OpenAI":
    """
    Create an attested OpenAI client.

    All chat completions are automatically attested. Supports both online and offline modes.
    Note: Streaming is not currently supported.

    Args:
        glacis_api_key: GLACIS API key (required for online mode)
        openai_api_key: OpenAI API key (default: from OPENAI_API_KEY env var)
        glacis_base_url: GLACIS API base URL
        service_id: Service identifier for attestations
        debug: Enable debug logging
        offline: Enable offline mode (local signing, no server)
        signing_seed: 32-byte Ed25519 signing seed (required for offline mode)
        **openai_kwargs: Additional arguments passed to OpenAI client

    Returns:
        Wrapped OpenAI client

    Example (online):
        >>> client = attested_openai(
        ...     glacis_api_key="glsk_live_xxx",
        ...     openai_api_key="sk-xxx"
        ... )
        >>> response = client.chat.completions.create(
        ...     model="gpt-4",
        ...     messages=[{"role": "user", "content": "Hello!"}]
        ... )

    Example (offline):
        >>> import os
        >>> seed = os.urandom(32)
        >>> client = attested_openai(
        ...     openai_api_key="sk-xxx",
        ...     offline=True,
        ...     signing_seed=seed,
        ... )
        >>> response = client.chat.completions.create(
        ...     model="gpt-4o",
        ...     messages=[{"role": "user", "content": "Hello!"}],
        ... )
        >>> receipt = get_last_receipt()
        >>> assert receipt.witness_status == "UNVERIFIED"
    """
    try:
        from openai import OpenAI
    except ImportError:
        raise ImportError(
            "OpenAI integration requires the 'openai' package. "
            "Install it with: pip install glacis[openai]"
        )

    from glacis import Glacis

    # Create Glacis client (online or offline)
    if offline:
        if not signing_seed:
            raise ValueError("signing_seed is required for offline mode")
        glacis = Glacis(
            mode="offline",
            signing_seed=signing_seed,
            debug=debug,
        )
    else:
        if not glacis_api_key:
            raise ValueError("glacis_api_key is required for online mode")
        glacis = Glacis(
            api_key=glacis_api_key,
            base_url=glacis_base_url,
            debug=debug,
        )

    # Create the OpenAI client
    client_kwargs: dict[str, Any] = {**openai_kwargs}
    if openai_api_key:
        client_kwargs["api_key"] = openai_api_key

    client = OpenAI(**client_kwargs)

    # Wrap the chat completions create method
    original_create = client.chat.completions.create

    def attested_create(*args: Any, **kwargs: Any) -> Any:
        # Check for streaming - not supported
        if kwargs.get("stream", False):
            raise NotImplementedError(
                "Streaming is not currently supported with attested_openai. "
                "Use stream=False for now."
            )

        # Extract input
        messages = kwargs.get("messages", [])
        model = kwargs.get("model", "unknown")

        # Make the API call
        response = original_create(*args, **kwargs)

        # Attest the response
        try:
            receipt = glacis.attest(
                service_id=service_id,
                operation_type="completion",
                input={
                    "model": model,
                    "messages": messages,
                },
                output={
                    "model": response.model,
                    "choices": [
                        {
                            "message": {
                                "role": c.message.role,
                                "content": c.message.content,
                            },
                            "finish_reason": c.finish_reason,
                        }
                        for c in response.choices
                    ],
                    "usage": {
                        "prompt_tokens": (
                            response.usage.prompt_tokens if response.usage else 0
                        ),
                        "completion_tokens": (
                            response.usage.completion_tokens if response.usage else 0
                        ),
                        "total_tokens": (
                            response.usage.total_tokens if response.usage else 0
                        ),
                    }
                    if response.usage
                    else None,
                },
                metadata={"provider": "openai", "model": model},
            )
            _thread_local.last_receipt = receipt
            if debug:
                print(f"[glacis] Attestation created: {receipt.attestation_id}")
        except Exception as e:
            if debug:
                print(f"[glacis] Attestation failed: {e}")

        return response

    # Replace the create method
    client.chat.completions.create = attested_create  # type: ignore

    return client
