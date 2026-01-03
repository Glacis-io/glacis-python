"""
GLACIS integration for Anthropic.

Provides an attested Anthropic client wrapper that automatically logs all
messages to the GLACIS transparency log.

Example:
    >>> from glacis.integrations.anthropic import attested_anthropic
    >>> client = attested_anthropic(glacis_api_key="glsk_live_xxx", anthropic_api_key="sk-xxx")
    >>> response = client.messages.create(
    ...     model="claude-3-opus-20240229",
    ...     messages=[{"role": "user", "content": "Hello!"}]
    ... )
    # Response is automatically attested to GLACIS
"""

from typing import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:
    from anthropic import Anthropic


def attested_anthropic(
    glacis_api_key: str,
    anthropic_api_key: Optional[str] = None,
    glacis_base_url: str = "https://api.glacis.io",
    service_id: str = "anthropic",
    debug: bool = False,
    **anthropic_kwargs: Any,
) -> "Anthropic":
    """
    Create an attested Anthropic client.

    All messages are automatically attested to the GLACIS transparency log.
    The input (messages) and output (response) are hashed locally - the actual
    content never leaves your infrastructure.

    Args:
        glacis_api_key: GLACIS API key
        anthropic_api_key: Anthropic API key (default: from ANTHROPIC_API_KEY env var)
        glacis_base_url: GLACIS API base URL
        service_id: Service identifier for attestations
        debug: Enable debug logging
        **anthropic_kwargs: Additional arguments passed to Anthropic client

    Returns:
        Wrapped Anthropic client

    Example:
        >>> client = attested_anthropic(
        ...     glacis_api_key="glsk_live_xxx",
        ...     anthropic_api_key="sk-xxx"
        ... )
        >>> response = client.messages.create(
        ...     model="claude-3-opus-20240229",
        ...     max_tokens=1024,
        ...     messages=[{"role": "user", "content": "Hello!"}]
        ... )
    """
    try:
        from anthropic import Anthropic
    except ImportError:
        raise ImportError(
            "Anthropic integration requires the 'anthropic' package. "
            "Install it with: pip install glacis[anthropic]"
        )

    from glacis import Glacis

    glacis = Glacis(
        api_key=glacis_api_key,
        base_url=glacis_base_url,
        debug=debug,
    )

    # Create the Anthropic client
    client_kwargs: dict[str, Any] = {**anthropic_kwargs}
    if anthropic_api_key:
        client_kwargs["api_key"] = anthropic_api_key

    client = Anthropic(**client_kwargs)

    # Wrap the messages create method
    original_create = client.messages.create

    def attested_create(*args: Any, **kwargs: Any) -> Any:
        # Extract input
        messages = kwargs.get("messages", [])
        model = kwargs.get("model", "unknown")
        system = kwargs.get("system")

        # Make the API call
        response = original_create(*args, **kwargs)

        # Attest the interaction
        try:
            input_data: dict[str, Any] = {
                "model": model,
                "messages": messages,
            }
            if system:
                input_data["system"] = system

            glacis.attest(
                service_id=service_id,
                operation_type="completion",
                input=input_data,
                output={
                    "model": response.model,
                    "content": [
                        {
                            "type": block.type,
                            "text": getattr(block, "text", None),
                        }
                        for block in response.content
                    ],
                    "stop_reason": response.stop_reason,
                    "usage": {
                        "input_tokens": response.usage.input_tokens,
                        "output_tokens": response.usage.output_tokens,
                    },
                },
                metadata={"provider": "anthropic", "model": model},
            )
        except Exception as e:
            if debug:
                print(f"[glacis] Attestation failed: {e}")

        return response

    # Replace the create method
    client.messages.create = attested_create  # type: ignore

    return client


def attested_async_anthropic(
    glacis_api_key: str,
    anthropic_api_key: Optional[str] = None,
    glacis_base_url: str = "https://api.glacis.io",
    service_id: str = "anthropic",
    debug: bool = False,
    **anthropic_kwargs: Any,
) -> Any:
    """
    Create an attested async Anthropic client.

    Same as `attested_anthropic` but for async usage.

    Example:
        >>> client = attested_async_anthropic(glacis_api_key="glsk_live_xxx")
        >>> response = await client.messages.create(...)
    """
    try:
        from anthropic import AsyncAnthropic
    except ImportError:
        raise ImportError(
            "Anthropic integration requires the 'anthropic' package. "
            "Install it with: pip install glacis[anthropic]"
        )

    from glacis import AsyncGlacis

    glacis = AsyncGlacis(
        api_key=glacis_api_key,
        base_url=glacis_base_url,
        debug=debug,
    )

    client_kwargs: dict[str, Any] = {**anthropic_kwargs}
    if anthropic_api_key:
        client_kwargs["api_key"] = anthropic_api_key

    client = AsyncAnthropic(**client_kwargs)

    original_create = client.messages.create

    async def attested_create(*args: Any, **kwargs: Any) -> Any:
        messages = kwargs.get("messages", [])
        model = kwargs.get("model", "unknown")
        system = kwargs.get("system")

        response = await original_create(*args, **kwargs)

        try:
            input_data: dict[str, Any] = {
                "model": model,
                "messages": messages,
            }
            if system:
                input_data["system"] = system

            await glacis.attest(
                service_id=service_id,
                operation_type="completion",
                input=input_data,
                output={
                    "model": response.model,
                    "content": [
                        {
                            "type": block.type,
                            "text": getattr(block, "text", None),
                        }
                        for block in response.content
                    ],
                    "stop_reason": response.stop_reason,
                    "usage": {
                        "input_tokens": response.usage.input_tokens,
                        "output_tokens": response.usage.output_tokens,
                    },
                },
                metadata={"provider": "anthropic", "model": model},
            )
        except Exception as e:
            if debug:
                print(f"[glacis] Attestation failed: {e}")

        return response

    client.messages.create = attested_create  # type: ignore

    return client
