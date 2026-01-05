"""
GLACIS integration for Azure AI Inference.

Provides an attested Azure ChatCompletionsClient wrapper that automatically logs all
completions to the GLACIS transparency log. Supports both online (server-witnessed)
and offline (locally-signed) modes.

Example (online):
    >>> from azure.core.credentials import AzureKeyCredential
    >>> from glacis.integrations.azure import attested_azure_inference
    >>> client = attested_azure_inference(
    ...     glacis_api_key="glsk_live_xxx",
    ...     endpoint="https://my-endpoint.inference.ai.azure.com",
    ...     credential=AzureKeyCredential("my-azure-key"),
    ... )
    >>> response = client.complete(
    ...     messages=[UserMessage("Hello!")],
    ...     model="gpt-4o",
    ... )
    # Response is automatically attested to GLACIS

Example (offline):
    >>> client = attested_azure_inference(
    ...     endpoint="https://my-endpoint.inference.ai.azure.com",
    ...     credential=AzureKeyCredential("my-azure-key"),
    ...     offline=True,
    ...     signing_seed=seed,
    ... )
    >>> response = client.complete(messages=[UserMessage("Hello!")], model="gpt-4o")
    >>> receipt = get_last_receipt()
"""

from __future__ import annotations

import threading
from typing import TYPE_CHECKING, Any, Optional, Union

if TYPE_CHECKING:
    from azure.ai.inference import ChatCompletionsClient
    from azure.core.credentials import AzureKeyCredential

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


def attested_azure_inference(
    glacis_api_key: Optional[str] = None,
    endpoint: Optional[str] = None,
    credential: Optional["AzureKeyCredential"] = None,
    glacis_base_url: str = "https://api.glacis.io",
    service_id: str = "azure-inference",
    debug: bool = False,
    offline: bool = False,
    signing_seed: Optional[bytes] = None,
    **azure_kwargs: Any,
) -> "ChatCompletionsClient":
    """
    Create an attested Azure AI Inference client.

    All chat completions are automatically attested. Supports both online and offline modes.
    Note: Streaming is not currently supported.

    Args:
        glacis_api_key: GLACIS API key (required for online mode)
        endpoint: Azure AI Inference endpoint URL
        credential: Azure credential (e.g., AzureKeyCredential)
        glacis_base_url: GLACIS API base URL
        service_id: Service identifier for attestations
        debug: Enable debug logging
        offline: Enable offline mode (local signing, no server)
        signing_seed: 32-byte Ed25519 signing seed (required for offline mode)
        **azure_kwargs: Additional arguments passed to ChatCompletionsClient

    Returns:
        Wrapped ChatCompletionsClient

    Example (online):
        >>> from azure.core.credentials import AzureKeyCredential
        >>> client = attested_azure_inference(
        ...     glacis_api_key="glsk_live_xxx",
        ...     endpoint="https://my-endpoint.inference.ai.azure.com",
        ...     credential=AzureKeyCredential("my-azure-key"),
        ... )
        >>> response = client.complete(
        ...     messages=[UserMessage("Hello!")],
        ...     model="gpt-4o",
        ... )

    Example (offline):
        >>> import os
        >>> seed = os.urandom(32)
        >>> client = attested_azure_inference(
        ...     endpoint="https://my-endpoint.inference.ai.azure.com",
        ...     credential=AzureKeyCredential("my-azure-key"),
        ...     offline=True,
        ...     signing_seed=seed,
        ... )
        >>> response = client.complete(messages=[UserMessage("Hello!")], model="gpt-4o")
        >>> receipt = get_last_receipt()
        >>> assert receipt.witness_status == "UNVERIFIED"
    """
    try:
        from azure.ai.inference import ChatCompletionsClient
    except ImportError:
        raise ImportError(
            "Azure AI Inference integration requires the 'azure-ai-inference' package. "
            "Install it with: pip install glacis[azure]"
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

    # Create the Azure ChatCompletionsClient
    if not endpoint:
        raise ValueError("endpoint is required")
    if not credential:
        raise ValueError("credential is required")

    client = ChatCompletionsClient(
        endpoint=endpoint,
        credential=credential,
        **azure_kwargs,
    )

    # Wrap the complete method
    original_complete = client.complete

    def attested_complete(*args: Any, **kwargs: Any) -> Any:
        # Check for streaming - not supported
        if kwargs.get("stream", False):
            raise NotImplementedError(
                "Streaming is not currently supported with attested_azure_inference. "
                "Use stream=False for now."
            )

        # Extract input
        messages = kwargs.get("messages", [])
        model = kwargs.get("model", "unknown")

        # Make the API call
        response = original_complete(*args, **kwargs)

        # Attest the response
        try:
            # Convert messages to serializable format
            serializable_messages: list[Any] = []
            for m in messages:
                if hasattr(m, "role") and hasattr(m, "content"):
                    serializable_messages.append({"role": m.role, "content": m.content})
                elif isinstance(m, dict):
                    serializable_messages.append(m)
                else:
                    serializable_messages.append(str(m))

            receipt = glacis.attest(
                service_id=service_id,
                operation_type="completion",
                input={
                    "model": model,
                    "messages": serializable_messages,
                },
                output={
                    "model": response.model if hasattr(response, "model") else model,
                    "choices": [
                        {
                            "message": {
                                "role": getattr(c.message, "role", "assistant"),
                                "content": getattr(c.message, "content", ""),
                            },
                            "finish_reason": getattr(c, "finish_reason", None),
                        }
                        for c in response.choices
                    ]
                    if hasattr(response, "choices")
                    else [],
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
                    if hasattr(response, "usage") and response.usage
                    else None,
                },
                metadata={"provider": "azure-inference", "model": model},
            )
            _thread_local.last_receipt = receipt
            if debug:
                print(f"[glacis] Attestation created: {receipt.attestation_id}")
        except Exception as e:
            if debug:
                print(f"[glacis] Attestation failed: {e}")

        return response

    # Replace the complete method
    client.complete = attested_complete  # type: ignore

    return client
