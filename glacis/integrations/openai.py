"""
GLACIS integration for OpenAI.

Provides an attested OpenAI client wrapper that automatically:
1. Runs enabled controls (PII/PHI redaction, jailbreak detection, etc.)
2. Logs all completions to the GLACIS transparency log
3. Creates control plane attestations

Example:
    >>> from glacis.integrations.openai import attested_openai, get_last_receipt
    >>> client = attested_openai(
    ...     openai_api_key="sk-xxx",
    ...     offline=True,
    ...     signing_seed=os.urandom(32),
    ... )
    >>> response = client.chat.completions.create(
    ...     model="gpt-4o",
    ...     messages=[{"role": "user", "content": "Hello!"}]
    ... )
    >>> receipt = get_last_receipt()
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal, Optional, Union

from glacis.integrations.base import (
    GlacisBlockedError,
    create_controls_runner,
    create_glacis_client,
    get_evidence,
    get_last_receipt,
    initialize_config,
    set_last_receipt,
    store_evidence,
    suppress_noisy_loggers,
)

if TYPE_CHECKING:
    from openai import OpenAI


def attested_openai(
    glacis_api_key: Optional[str] = None,
    openai_api_key: Optional[str] = None,
    glacis_base_url: str = "https://api.glacis.io",
    service_id: str = "openai",
    debug: bool = False,
    offline: Optional[bool] = None,
    signing_seed: Optional[bytes] = None,
    redaction: Union[bool, Literal["fast", "full"], None] = None,
    config: Optional[str] = None,
    **openai_kwargs: Any,
) -> "OpenAI":
    """
    Create an attested OpenAI client with controls (PII redaction, jailbreak detection).

    Args:
        glacis_api_key: GLACIS API key (required for online mode)
        openai_api_key: OpenAI API key (default: from OPENAI_API_KEY env var)
        glacis_base_url: GLACIS API base URL
        service_id: Service identifier for attestations
        debug: Enable debug logging
        offline: Enable offline mode (local signing, no server)
        signing_seed: 32-byte Ed25519 signing seed (required for offline mode)
        redaction: PII/PHI redaction mode - "fast", "full", True, False, or None
        config: Path to glacis.yaml config file
        **openai_kwargs: Additional arguments passed to OpenAI client

    Returns:
        Wrapped OpenAI client

    Raises:
        GlacisBlockedError: If a control blocks the request
    """
    # Suppress noisy loggers
    suppress_noisy_loggers(["openai", "openai._base_client"])

    try:
        from openai import OpenAI
    except ImportError:
        raise ImportError(
            "OpenAI integration requires the 'openai' package. "
            "Install it with: pip install glacis[openai]"
        )


    # Initialize config and determine modes
    cfg, effective_offline, effective_service_id = initialize_config(
        config_path=config,
        redaction=redaction,
        offline=offline,
        glacis_api_key=glacis_api_key,
        default_service_id="openai",
        service_id=service_id,
    )

    # Create controls runner and Glacis client
    controls_runner = create_controls_runner(cfg, debug)
    glacis = create_glacis_client(
        offline=effective_offline,
        signing_seed=signing_seed,
        glacis_api_key=glacis_api_key,
        glacis_base_url=glacis_base_url,
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
        if kwargs.get("stream", False):
            raise NotImplementedError(
                "Streaming is not currently supported with attested_openai. "
                "Use stream=False for now."
            )

        messages = kwargs.get("messages", [])
        model = kwargs.get("model", "unknown")

        # Run controls if enabled
        if controls_runner:
            from glacis.integrations.base import (
                ControlResultsAccumulator,
                create_control_plane_attestation_from_accumulator,
                handle_blocked_request,
                process_text_for_controls,
            )

            accumulator = ControlResultsAccumulator()
            processed_messages = []

            # Find the last user message index (the new message to check)
            last_user_idx = -1
            for i, msg in enumerate(messages):
                if isinstance(msg, dict) and msg.get("role") == "user":
                    last_user_idx = i

            for i, msg in enumerate(messages):
                role = msg.get("role", "") if isinstance(msg, dict) else ""
                # Only run controls on the LAST user message (the new one)
                if (
                    isinstance(msg, dict)
                    and isinstance(msg.get("content"), str)
                    and role == "user"
                    and i == last_user_idx
                ):
                    content = msg["content"]
                    final_text = process_text_for_controls(controls_runner, content, accumulator)
                    processed_messages.append({**msg, "content": final_text})
                else:
                    processed_messages.append(msg)

            kwargs["messages"] = processed_messages
            messages = processed_messages

            # Build control plane attestation
            control_plane_results = create_control_plane_attestation_from_accumulator(
                accumulator, cfg, model, "openai"
            )

            # Check if we need to block BEFORE making the API call
            if accumulator.should_block:
                handle_blocked_request(
                    glacis_client=glacis,
                    service_id=effective_service_id,
                    input_data={"model": model, "messages": messages},
                    control_plane_results=control_plane_results,
                    provider="openai",
                    model=model,
                    jailbreak_score=accumulator._jailbreak_score,
                    debug=debug,
                )
        else:
            control_plane_results = None

        # Make the API call (only if not blocked)
        response = original_create(*args, **kwargs)

        # Build input/output data
        input_data = {"model": model, "messages": messages}
        output_data = {
            "model": response.model,
            "choices": [
                {
                    "message": {"role": c.message.role, "content": c.message.content},
                    "finish_reason": c.finish_reason,
                }
                for c in response.choices
            ],
            "usage": {
                "prompt_tokens": response.usage.prompt_tokens if response.usage else 0,
                "completion_tokens": response.usage.completion_tokens if response.usage else 0,
                "total_tokens": response.usage.total_tokens if response.usage else 0,
            } if response.usage else None,
        }

        # Attest and store
        try:
            receipt = glacis.attest(
                service_id=effective_service_id,
                operation_type="completion",
                input=input_data,
                output=output_data,
                metadata={"provider": "openai", "model": model},
                control_plane_results=control_plane_results,
            )
            set_last_receipt(receipt)
            store_evidence(
                receipt=receipt,
                service_id=effective_service_id,
                operation_type="completion",
                input_data=input_data,
                output_data=output_data,
                control_plane_results=control_plane_results,
                metadata={"provider": "openai", "model": model},
                debug=debug,
            )
        except Exception as e:
            if debug:
                print(f"[glacis] Attestation failed: {e}")

        return response

    client.chat.completions.create = attested_create  # type: ignore
    return client


__all__ = [
    "attested_openai",
    "get_last_receipt",
    "get_evidence",
    "GlacisBlockedError",
]
