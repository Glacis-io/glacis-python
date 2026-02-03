"""
GLACIS integration for Anthropic.

Provides an attested Anthropic client wrapper that automatically:
1. Runs enabled controls (PII/PHI redaction, jailbreak detection, etc.)
2. Logs all completions to the GLACIS transparency log
3. Creates control plane attestations

Example:
    >>> from glacis.integrations.anthropic import attested_anthropic, get_last_receipt
    >>> client = attested_anthropic(
    ...     anthropic_api_key="sk-ant-xxx",
    ...     offline=True,
    ...     signing_seed=os.urandom(32),
    ... )
    >>> response = client.messages.create(
    ...     model="claude-3-5-sonnet-20241022",
    ...     max_tokens=1024,
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
    from anthropic import Anthropic


def attested_anthropic(
    glacis_api_key: Optional[str] = None,
    anthropic_api_key: Optional[str] = None,
    glacis_base_url: str = "https://api.glacis.io",
    service_id: str = "anthropic",
    debug: bool = False,
    offline: Optional[bool] = None,
    signing_seed: Optional[bytes] = None,
    redaction: Union[bool, Literal["fast", "full"], None] = None,
    config: Optional[str] = None,
    **anthropic_kwargs: Any,
) -> "Anthropic":
    """
    Create an attested Anthropic client with controls (PII redaction, jailbreak detection).

    Args:
        glacis_api_key: GLACIS API key (required for online mode)
        anthropic_api_key: Anthropic API key (default: from ANTHROPIC_API_KEY env var)
        glacis_base_url: GLACIS API base URL
        service_id: Service identifier for attestations
        debug: Enable debug logging
        offline: Enable offline mode (local signing, no server)
        signing_seed: 32-byte Ed25519 signing seed (required for offline mode)
        redaction: PII/PHI redaction mode - "fast", "full", True, False, or None
        config: Path to glacis.yaml config file
        **anthropic_kwargs: Additional arguments passed to Anthropic client

    Returns:
        Wrapped Anthropic client

    Raises:
        GlacisBlockedError: If a control blocks the request
    """
    # Suppress noisy loggers
    suppress_noisy_loggers(["anthropic", "anthropic._base_client"])

    try:
        from anthropic import Anthropic
    except ImportError:
        raise ImportError(
            "Anthropic integration requires the 'anthropic' package. "
            "Install it with: pip install glacis[anthropic]"
        )


    # Initialize config and determine modes
    cfg, effective_offline, effective_service_id = initialize_config(
        config_path=config,
        redaction=redaction,
        offline=offline,
        glacis_api_key=glacis_api_key,
        default_service_id="anthropic",
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

    # Create the Anthropic client
    client_kwargs: dict[str, Any] = {**anthropic_kwargs}
    if anthropic_api_key:
        client_kwargs["api_key"] = anthropic_api_key

    client = Anthropic(**client_kwargs)

    # Wrap the messages create method
    original_create = client.messages.create

    def attested_create(*args: Any, **kwargs: Any) -> Any:
        if kwargs.get("stream", False):
            raise NotImplementedError(
                "Streaming is not currently supported with attested_anthropic. "
                "Use stream=False for now."
            )

        messages = kwargs.get("messages", [])
        model = kwargs.get("model", "unknown")
        system = kwargs.get("system")

        # Run controls if enabled
        if controls_runner:
            from glacis.integrations.base import (
                ControlResultsAccumulator,
                create_control_plane_attestation_from_accumulator,
                handle_blocked_request,
                process_text_for_controls,
            )

            accumulator = ControlResultsAccumulator()

            # Process system prompt through controls
            # (system prompt is always "new" in current context)
            if system and isinstance(system, str):
                final_system = process_text_for_controls(
                    controls_runner, system, accumulator
                )
                kwargs["system"] = final_system

            # Process messages with delta scanning (only scan last user message)
            processed_messages = []

            # Find the last user message index
            last_user_idx = -1
            for i, msg in enumerate(messages):
                if isinstance(msg, dict) and msg.get("role") == "user":
                    last_user_idx = i

            for i, msg in enumerate(messages):
                role = msg.get("role", "") if isinstance(msg, dict) else ""

                # Only run controls on the LAST user message (the new one)
                if role == "user" and i == last_user_idx:
                    if isinstance(msg, dict) and isinstance(msg.get("content"), str):
                        content = msg["content"]
                        final_text = process_text_for_controls(
                            controls_runner, content, accumulator
                        )
                        processed_messages.append({**msg, "content": final_text})

                    elif isinstance(msg, dict) and isinstance(msg.get("content"), list):
                        # Handle content blocks (text, image, etc.)
                        redacted_content = []
                        for block in msg["content"]:
                            if isinstance(block, dict) and block.get("type") == "text":
                                text = block.get("text", "")
                                final_text = process_text_for_controls(
                                    controls_runner, text, accumulator
                                )
                                redacted_content.append({**block, "text": final_text})
                            else:
                                redacted_content.append(block)
                        processed_messages.append({**msg, "content": redacted_content})
                    else:
                        processed_messages.append(msg)
                else:
                    processed_messages.append(msg)

            kwargs["messages"] = processed_messages
            messages = processed_messages

            if debug:
                if accumulator.pii_summary:
                    print(
                        f"[glacis] PII redacted: {accumulator.pii_summary.categories} "
                        f"({accumulator.pii_summary.count} items)"
                    )
                if accumulator.jailbreak_summary and accumulator.jailbreak_summary.detected:
                    print(
                        f"[glacis] Jailbreak detected: "
                        f"score={accumulator.jailbreak_summary.score:.2f}, "
                        f"action={accumulator.jailbreak_summary.action}"
                    )

            # Build control plane attestation
            control_plane_results = create_control_plane_attestation_from_accumulator(
                accumulator, cfg, model, "anthropic", "messages.create"
            )

            # Check if we need to block BEFORE making the API call
            if accumulator.should_block:
                handle_blocked_request(
                    glacis_client=glacis,
                    service_id=effective_service_id,
                    input_data={
                        "model": model,
                        "messages": messages,
                        "system": kwargs.get("system")
                    },
                    control_plane_results=control_plane_results,
                    provider="anthropic",
                    model=model,
                    jailbreak_score=accumulator.jailbreak_summary.score
                    if accumulator.jailbreak_summary
                    else 0.0,
                    debug=debug,
                )
        else:
            control_plane_results = None

        # Make the API call (only if not blocked)
        response = original_create(*args, **kwargs)

        # Build input/output data
        input_data: dict[str, Any] = {"model": model, "messages": messages}
        if system:
            input_data["system"] = kwargs.get("system", system)

        output_data = {
            "model": response.model,
            "content": [
                {"type": block.type, "text": getattr(block, "text", None)}
                for block in response.content
            ],
            "stop_reason": response.stop_reason,
            "usage": {
                "input_tokens": response.usage.input_tokens,
                "output_tokens": response.usage.output_tokens,
            },
        }

        # Attest and store
        try:
            receipt = glacis.attest(
                service_id=effective_service_id,
                operation_type="completion",
                input=input_data,
                output=output_data,
                metadata={"provider": "anthropic", "model": model},
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
                metadata={"provider": "anthropic", "model": model},
                debug=debug,
            )
        except Exception as e:
            if debug:
                print(f"[glacis] Attestation failed: {e}")

        return response

    client.messages.create = attested_create  # type: ignore
    return client


__all__ = [
    "attested_anthropic",
    "get_last_receipt",
    "get_evidence",
    "GlacisBlockedError",
]
