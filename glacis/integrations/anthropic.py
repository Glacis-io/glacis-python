"""
GLACIS integration for Anthropic.

Provides an attested Anthropic client wrapper that automatically:
1. Runs input/output controls (PII/PHI detection, jailbreak detection, word filter, etc.)
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

from typing import TYPE_CHECKING, Any, Optional

from glacis.integrations.base import (
    ControlResultsAccumulator,
    GlacisBlockedError,
    attest_and_store,
    build_metadata,
    check_input_block,
    check_output_block,
    create_control_plane_results,
    get_evidence,
    get_last_receipt,
    run_input_controls,
    run_output_controls,
    setup_integration,
    suppress_noisy_loggers,
)

if TYPE_CHECKING:
    from anthropic import Anthropic

    from glacis.controls.base import BaseControl


def attested_anthropic(
    glacis_api_key: Optional[str] = None,
    anthropic_api_key: Optional[str] = None,
    glacis_base_url: str = "https://api.glacis.io",
    service_id: str = "anthropic",
    debug: bool = False,
    offline: Optional[bool] = None,
    signing_seed: Optional[bytes] = None,
    policy_key: Optional[bytes] = None,
    config: Optional[str] = None,
    input_controls: Optional[list["BaseControl"]] = None,
    output_controls: Optional[list["BaseControl"]] = None,
    metadata: Optional[dict[str, str]] = None,
    **anthropic_kwargs: Any,
) -> "Anthropic":
    """
    Create an attested Anthropic client with controls.

    Args:
        glacis_api_key: GLACIS API key (required for online mode)
        anthropic_api_key: Anthropic API key (default: from ANTHROPIC_API_KEY env var)
        glacis_base_url: GLACIS API base URL
        service_id: Service identifier for attestations
        debug: Enable debug logging
        offline: Enable offline mode (local signing, no server)
        signing_seed: 32-byte Ed25519 signing seed (required for offline mode)
        policy_key: 32-byte HMAC key for sampling (falls back to signing_seed)
        config: Path to glacis.yaml config file
        input_controls: Custom controls for input stage
        output_controls: Custom controls for output stage
        metadata: Custom metadata to include in every attestation (string values only).
            Merged with provider defaults. Cannot override 'provider' or 'model'.
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

    ctx = setup_integration(
        config=config,
        offline=offline,
        glacis_api_key=glacis_api_key,
        glacis_base_url=glacis_base_url,
        default_service_id="anthropic",
        service_id=service_id,
        debug=debug,
        signing_seed=signing_seed,
        policy_key=policy_key,
        input_controls=input_controls,
        output_controls=output_controls,
        metadata=metadata,
    )

    # Create the Anthropic client
    client_kwargs: dict[str, Any] = {**anthropic_kwargs}
    if anthropic_api_key:
        client_kwargs["api_key"] = anthropic_api_key

    client = Anthropic(**client_kwargs)

    # Wrap the messages create method
    original_create = client.messages.create

    def attested_create(*args: Any, **kwargs: Any) -> Any:
        messages = kwargs.get("messages", [])
        model = kwargs.get("model", "unknown")
        system = kwargs.get("system")

        accumulator = ControlResultsAccumulator()

        # --- Input controls ---
        if ctx.controls_runner and ctx.controls_runner.has_input_controls:
            # Find last user message text and run input controls
            for i in range(len(messages) - 1, -1, -1):
                msg = messages[i]
                if isinstance(msg, dict) and msg.get("role") == "user":
                    if isinstance(msg.get("content"), str):
                        run_input_controls(
                            ctx.controls_runner, msg["content"], accumulator,
                        )
                    elif isinstance(msg.get("content"), list):
                        # Multi-block content — run controls on text blocks
                        for block in msg["content"]:
                            if isinstance(block, dict) and block.get("type") == "text":
                                run_input_controls(
                                    ctx.controls_runner, block.get("text", ""), accumulator,
                                )
                    break

        # Extract system prompt hash and temperature for CPR
        from glacis.crypto import hash_payload
        system_prompt_hash = hash_payload(system) if system and isinstance(system, str) else None
        temperature = kwargs.get("temperature")

        # Build input data (reused for blocking + attestation)
        input_data: dict[str, Any] = {"model": model, "messages": messages}
        if system:
            input_data["system"] = system

        # Check if input controls want to block
        check_input_block(
            ctx, accumulator, model, "anthropic", input_data,
            system_prompt_hash, temperature,
        )

        # Make the API call
        response = original_create(*args, **kwargs)

        # --- Output controls ---
        response_text = None
        if response.content:
            for block in response.content:
                if hasattr(block, "text") and block.text:
                    response_text = block.text
                    break

        if ctx.controls_runner and ctx.controls_runner.has_output_controls and response_text:
            run_output_controls(
                ctx.controls_runner, response_text, accumulator,
            )

        # Build control plane results
        control_plane_results = create_control_plane_results(
            accumulator, ctx.cfg, model, "anthropic",
            system_prompt_hash=system_prompt_hash,
            temperature=temperature,
        )

        # Handle output blocking
        check_output_block(
            ctx, accumulator, model, "anthropic", input_data, control_plane_results,
        )

        # Build output data for evidence
        output_data = {
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
        }

        md = build_metadata("anthropic", model, ctx.custom_metadata)

        # Attest and store
        attest_and_store(ctx, input_data, output_data, md, control_plane_results)

        return response

    client.messages.create = attested_create  # type: ignore
    return client


__all__ = [
    "attested_anthropic",
    "get_last_receipt",
    "get_evidence",
    "GlacisBlockedError",
]
