"""
GLACIS integration for OpenAI.

Provides an attested OpenAI client wrapper that automatically:
1. Runs input/output controls (PII/PHI detection, jailbreak detection, word filter, etc.)
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
    from openai import OpenAI

    from glacis.controls.base import BaseControl


def attested_openai(
    glacis_api_key: Optional[str] = None,
    openai_api_key: Optional[str] = None,
    glacis_base_url: str = "https://api.glacis.io",
    service_id: str = "openai",
    debug: bool = False,
    offline: Optional[bool] = None,
    signing_seed: Optional[bytes] = None,
    policy_key: Optional[bytes] = None,
    config: Optional[str] = None,
    input_controls: Optional[list["BaseControl"]] = None,
    output_controls: Optional[list["BaseControl"]] = None,
    metadata: Optional[dict[str, str]] = None,
    **openai_kwargs: Any,
) -> "OpenAI":
    """
    Create an attested OpenAI client with controls.

    Args:
        glacis_api_key: GLACIS API key (required for online mode)
        openai_api_key: OpenAI API key (default: from OPENAI_API_KEY env var)
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

    ctx = setup_integration(
        config=config,
        offline=offline,
        glacis_api_key=glacis_api_key,
        glacis_base_url=glacis_base_url,
        default_service_id="openai",
        service_id=service_id,
        debug=debug,
        signing_seed=signing_seed,
        policy_key=policy_key,
        input_controls=input_controls,
        output_controls=output_controls,
        metadata=metadata,
    )

    # Create the OpenAI client
    client_kwargs: dict[str, Any] = {**openai_kwargs}
    if openai_api_key:
        client_kwargs["api_key"] = openai_api_key

    client = OpenAI(**client_kwargs)

    # Wrap the chat completions create method
    original_create = client.chat.completions.create

    def attested_create(*args: Any, **kwargs: Any) -> Any:
        messages = kwargs.get("messages", [])
        model = kwargs.get("model", "unknown")

        accumulator = ControlResultsAccumulator()

        # --- Input controls ---
        if ctx.controls_runner and ctx.controls_runner.has_input_controls:
            # Find last user message text and run controls
            for i in range(len(messages) - 1, -1, -1):
                msg = messages[i]
                if (
                    isinstance(msg, dict)
                    and msg.get("role") == "user"
                    and isinstance(msg.get("content"), str)
                ):
                    run_input_controls(
                        ctx.controls_runner, msg["content"], accumulator,
                    )
                    break

        # Extract system prompt hash and temperature for CPR
        from glacis.crypto import hash_payload
        system_prompt = None
        for msg in messages:
            if isinstance(msg, dict) and msg.get("role") == "system":
                system_prompt = msg.get("content", "")
                break
        system_prompt_hash = hash_payload(system_prompt) if system_prompt else None
        temperature = kwargs.get("temperature")

        # Build input data (reused for blocking + attestation)
        input_data = {"model": model, "messages": messages}

        # Check if input controls want to block
        check_input_block(
            ctx, accumulator, model, "openai", input_data,
            system_prompt_hash, temperature,
        )

        # Make the API call
        response = original_create(*args, **kwargs)

        # --- Output controls ---
        response_text = None
        if response.choices:
            response_text = response.choices[0].message.content

        if ctx.controls_runner and ctx.controls_runner.has_output_controls and response_text:
            run_output_controls(
                ctx.controls_runner, response_text, accumulator,
            )

        # Build control plane results (includes both input + output controls)
        control_plane_results = create_control_plane_results(
            accumulator, ctx.cfg, model, "openai",
            system_prompt_hash=system_prompt_hash,
            temperature=temperature,
        )

        # Handle output blocking
        check_output_block(
            ctx, accumulator, model, "openai", input_data, control_plane_results,
        )

        # Build output data for evidence
        output_data = {
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
                "prompt_tokens": response.usage.prompt_tokens if response.usage else 0,
                "completion_tokens": response.usage.completion_tokens if response.usage else 0,
                "total_tokens": response.usage.total_tokens if response.usage else 0,
            } if response.usage else None,
        }

        md = build_metadata("openai", model, ctx.custom_metadata)

        # Attest and store
        attest_and_store(ctx, input_data, output_data, md, control_plane_results)

        return response

    client.chat.completions.create = attested_create  # type: ignore
    return client


__all__ = [
    "attested_openai",
    "get_last_receipt",
    "get_evidence",
    "GlacisBlockedError",
]
