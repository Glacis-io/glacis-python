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
    create_control_plane_results,
    create_controls_runner,
    create_glacis_client,
    get_evidence,
    get_last_receipt,
    handle_blocked_request,
    initialize_config,
    run_input_controls,
    run_output_controls,
    set_last_receipt,
    store_evidence,
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
    config: Optional[str] = None,
    input_controls: Optional[list["BaseControl"]] = None,
    output_controls: Optional[list["BaseControl"]] = None,
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
        config: Path to glacis.yaml config file
        input_controls: Custom controls for input stage
        output_controls: Custom controls for output stage
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
        offline=offline,
        glacis_api_key=glacis_api_key,
        default_service_id="openai",
        service_id=service_id,
    )

    # Create controls runner and Glacis client
    controls_runner = create_controls_runner(
        cfg, debug,
        input_controls=input_controls,
        output_controls=output_controls,
    )
    _storage_backend = cfg.evidence_storage.backend
    _storage_path = cfg.evidence_storage.path
    _output_block_action = cfg.controls.output_block_action
    glacis = create_glacis_client(
        offline=effective_offline,
        signing_seed=signing_seed,
        glacis_api_key=glacis_api_key,
        glacis_base_url=glacis_base_url,
        debug=debug,
        storage_backend=_storage_backend,
        storage_path=_storage_path,
        sampling_config=cfg.sampling,
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
        if controls_runner and controls_runner.has_input_controls:
            # Find last user message text and run controls
            for i in range(len(messages) - 1, -1, -1):
                msg = messages[i]
                if (
                    isinstance(msg, dict)
                    and msg.get("role") == "user"
                    and isinstance(msg.get("content"), str)
                ):
                    run_input_controls(
                        controls_runner, msg["content"], accumulator,
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

        # Check if input controls want to block
        if accumulator.should_block:
            control_plane_results = create_control_plane_results(
                accumulator, cfg, model, "openai",
                system_prompt_hash=system_prompt_hash,
                temperature=temperature,
            )
            blocking = accumulator.get_blocking_control()
            handle_blocked_request(
                glacis_client=glacis,
                service_id=effective_service_id,
                input_data={"model": model, "messages": messages},
                control_plane_results=control_plane_results,
                provider="openai",
                model=model,
                blocking_control_type=blocking.type if blocking else "unknown",
                blocking_score=blocking.score if blocking else None,
                debug=debug,
                storage_backend=_storage_backend,
                storage_path=_storage_path,
            )

        # Make the API call
        response = original_create(*args, **kwargs)

        # --- Output controls ---
        response_text = None
        if response.choices:
            response_text = response.choices[0].message.content

        if controls_runner and controls_runner.has_output_controls and response_text:
            run_output_controls(
                controls_runner, response_text, accumulator,
            )

        # Build control plane results (includes both input + output controls)
        control_plane_results = create_control_plane_results(
            accumulator, cfg, model, "openai",
            system_prompt_hash=system_prompt_hash,
            temperature=temperature,
        )

        # Handle output blocking
        if accumulator.should_block and controls_runner and controls_runner.has_output_controls:
            if _output_block_action == "block":
                blocking = accumulator.get_blocking_control()
                handle_blocked_request(
                    glacis_client=glacis,
                    service_id=effective_service_id,
                    input_data={"model": model, "messages": messages},
                    control_plane_results=control_plane_results,
                    provider="openai",
                    model=model,
                    blocking_control_type=blocking.type if blocking else "unknown",
                    blocking_score=blocking.score if blocking else None,
                    debug=debug,
                    storage_backend=_storage_backend,
                    storage_path=_storage_path,
                )
            # "forward" mode: continue, determination will show "blocked" in attestation

        # Build input/output data for evidence
        input_data = {"model": model, "messages": messages}
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

        metadata: dict[str, Any] = {"provider": "openai", "model": model}

        # Attest and store
        try:
            receipt = glacis.attest(
                service_id=effective_service_id,
                operation_type="completion",
                input=input_data,
                output=output_data,
                metadata=metadata,
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
                metadata=metadata,
                debug=debug,
                storage_backend=_storage_backend,
                storage_path=_storage_path,
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
