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
    config: Optional[str] = None,
    input_controls: Optional[list["BaseControl"]] = None,
    output_controls: Optional[list["BaseControl"]] = None,
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
        config: Path to glacis.yaml config file
        input_controls: Custom controls for input stage
        output_controls: Custom controls for output stage
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
        offline=offline,
        glacis_api_key=glacis_api_key,
        default_service_id="anthropic",
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
        if controls_runner and controls_runner.has_input_controls:
            # Find last user message text and run input controls
            for i in range(len(messages) - 1, -1, -1):
                msg = messages[i]
                if isinstance(msg, dict) and msg.get("role") == "user":
                    if isinstance(msg.get("content"), str):
                        run_input_controls(
                            controls_runner, msg["content"], accumulator,
                        )
                    elif isinstance(msg.get("content"), list):
                        # Multi-block content — run controls on text blocks
                        for block in msg["content"]:
                            if isinstance(block, dict) and block.get("type") == "text":
                                run_input_controls(
                                    controls_runner, block.get("text", ""), accumulator,
                                )
                    break

        # Extract system prompt hash and temperature for CPR
        from glacis.crypto import hash_payload
        system_prompt_hash = hash_payload(system) if system and isinstance(system, str) else None
        temperature = kwargs.get("temperature")

        # Check if input controls want to block
        if accumulator.should_block:
            control_plane_results = create_control_plane_results(
                accumulator, cfg, model, "anthropic",
                system_prompt_hash=system_prompt_hash,
                temperature=temperature,
            )
            blocking = accumulator.get_blocking_control()
            handle_blocked_request(
                glacis_client=glacis,
                service_id=effective_service_id,
                input_data={"model": model, "messages": messages, "system": system},
                control_plane_results=control_plane_results,
                provider="anthropic",
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
        if response.content:
            for block in response.content:
                if hasattr(block, "text") and block.text:
                    response_text = block.text
                    break

        if controls_runner and controls_runner.has_output_controls and response_text:
            run_output_controls(
                controls_runner, response_text, accumulator,
            )

        # Build control plane results
        control_plane_results = create_control_plane_results(
            accumulator, cfg, model, "anthropic",
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
                    input_data={"model": model, "messages": messages, "system": system},
                    control_plane_results=control_plane_results,
                    provider="anthropic",
                    model=model,
                    blocking_control_type=blocking.type if blocking else "unknown",
                    blocking_score=blocking.score if blocking else None,
                    debug=debug,
                    storage_backend=_storage_backend,
                    storage_path=_storage_path,
                )

        # Build input/output data for evidence
        input_data: dict[str, Any] = {"model": model, "messages": messages}
        if system:
            input_data["system"] = system

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

        metadata: dict[str, Any] = {"provider": "anthropic", "model": model}

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

    client.messages.create = attested_create  # type: ignore
    return client


__all__ = [
    "attested_anthropic",
    "get_last_receipt",
    "get_evidence",
    "GlacisBlockedError",
]
