"""
GLACIS integration for Google Gemini.

Provides an attested Gemini client wrapper that automatically:
1. Runs input/output controls (PII/PHI detection, jailbreak detection, word filter, etc.)
2. Logs all completions to the GLACIS transparency log
3. Creates control plane attestations

Example:
    >>> from glacis.integrations.gemini import attested_gemini, get_last_receipt
    >>> client = attested_gemini(
    ...     gemini_api_key="...",
    ...     offline=True,
    ...     signing_seed=os.urandom(32),
    ... )
    >>> response = client.models.generate_content(
    ...     model="gemini-2.5-flash",
    ...     contents="Hello!"
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
    from google.genai import Client

    from glacis.controls.base import BaseControl


def attested_gemini(
    glacis_api_key: Optional[str] = None,
    gemini_api_key: Optional[str] = None,
    glacis_base_url: str = "https://api.glacis.io",
    service_id: str = "gemini",
    debug: bool = False,
    offline: Optional[bool] = None,
    signing_seed: Optional[bytes] = None,
    policy_key: Optional[bytes] = None,
    config: Optional[str] = None,
    input_controls: Optional[list["BaseControl"]] = None,
    output_controls: Optional[list["BaseControl"]] = None,
    metadata: Optional[dict[str, str]] = None,
    **gemini_kwargs: Any,
) -> "Client":
    """
    Create an attested Gemini client with controls.

    Args:
        glacis_api_key: GLACIS API key (required for online mode)
        gemini_api_key: Google Gemini API key (default: from GOOGLE_API_KEY env var)
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
        **gemini_kwargs: Additional arguments passed to genai.Client

    Returns:
        Wrapped Gemini client

    Raises:
        GlacisBlockedError: If a control blocks the request
    """
    # Suppress noisy loggers
    suppress_noisy_loggers(["google.genai", "google.auth", "urllib3"])

    try:
        from google import genai
    except ImportError:
        raise ImportError(
            "Gemini integration requires the 'google-genai' package. "
            "Install it with: pip install glacis[gemini]"
        )

    ctx = setup_integration(
        config=config,
        offline=offline,
        glacis_api_key=glacis_api_key,
        glacis_base_url=glacis_base_url,
        default_service_id="gemini",
        service_id=service_id,
        debug=debug,
        signing_seed=signing_seed,
        policy_key=policy_key,
        input_controls=input_controls,
        output_controls=output_controls,
        metadata=metadata,
    )

    # Create the Gemini client
    client_kwargs: dict[str, Any] = {**gemini_kwargs}
    if gemini_api_key:
        client_kwargs["api_key"] = gemini_api_key

    client = genai.Client(**client_kwargs)

    # Wrap the models.generate_content method
    original_generate_content = client.models.generate_content

    def attested_generate_content(*args: Any, **kwargs: Any) -> Any:
        contents = kwargs.get("contents", args[0] if args else None)
        model = kwargs.get("model", "unknown")
        config_param = kwargs.get("config")

        # Extract system_instruction from config if present
        system_instruction = None
        if config_param is not None:
            if isinstance(config_param, dict):
                system_instruction = config_param.get("system_instruction")
            elif hasattr(config_param, "system_instruction"):
                system_instruction = config_param.system_instruction  # type: ignore[union-attr]

        accumulator = ControlResultsAccumulator()

        # --- Input controls ---
        if ctx.controls_runner and ctx.controls_runner.has_input_controls:
            # Extract user text from contents
            user_text = _extract_user_text(contents)
            if user_text:
                run_input_controls(ctx.controls_runner, user_text, accumulator)

        # Extract system prompt hash and temperature for CPR
        from glacis.crypto import hash_payload
        _sys_prompt_hash = (
            hash_payload(system_instruction)
            if system_instruction and isinstance(system_instruction, str)
            else None
        )
        _temperature = None
        if config_param is not None:
            if isinstance(config_param, dict):
                _temperature = config_param.get("temperature")
            elif hasattr(config_param, "temperature"):
                _temperature = config_param.temperature  # type: ignore[union-attr]

        # Build input data (reused for blocking + attestation)
        input_data: dict[str, Any] = {
            "model": model,
            "contents": _serialize_contents(contents),
        }
        if system_instruction:
            input_data["system_instruction"] = (
                system_instruction
                if isinstance(system_instruction, str)
                else str(system_instruction)
            )

        # Check if input controls want to block
        check_input_block(
            ctx, accumulator, model, "gemini", input_data,
            _sys_prompt_hash, _temperature,
        )

        # Make the API call
        response = original_generate_content(*args, **kwargs)

        # --- Output controls ---
        response_text = _extract_response_text(response)

        if ctx.controls_runner and ctx.controls_runner.has_output_controls and response_text:
            run_output_controls(
                ctx.controls_runner, response_text, accumulator,
            )

        # Build control plane results
        control_plane_results = create_control_plane_results(
            accumulator, ctx.cfg, model, "gemini",
            system_prompt_hash=_sys_prompt_hash,
            temperature=_temperature,
        )

        # Handle output blocking
        check_output_block(
            ctx, accumulator, model, "gemini", input_data, control_plane_results,
        )

        # Build output data for evidence
        output_data: dict[str, Any] = {
            "model_version": response.model_version,
            "candidates": [],
            "usage": None,
        }

        if response.candidates:
            for candidate in response.candidates:
                finish = (
                    str(candidate.finish_reason)
                    if candidate.finish_reason
                    else None
                )
                candidate_data: dict[str, Any] = {
                    "finish_reason": finish,
                }
                if candidate.content and candidate.content.parts:
                    candidate_data["content"] = {
                        "role": candidate.content.role,
                        "parts": [
                            {"text": part.text}
                            for part in candidate.content.parts
                            if hasattr(part, "text") and part.text is not None
                        ],
                    }
                output_data["candidates"].append(candidate_data)

        if response.usage_metadata:
            output_data["usage"] = {
                "prompt_tokens": (
                    response.usage_metadata.prompt_token_count or 0
                ),
                "candidates_tokens": (
                    response.usage_metadata.candidates_token_count or 0
                ),
                "total_tokens": (
                    response.usage_metadata.total_token_count or 0
                ),
            }

        md = build_metadata("gemini", model, ctx.custom_metadata)

        # Attest and store
        attest_and_store(ctx, input_data, output_data, md, control_plane_results)

        return response

    client.models.generate_content = attested_generate_content  # type: ignore
    return client


def _extract_user_text(contents: Any) -> Optional[str]:
    """Extract user text from Gemini contents for control scanning."""
    if isinstance(contents, str):
        return contents
    if isinstance(contents, list):
        for i in range(len(contents) - 1, -1, -1):
            item = contents[i]
            if isinstance(item, str):
                return item
            elif isinstance(item, dict):
                if item.get("role", "") == "user":
                    parts = item.get("parts", [])
                    for part in parts:
                        if isinstance(part, str):
                            return part
                        elif isinstance(part, dict) and "text" in part:
                            return str(part["text"])
            elif hasattr(item, "role") and hasattr(item, "parts"):
                if item.role == "user" and item.parts:
                    for part in item.parts:
                        if hasattr(part, "text") and part.text:
                            return str(part.text)
    return None


def _extract_response_text(response: Any) -> Optional[str]:
    """Extract text from Gemini response for output controls."""
    if response.candidates:
        for candidate in response.candidates:
            if candidate.content and candidate.content.parts:
                for part in candidate.content.parts:
                    if hasattr(part, "text") and part.text:
                        return str(part.text)
    return None


def _serialize_contents(contents: Any) -> Any:
    """Serialize contents to a JSON-safe format for attestation."""
    if isinstance(contents, str):
        return contents
    if isinstance(contents, list):
        result: list[Any] = []
        for item in contents:
            if isinstance(item, str):
                result.append(item)
            elif isinstance(item, dict):
                result.append(item)
            elif hasattr(item, "role") and hasattr(item, "parts"):
                parts = []
                if item.parts:
                    for part in item.parts:
                        if hasattr(part, "text") and part.text is not None:
                            parts.append({"text": part.text})
                        else:
                            parts.append({"type": type(part).__name__})
                result.append({"role": item.role, "parts": parts})
            else:
                result.append(str(item))
        return result
    if hasattr(contents, "role") and hasattr(contents, "parts"):
        parts = []
        if contents.parts:
            for part in contents.parts:
                if hasattr(part, "text") and part.text is not None:
                    parts.append({"text": part.text})
                else:
                    parts.append({"type": type(part).__name__})
        return {"role": contents.role, "parts": parts}
    return str(contents)


__all__ = [
    "attested_gemini",
    "get_last_receipt",
    "get_evidence",
    "GlacisBlockedError",
]
