"""
GLACIS integration for Google Gemini.

Provides an attested Gemini client wrapper that automatically:
1. Runs enabled controls (PII/PHI redaction, jailbreak detection, etc.)
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
    from google.genai import Client


def attested_gemini(
    glacis_api_key: Optional[str] = None,
    gemini_api_key: Optional[str] = None,
    glacis_base_url: str = "https://api.glacis.io",
    service_id: str = "gemini",
    debug: bool = False,
    offline: Optional[bool] = None,
    signing_seed: Optional[bytes] = None,
    redaction: Union[bool, Literal["fast", "full"], None] = None,
    config: Optional[str] = None,
    **gemini_kwargs: Any,
) -> "Client":
    """
    Create an attested Gemini client with controls (PII redaction, jailbreak detection).

    Args:
        glacis_api_key: GLACIS API key (required for online mode)
        gemini_api_key: Google Gemini API key (default: from GOOGLE_API_KEY env var)
        glacis_base_url: GLACIS API base URL
        service_id: Service identifier for attestations
        debug: Enable debug logging
        offline: Enable offline mode (local signing, no server)
        signing_seed: 32-byte Ed25519 signing seed (required for offline mode)
        redaction: PII/PHI redaction mode - "fast", "full", True, False, or None
        config: Path to glacis.yaml config file
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

    # Initialize config and determine modes
    cfg, effective_offline, effective_service_id = initialize_config(
        config_path=config,
        redaction=redaction,
        offline=offline,
        glacis_api_key=glacis_api_key,
        default_service_id="gemini",
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
                system_instruction = config_param.system_instruction

        # Run controls if enabled
        if controls_runner:
            from glacis.integrations.base import (
                ControlResultsAccumulator,
                create_control_plane_attestation_from_accumulator,
                handle_blocked_request,
                process_text_for_controls,
            )

            accumulator = ControlResultsAccumulator()

            # Process system_instruction through controls if it's a string
            if system_instruction and isinstance(system_instruction, str):
                final_system = process_text_for_controls(
                    controls_runner, system_instruction, accumulator
                )
                # Update config with redacted system_instruction
                if isinstance(config_param, dict):
                    kwargs["config"] = {**config_param, "system_instruction": final_system}
                elif hasattr(config_param, "system_instruction"):
                    config_param.system_instruction = final_system

            # Process contents through controls
            if isinstance(contents, str):
                # Simple string prompt
                final_text = process_text_for_controls(
                    controls_runner, contents, accumulator
                )
                if "contents" in kwargs:
                    kwargs["contents"] = final_text
                elif args:
                    args = (final_text,) + args[1:]
                contents = final_text
            elif isinstance(contents, list):
                # List of Content objects or parts - find last user text
                processed_contents = list(contents)
                for i in range(len(processed_contents) - 1, -1, -1):
                    item = processed_contents[i]
                    if isinstance(item, str):
                        final_text = process_text_for_controls(
                            controls_runner, item, accumulator
                        )
                        processed_contents[i] = final_text
                        break
                    elif isinstance(item, dict):
                        # ContentDict with role/parts
                        role = item.get("role", "")
                        if role == "user":
                            parts = item.get("parts", [])
                            for j in range(len(parts) - 1, -1, -1):
                                part = parts[j]
                                if isinstance(part, str):
                                    final_text = process_text_for_controls(
                                        controls_runner, part, accumulator
                                    )
                                    parts[j] = final_text
                                    break
                                elif isinstance(part, dict) and "text" in part:
                                    final_text = process_text_for_controls(
                                        controls_runner, part["text"], accumulator
                                    )
                                    parts[j] = {**part, "text": final_text}
                                    break
                            processed_contents[i] = {**item, "parts": parts}
                            break
                    elif hasattr(item, "role") and hasattr(item, "parts"):
                        # Content object
                        if item.role == "user" and item.parts:
                            for j in range(len(item.parts) - 1, -1, -1):
                                part = item.parts[j]
                                if hasattr(part, "text") and part.text:
                                    final_text = process_text_for_controls(
                                        controls_runner, part.text, accumulator
                                    )
                                    item.parts[j].text = final_text
                                    break
                            break

                if "contents" in kwargs:
                    kwargs["contents"] = processed_contents
                elif args:
                    args = (processed_contents,) + args[1:]
                contents = processed_contents

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
                accumulator, cfg, model, "gemini", "models.generate_content"
            )

            # Check if we need to block BEFORE making the API call
            if accumulator.should_block:
                handle_blocked_request(
                    glacis_client=glacis,
                    service_id=effective_service_id,
                    input_data={"model": model, "contents": _serialize_contents(contents)},
                    control_plane_results=control_plane_results,
                    provider="gemini",
                    model=model,
                    jailbreak_score=accumulator.jailbreak_summary.score
                    if accumulator.jailbreak_summary
                    else 0.0,
                    debug=debug,
                )
        else:
            control_plane_results = None

        # Make the API call (only if not blocked)
        response = original_generate_content(*args, **kwargs)

        # Build input/output data
        input_data: dict[str, Any] = {
            "model": model,
            "contents": _serialize_contents(contents),
        }
        if system_instruction:
            input_data["system_instruction"] = (
                system_instruction if isinstance(system_instruction, str)
                else str(system_instruction)
            )

        output_data: dict[str, Any] = {
            "model_version": response.model_version,
            "candidates": [],
            "usage": None,
        }

        if response.candidates:
            for candidate in response.candidates:
                finish = (
                    str(candidate.finish_reason) if candidate.finish_reason else None
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
                "prompt_tokens": response.usage_metadata.prompt_token_count or 0,
                "candidates_tokens": response.usage_metadata.candidates_token_count or 0,
                "total_tokens": response.usage_metadata.total_token_count or 0,
            }

        # Attest and store
        try:
            receipt = glacis.attest(
                service_id=effective_service_id,
                operation_type="completion",
                input=input_data,
                output=output_data,
                metadata={"provider": "gemini", "model": model},
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
                metadata={"provider": "gemini", "model": model},
                debug=debug,
            )
        except Exception as e:
            if debug:
                print(f"[glacis] Attestation failed: {e}")

        return response

    client.models.generate_content = attested_generate_content  # type: ignore
    return client


def _serialize_contents(contents: Any) -> Any:
    """Serialize contents to a JSON-safe format for attestation."""
    if isinstance(contents, str):
        return contents
    if isinstance(contents, list):
        result = []
        for item in contents:
            if isinstance(item, str):
                result.append(item)
            elif isinstance(item, dict):
                result.append(item)
            elif hasattr(item, "role") and hasattr(item, "parts"):
                # Content object
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
