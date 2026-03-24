"""
GLACIS integration for LiteLLM.

Provides an attested LiteLLM wrapper that automatically:
1. Runs input/output controls (PII/PHI detection, jailbreak detection, word filter, etc.)
2. Logs all completions to the GLACIS transparency log
3. Creates control plane attestations

LiteLLM is a unified proxy for 100+ LLM providers. Since it exposes module-level
functions (not a client object), this integration returns a thin wrapper object
with `.completion()` and `.acompletion()` methods.

Example:
    >>> from glacis.integrations.litellm import attested_litellm, get_last_receipt
    >>> client = attested_litellm(
    ...     offline=True,
    ...     signing_seed=os.urandom(32),
    ... )
    >>> response = client.completion(
    ...     model="gpt-4",
    ...     messages=[{"role": "user", "content": "Hello!"}]
    ... )
    >>> receipt = get_last_receipt()
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Optional

from glacis.integrations.base import (
    ControlResultsAccumulator,
    GlacisBlockedError,
    GlacisTagScope,
    GlacisOperationContext,
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
    from glacis.controls.base import BaseControl
    from glacis.integrations.base import IntegrationContext


class AttestedLiteLLM:
    """Attested LiteLLM wrapper with control pipeline + attestation.

    Wraps litellm.completion() and litellm.acompletion() with automatic
    attestation and optional input/output controls.
    """

    def __init__(self, ctx: "IntegrationContext") -> None:
        self._ctx = ctx

    def glacis_tags(self, metadata: dict[str, str]) -> GlacisTagScope:
        """Set per-call metadata for wrapper calls within this block."""
        return GlacisTagScope(metadata)

    def glacis_operation(self) -> GlacisOperationContext:
        """Create an operation scope — all calls within share an operation_id."""
        return GlacisOperationContext()

    def completion(self, *args: Any, **kwargs: Any) -> Any:
        """Attested wrapper around litellm.completion().

        Accepts the same arguments as litellm.completion().
        """
        try:
            import litellm as _litellm
        except ImportError:
            raise ImportError(
                "LiteLLM integration requires the 'litellm' package. "
                "Install it with: pip install glacis[litellm]"
            )

        ctx = self._ctx
        suppress_noisy_loggers(["litellm", "LiteLLM"])

        messages = kwargs.get("messages", args[1] if len(args) > 1 else [])
        model = kwargs.get("model", args[0] if args else "unknown")

        accumulator = ControlResultsAccumulator()

        # --- Input controls ---
        if ctx.controls_runner and ctx.controls_runner.has_input_controls:
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
            ctx, accumulator, model, "litellm", input_data,
            system_prompt_hash, temperature,
        )

        # Make the LiteLLM API call
        response = _litellm.completion(*args, **kwargs)

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
            accumulator, ctx.cfg, model, "litellm",
            system_prompt_hash=system_prompt_hash,
            temperature=temperature,
        )

        # Handle output blocking
        check_output_block(
            ctx, accumulator, model, "litellm", input_data, control_plane_results,
        )

        # Build output data for evidence (OpenAI-compatible structure)
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

        md = build_metadata("litellm", model, ctx.custom_metadata)

        # Attest and store
        attest_and_store(ctx, input_data, output_data, md, control_plane_results)

        return response

    async def acompletion(self, *args: Any, **kwargs: Any) -> Any:
        """Attested wrapper around litellm.acompletion().

        Accepts the same arguments as litellm.acompletion().
        """
        try:
            import litellm as _litellm
        except ImportError:
            raise ImportError(
                "LiteLLM integration requires the 'litellm' package. "
                "Install it with: pip install glacis[litellm]"
            )

        ctx = self._ctx
        suppress_noisy_loggers(["litellm", "LiteLLM"])

        messages = kwargs.get("messages", args[1] if len(args) > 1 else [])
        model = kwargs.get("model", args[0] if args else "unknown")

        accumulator = ControlResultsAccumulator()

        # --- Input controls ---
        if ctx.controls_runner and ctx.controls_runner.has_input_controls:
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

        from glacis.crypto import hash_payload

        system_prompt = None
        for msg in messages:
            if isinstance(msg, dict) and msg.get("role") == "system":
                system_prompt = msg.get("content", "")
                break
        system_prompt_hash = hash_payload(system_prompt) if system_prompt else None
        temperature = kwargs.get("temperature")

        input_data = {"model": model, "messages": messages}

        check_input_block(
            ctx, accumulator, model, "litellm", input_data,
            system_prompt_hash, temperature,
        )

        # Make the async LiteLLM API call
        response = await _litellm.acompletion(*args, **kwargs)

        # --- Output controls ---
        response_text = None
        if response.choices:
            response_text = response.choices[0].message.content

        if ctx.controls_runner and ctx.controls_runner.has_output_controls and response_text:
            run_output_controls(
                ctx.controls_runner, response_text, accumulator,
            )

        control_plane_results = create_control_plane_results(
            accumulator, ctx.cfg, model, "litellm",
            system_prompt_hash=system_prompt_hash,
            temperature=temperature,
        )

        check_output_block(
            ctx, accumulator, model, "litellm", input_data, control_plane_results,
        )

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

        md = build_metadata("litellm", model, ctx.custom_metadata)
        attest_and_store(ctx, input_data, output_data, md, control_plane_results)

        return response


def attested_litellm(
    glacis_api_key: Optional[str] = None,
    glacis_base_url: str = "https://api.glacis.io",
    service_id: str = "litellm",
    debug: bool = False,
    offline: Optional[bool] = None,
    signing_seed: Optional[bytes] = None,
    policy_key: Optional[bytes] = None,
    config: Optional[str] = None,
    input_controls: Optional[list["BaseControl"]] = None,
    output_controls: Optional[list["BaseControl"]] = None,
    metadata: Optional[dict[str, str]] = None,
) -> AttestedLiteLLM:
    """
    Create an attested LiteLLM client with controls.

    Args:
        glacis_api_key: GLACIS API key (required for online mode)
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

    Returns:
        AttestedLiteLLM wrapper with .completion() and .acompletion() methods

    Raises:
        GlacisBlockedError: If a control blocks the request
    """
    try:
        import litellm as _litellm  # noqa: F401
    except ImportError:
        raise ImportError(
            "LiteLLM integration requires the 'litellm' package. "
            "Install it with: pip install glacis[litellm]"
        )

    ctx = setup_integration(
        config=config,
        offline=offline,
        glacis_api_key=glacis_api_key,
        glacis_base_url=glacis_base_url,
        default_service_id="litellm",
        service_id=service_id,
        debug=debug,
        signing_seed=signing_seed,
        policy_key=policy_key,
        input_controls=input_controls,
        output_controls=output_controls,
        metadata=metadata,
    )

    return AttestedLiteLLM(ctx)


__all__ = [
    "attested_litellm",
    "AttestedLiteLLM",
    "get_last_receipt",
    "get_evidence",
    "GlacisBlockedError",
    "GlacisTagScope",
    "GlacisOperationContext",
]
