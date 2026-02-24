"""
Demo test: Generate QA pairs with OpenAI, attest the batch, then decompose
and attest each individual pair.

Shows the full GLACIS flow:
  1. Call OpenAI to generate QA pairs from a source document
  2. Attest the entire batch (input=document, output=all QA pairs)
  3. Decompose the batch into individual QA pairs, each getting its own attestation
  4. All attestations share the same operation_id for traceability
  5. Evidence is stored locally for each attestation

Run:
    OPENAI_API_KEY=sk-xxx pytest tests/test_qa_generation_demo.py -v -s

Skip controls (faster):
    OPENAI_API_KEY=sk-xxx GLACIS_NO_CONTROLS=1 pytest tests/test_qa_generation_demo.py -v -s
"""

import json
import os
import tempfile
from pathlib import Path

import pytest

# Skip entire module if no OpenAI key
pytestmark = pytest.mark.skipif(
    not os.environ.get("OPENAI_API_KEY"),
    reason="OPENAI_API_KEY not set",
)


SAMPLE_DOCUMENT = """
MEDICATION SAFETY GUIDELINES — Warfarin Management

1. INR Monitoring
   Patients on warfarin should have their INR checked at least every 4 weeks
   once stable. Target INR for atrial fibrillation is typically 2.0–3.0.

2. Drug Interactions
   NSAIDs (e.g., ibuprofen, naproxen) significantly increase bleeding risk
   when combined with warfarin. Acetaminophen is preferred for pain management.
   Vitamin K–rich foods (leafy greens) can reduce warfarin effectiveness.

3. Patient Education
   Patients should report any unusual bruising, blood in urine or stool,
   or prolonged bleeding from cuts. Alcohol consumption should be limited
   to no more than 1–2 drinks per day.

4. Dose Adjustments
   INR above 3.5 without bleeding: hold 1–2 doses and recheck in 2–3 days.
   INR above 5.0: hold warfarin, consider vitamin K 1–2.5 mg orally,
   recheck INR in 24 hours.
"""

QA_SYSTEM_PROMPT = """You are a medical education QA generator.
Given a clinical document, generate exactly 4 question-answer pairs that test
understanding of the key clinical concepts.

Respond with valid JSON only — an array of objects with "question" and "answer" keys.
Example: [{"question": "...", "answer": "..."}, ...]"""


@pytest.fixture
def signing_seed():
    return os.urandom(32)


@pytest.fixture
def temp_storage(tmp_path):
    """Temporary JSON evidence storage directory."""
    return tmp_path / "glacis_evidence"


def test_qa_generation_and_attestation(signing_seed, temp_storage, capsys):
    """Full flow: generate QA pairs, attest batch, decompose, attest each pair."""
    from glacis import Glacis
    from glacis.storage import create_storage

    # --- Setup -----------------------------------------------------------
    glacis = Glacis(
        mode="offline",
        signing_seed=signing_seed,
        storage_backend="json",
        storage_path=temp_storage,
    )
    evidence_store = create_storage(backend="json", path=temp_storage)

    skip_controls = os.environ.get("GLACIS_NO_CONTROLS")
    if skip_controls:
        # Faster: skip PII/jailbreak controls, just do attestation
        from openai import OpenAI
        openai_client = OpenAI()
    else:
        from glacis.integrations.openai import attested_openai, get_last_receipt
        openai_client = attested_openai(
            openai_api_key=os.environ["OPENAI_API_KEY"],
            offline=True,
            signing_seed=signing_seed,
            debug=True,
            config=None,  # use default glacis.yaml
        )

    # --- Step 1: Generate QA pairs via OpenAI ----------------------------
    print("\n" + "=" * 70)
    print("STEP 1: Generating QA pairs from clinical document")
    print("=" * 70)

    response = openai_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": QA_SYSTEM_PROMPT},
            {"role": "user", "content": f"Generate QA pairs from this document:\n\n{SAMPLE_DOCUMENT}"},
        ],
        temperature=0.3,
        max_tokens=1000,
    )

    raw_output = response.choices[0].message.content
    print(f"\nRaw LLM output:\n{raw_output}\n")

    # Parse the QA pairs
    qa_pairs = json.loads(raw_output)
    assert isinstance(qa_pairs, list), "Expected a JSON array of QA pairs"
    assert len(qa_pairs) >= 2, f"Expected at least 2 QA pairs, got {len(qa_pairs)}"

    for i, pair in enumerate(qa_pairs):
        assert "question" in pair, f"QA pair {i} missing 'question'"
        assert "answer" in pair, f"QA pair {i} missing 'answer'"

    print(f"Parsed {len(qa_pairs)} QA pairs")
    for i, pair in enumerate(qa_pairs):
        print(f"  Q{i+1}: {pair['question'][:80]}...")

    # --- Step 2: Attest the full batch -----------------------------------
    print("\n" + "=" * 70)
    print("STEP 2: Attesting the full batch")
    print("=" * 70)

    op = glacis.operation()

    batch_input = {
        "document": SAMPLE_DOCUMENT,
        "system_prompt": QA_SYSTEM_PROMPT,
        "model": "gpt-4o-mini",
        "temperature": 0.3,
    }
    batch_output = {"qa_pairs": qa_pairs}

    batch_receipt = glacis.attest(
        service_id="qa-generator",
        operation_type="completion",
        input=batch_input,
        output=batch_output,
        metadata={"model": "gpt-4o-mini", "pair_count": str(len(qa_pairs))},
        operation_id=op.operation_id,
        operation_sequence=op.next_sequence(),
    )

    # Store full evidence locally (input + output for audit trail)
    evidence_store.store_evidence(
        attestation_id=batch_receipt.id,
        attestation_hash=batch_receipt.evidence_hash,
        mode="offline",
        service_id="qa-generator",
        operation_type="completion",
        timestamp=batch_receipt.timestamp or 0,
        input_data=batch_input,
        output_data=batch_output,
    )

    print(f"  Batch attestation ID:  {batch_receipt.id}")
    print(f"  Operation ID:          {batch_receipt.operation_id}")
    print(f"  Operation Sequence:    {batch_receipt.operation_sequence}")
    print(f"  Evidence Hash:         {batch_receipt.evidence_hash[:32]}...")
    print(f"  Witness Status:        {batch_receipt.witness_status}")

    # Verify batch receipt signature
    batch_verify = glacis.verify(batch_receipt)
    assert batch_verify.valid, "Batch attestation signature should be valid"
    assert batch_verify.signature_valid, "Batch signature check failed"
    print(f"  Signature Valid:       {batch_verify.signature_valid}")

    # --- Step 3: Decompose into individual QA pair attestations ----------
    print("\n" + "=" * 70)
    print("STEP 3: Decomposing batch into individual QA pair attestations")
    print("=" * 70)

    pair_receipts = glacis.decompose(
        attestation=batch_receipt,
        items=qa_pairs,
        operation_type="qa_pair",
        source_data={"document": SAMPLE_DOCUMENT},
    )

    assert len(pair_receipts) == len(qa_pairs)

    for i, receipt in enumerate(pair_receipts):
        print(f"\n  QA Pair {i+1}:")
        print(f"    Attestation ID:      {receipt.id}")
        print(f"    Operation ID:        {receipt.operation_id}")
        print(f"    Operation Sequence:  {receipt.operation_sequence}")
        print(f"    Evidence Hash:       {receipt.evidence_hash[:32]}...")

        # All share the same operation_id as the batch
        assert receipt.operation_id == batch_receipt.operation_id, \
            "Decomposed attestation should share parent's operation_id"

        # Sequences increment from parent
        assert receipt.operation_sequence == batch_receipt.operation_sequence + 1 + i

        # Each has a valid signature
        verify_result = glacis.verify(receipt)
        assert verify_result.valid, f"QA pair {i} attestation should be valid"

        # Store evidence for each decomposed pair
        evidence_store.store_evidence(
            attestation_id=receipt.id,
            attestation_hash=receipt.evidence_hash,
            mode="offline",
            service_id="qa-generator",
            operation_type="qa_pair",
            timestamp=receipt.timestamp or 0,
            input_data={"document": SAMPLE_DOCUMENT, "parent_attestation_id": batch_receipt.id},
            output_data=qa_pairs[i],
        )

    # --- Step 4: Verify evidence storage ---------------------------------
    print("\n" + "=" * 70)
    print("STEP 4: Checking local evidence storage")
    print("=" * 70)

    # Check batch evidence
    batch_evidence = evidence_store.get_evidence(batch_receipt.id)
    assert batch_evidence is not None, "Batch evidence should be stored"
    assert "qa_pairs" in batch_evidence["output"]
    print(f"  Batch evidence stored:   {batch_receipt.id}")

    # Check each pair's evidence
    for i, receipt in enumerate(pair_receipts):
        pair_evidence = evidence_store.get_evidence(receipt.id)
        assert pair_evidence is not None, f"QA pair {i} evidence should be stored"
        print(f"  Pair {i+1} evidence stored: {receipt.id}")
        # Each pair's evidence should have the QA content
        assert "question" in pair_evidence["output"]
        assert "answer" in pair_evidence["output"]

    total_attestations = 1 + len(pair_receipts)
    print(f"\n  Total attestations: {total_attestations} "
          f"(1 batch + {len(pair_receipts)} individual pairs)")
    print(f"  All linked by operation_id: {batch_receipt.operation_id}")

    # --- Summary ---------------------------------------------------------
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"  Document:           {len(SAMPLE_DOCUMENT)} chars")
    print(f"  QA pairs generated: {len(qa_pairs)}")
    print(f"  Batch attestation:  {batch_receipt.id}")
    print(f"  Pair attestations:  {len(pair_receipts)}")
    print(f"  Operation ID:       {batch_receipt.operation_id}")
    print(f"  All signatures:     VALID")
    print(f"  Evidence stored:    {temp_storage}")

    glacis.close()
