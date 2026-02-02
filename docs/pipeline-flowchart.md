# GLACIS Attestation Pipeline

## High-Level Flow

```mermaid
flowchart TB
    subgraph UserApp["User Application"]
        A[AI Request<br/>messages, prompt]
    end

    subgraph GlacisSDK["GLACIS SDK"]
        subgraph Controls["Control Plane (Optional)"]
            B[PII/PHI Redaction]
            C[Jailbreak Detection]
        end

        subgraph Payload["Attestation Payload"]
            P1[Input Data]
            P2[Output Data]
            P3[Control Results]
            P4["{input, output,<br/>control_plane_results}"]
        end

        subgraph Crypto["Cryptographic Layer"]
            D[RFC 8785<br/>Canonical JSON]
            E[SHA-256<br/>Hash]
            E2[64-char hex]
        end

        subgraph Attestation["Attestation Layer"]
            F{Mode?}
            G[Online Mode<br/>POST hash to api.glacis.io]
            H[Offline Mode<br/>Ed25519 Sign hash]
        end

        subgraph Storage["Local Storage"]
            I[(SQLite<br/>Evidence DB)]
        end
    end

    subgraph External["External Services"]
        J[GLACIS API<br/>Merkle Tree + Witness]
        K[AI Provider<br/>OpenAI / Anthropic]
    end

    A --> B
    B -->|Redacted Text| C
    C -->|Control Results| P3

    A -->|Original Input| P1
    K -->|Response| P2

    P1 --> P4
    P2 --> P4
    P3 --> P4

    P4 --> D
    D --> E
    E --> E2
    E2 -->|Hash Only| F
    F -->|Online| G
    F -->|Offline| H
    G --> J
    J -->|Receipt| I
    H -->|Receipt| I

    A -.->|Forward Request| K

    style P4 fill:#f9f,stroke:#333,stroke-width:2px
    style E2 fill:#ff9,stroke:#333,stroke-width:2px
```

## Detailed Integration Flow (OpenAI Example)

```mermaid
sequenceDiagram
    participant App as Application
    participant Wrap as attested_openai()
    participant Ctrl as ControlsRunner
    participant Crypto as crypto.py
    participant Client as Glacis Client
    participant Store as ReceiptStorage
    participant API as GLACIS API
    participant OAI as OpenAI API

    App->>Wrap: client.chat.completions.create()

    Note over Wrap: Extract messages from request

    alt Controls Enabled
        Wrap->>Ctrl: runner.run(user_message)
        Ctrl->>Ctrl: PIIControl.check()
        Ctrl->>Ctrl: JailbreakControl.check()
        Ctrl-->>Wrap: [ControlResult, ControlResult]

        alt Jailbreak Detected (action=block)
            Wrap-->>App: raise GlacisBlockedError
        end
    end

    Note over Wrap: Build attestation payload

    Wrap->>OAI: Forward request to OpenAI
    OAI-->>Wrap: ChatCompletion response

    Wrap->>Crypto: hash_payload({input, output})
    Crypto->>Crypto: canonical_json() → RFC 8785
    Crypto->>Crypto: sha256() → 64-char hex
    Crypto-->>Wrap: payload_hash

    alt Online Mode
        Wrap->>Client: attest(service_id, hash, ...)
        Client->>API: POST /v1/attest {hash}
        API-->>Client: AttestReceipt
        Client-->>Wrap: receipt
    else Offline Mode
        Wrap->>Client: attest(service_id, hash, ...)
        Client->>Crypto: ed25519.sign(hash)
        Crypto-->>Client: signature
        Client-->>Wrap: OfflineAttestReceipt
    end

    Wrap->>Store: store_evidence(receipt, input, output, control_results)
    Store-->>Wrap: OK

    Note over Wrap: Store receipt in thread-local

    Wrap-->>App: ChatCompletion response

    Note over App: Call get_last_receipt() for audit
```

## Control Plane Pipeline

```mermaid
flowchart LR
    subgraph Input
        A[Raw Text]
    end

    subgraph PIIControl["PII Control"]
        B{Mode?}
        C[Fast Mode<br/>Regex Patterns]
        D[Full Mode<br/>Presidio NER]
        E[Redacted Text<br/>SSN: [US_SSN]]
    end

    subgraph JailbreakControl["Jailbreak Control"]
        F[PromptGuard<br/>Model]
        G{Score > Threshold?}
        H[action: pass]
        I[action: warn]
        J[action: block]
    end

    subgraph Output
        K[ControlResult[]]
    end

    A --> B
    B -->|fast| C
    B -->|full| D
    C --> E
    D --> E
    E --> F
    F --> G
    G -->|No| H
    G -->|Yes, warn| I
    G -->|Yes, block| J
    H --> K
    I --> K
    J --> K
```

## Online vs Offline Attestation

```mermaid
flowchart TB
    subgraph Online["Online Mode"]
        O1[Hash Payload]
        O2[POST to api.glacis.io]
        O3[Server Witnesses]
        O4[Add to Merkle Tree]
        O5[Return Signed Receipt]
        O6[verify_url for audit]

        O1 --> O2 --> O3 --> O4 --> O5 --> O6
    end

    subgraph Offline["Offline Mode"]
        F1[Hash Payload]
        F2[Ed25519 Sign Locally]
        F3[Generate oatt_ ID]
        F4[Store in SQLite]
        F5[witness_status: UNVERIFIED]

        F1 --> F2 --> F3 --> F4 --> F5
    end

    subgraph Verification["Later Verification"]
        V1[Load Receipt]
        V2{Online or Offline?}
        V3[Verify via API<br/>Merkle Proof]
        V4[Verify Signature<br/>Locally]
        V5[Valid + VERIFIED]
        V6[Valid + UNVERIFIED]

        V1 --> V2
        V2 -->|att_| V3 --> V5
        V2 -->|oatt_| V4 --> V6
    end
```

## Evidence Storage Schema

```mermaid
erDiagram
    RECEIPTS {
        string attestation_id PK
        string payload_hash
        string signature
        string public_key
        string witness_status
        datetime created_at
    }

    EVIDENCE {
        string attestation_id PK
        string attestation_hash
        string mode
        string service_id
        string operation_type
        datetime timestamp
        json input_data
        json output_data
        json control_plane_results
        json metadata
    }

    RECEIPTS ||--o| EVIDENCE : "attestation_id"
```

## Full Pipeline Summary

```mermaid
flowchart TB
    subgraph Phase1["1. Request Interception"]
        A[User calls client.chat.completions.create]
    end

    subgraph Phase2["2. Control Plane"]
        B[PII Detection & Redaction]
        C[Jailbreak Detection]
        D{Block?}
    end

    subgraph Phase3["3. Hash Locally"]
        E[RFC 8785 Canonical JSON]
        F[SHA-256 Hash]
        G[64-char hex string]
    end

    subgraph Phase4["4. Prove Globally"]
        H{Online?}
        I[POST hash to GLACIS API]
        J[Ed25519 Sign Locally]
        K[Receive/Generate Receipt]
    end

    subgraph Phase5["5. Store Evidence"]
        L[Full payload → SQLite]
        M[Receipt → Thread-local]
    end

    subgraph Phase6["6. Forward Request"]
        N[Call AI Provider]
        O[Return Response]
    end

    A --> B --> C --> D
    D -->|Yes| X[GlacisBlockedError]
    D -->|No| E --> F --> G --> H
    H -->|Yes| I --> K
    H -->|No| J --> K
    K --> L --> M --> N --> O
```
