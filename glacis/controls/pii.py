"""
PII/PHI Redaction Control.

HIPAA-compliant detection and redaction of the 18 Safe Harbor identifiers
using Microsoft Presidio with custom healthcare-specific recognizers.

Supported backends:
- presidio: Microsoft Presidio (default)

Two modes:
- fast: Regex-only detection (<2ms typical)
- full: Regex + spaCy NER (~15-20ms typical)
"""

from __future__ import annotations

import logging
import time
import warnings
from typing import TYPE_CHECKING, Any, Optional

from glacis.controls.base import BaseControl, ControlResult

if TYPE_CHECKING:
    from presidio_analyzer import AnalyzerEngine, RecognizerResult
    from presidio_anonymizer import AnonymizerEngine

    from glacis.config import PiiPhiConfig

logger = logging.getLogger("glacis.controls.pii")


# Supported backends for PII detection
SUPPORTED_BACKENDS = ["presidio"]


class PIIControl(BaseControl):
    """
    PII/PHI detection and redaction control.

    Uses Microsoft Presidio with custom recognizers for the 18 HIPAA Safe Harbor
    identifiers. Supports two operating modes:

    - "fast": Regex-only detection, typically <2ms
    - "full": Regex + spaCy NER for improved name/location detection, ~15-20ms

    Args:
        config: PiiPhiConfig with enabled, backend, and mode settings

    Example:
        >>> from glacis.config import PiiPhiConfig
        >>> config = PiiPhiConfig(enabled=True, backend="presidio", mode="fast")
        >>> control = PIIControl(config)
        >>> result = control.check("SSN: 123-45-6789")
        >>> result.detected
        True
        >>> result.modified_text
        "SSN: [US_SSN]"
    """

    control_type = "pii"

    # HIPAA Safe Harbor entity types
    HIPAA_ENTITIES: list[str] = [
        # Presidio OOTB
        "PERSON",
        "DATE_TIME",
        "PHONE_NUMBER",
        "EMAIL_ADDRESS",
        "US_SSN",
        "US_DRIVER_LICENSE",
        "URL",
        "IP_ADDRESS",
        "CREDIT_CARD",
        "US_BANK_NUMBER",
        "IBAN_CODE",
        "US_PASSPORT",
        "US_ITIN",
        # Custom HIPAA-specific
        "MEDICAL_RECORD_NUMBER",
        "HEALTH_PLAN_BENEFICIARY",
        "NPI",
        "DEA_NUMBER",
        "MEDICAL_LICENSE",
        "US_ZIP_CODE",
        "STREET_ADDRESS",
        "VIN",
        "LICENSE_PLATE",
        "DEVICE_SERIAL",
        "UDI",
        "IMEI",
        "FAX_NUMBER",
        "BIOMETRIC_ID",
        "UUID",
    ]

    def __init__(self, config: "PiiPhiConfig") -> None:
        self._config = config
        self._mode = config.mode

        # Validate backend
        if config.backend not in SUPPORTED_BACKENDS:
            raise ValueError(
                f"Unknown PII backend: {config.backend}. "
                f"Available: {SUPPORTED_BACKENDS}"
            )

        # Default threshold: higher for "full" mode to reduce NER false positives
        if self._mode == "full":
            self._score_threshold = 0.7
        else:
            self._score_threshold = 0.5

        self._analyzer: Optional["AnalyzerEngine"] = None
        self._anonymizer: Optional["AnonymizerEngine"] = None
        self._spacy_available: bool = False
        self._initialized: bool = False

    def _ensure_initialized(self) -> None:
        """Lazy initialization of Presidio engines."""
        if self._initialized:
            return

        # Suppress noisy loggers from Presidio and spaCy
        for logger_name in [
            "presidio-analyzer",
            "presidio-anonymizer",
            "presidio_analyzer",
            "presidio_anonymizer",
            "spacy",
        ]:
            logging.getLogger(logger_name).setLevel(logging.WARNING)

        try:
            from presidio_analyzer import AnalyzerEngine, RecognizerRegistry
            from presidio_anonymizer import AnonymizerEngine
        except ImportError as e:
            raise ImportError(
                "PII control requires presidio-analyzer and presidio-anonymizer. "
                "Install with: pip install glacis[redaction]"
            ) from e

        registry = RecognizerRegistry()

        if self._mode == "fast":
            # Fast mode: Only pattern-based recognizers, NO spaCy
            for recognizer in self._build_core_pattern_recognizers():
                registry.add_recognizer(recognizer)
        else:
            # Full mode: Load all predefined recognizers including SpacyRecognizer
            registry.load_predefined_recognizers()
            for recognizer in self._build_core_pattern_recognizers():
                registry.add_recognizer(recognizer)

        # Add custom HIPAA recognizers
        for recognizer in self._build_healthcare_recognizers():
            registry.add_recognizer(recognizer)
        for recognizer in self._build_geographic_recognizers():
            registry.add_recognizer(recognizer)
        for recognizer in self._build_identifier_recognizers():
            registry.add_recognizer(recognizer)

        # Configure NLP engine based on mode
        if self._mode == "full":
            nlp_engine = self._try_load_spacy()
            if nlp_engine:
                self._analyzer = AnalyzerEngine(
                    registry=registry,
                    nlp_engine=nlp_engine,
                    supported_languages=["en"],
                )
                self._spacy_available = True
            else:
                warnings.warn(
                    "spaCy model 'en_core_web_md' not available. "
                    "Falling back to 'fast' mode (regex-only). "
                    "Install with: python -m spacy download en_core_web_md",
                    UserWarning,
                    stacklevel=2,
                )
                self._analyzer = AnalyzerEngine(
                    registry=registry,
                    supported_languages=["en"],
                )
                self._spacy_available = False
        else:
            self._analyzer = AnalyzerEngine(
                registry=registry,
                supported_languages=["en"],
            )
            self._spacy_available = False

        self._anonymizer = AnonymizerEngine()
        self._initialized = True

    def _try_load_spacy(self) -> Optional[Any]:
        """Attempt to load spaCy NLP engine."""
        try:
            from presidio_analyzer.nlp_engine import NlpEngineProvider

            configuration = {
                "nlp_engine_name": "spacy",
                "models": [{"lang_code": "en", "model_name": "en_core_web_md"}],
            }
            provider = NlpEngineProvider(nlp_configuration=configuration)
            return provider.create_engine()
        except Exception as e:
            logger.debug(f"Failed to load spaCy: {e}")
            return None

    def check(self, text: str) -> ControlResult:
        """
        Check text for PII/PHI and redact if found.

        Args:
            text: Input text to check

        Returns:
            ControlResult with detection info and redacted text
        """
        self._ensure_initialized()

        start_time = time.perf_counter()

        if not text or not text.strip():
            return ControlResult(
                control_type=self.control_type,
                detected=False,
                action="pass",
                categories=[],
                latency_ms=0,
                modified_text=text,
                metadata={"backend": self._config.backend, "mode": self._mode},
            )

        assert self._analyzer is not None
        assert self._anonymizer is not None

        results: list["RecognizerResult"] = self._analyzer.analyze(
            text=text,
            language="en",
            score_threshold=self._score_threshold,
        )

        latency_ms = int((time.perf_counter() - start_time) * 1000)

        if not results:
            return ControlResult(
                control_type=self.control_type,
                detected=False,
                action="pass",
                categories=[],
                latency_ms=latency_ms,
                modified_text=text,
                metadata={"backend": self._config.backend, "mode": self._mode},
            )

        # De-duplicate overlapping detections
        results = self._resolve_overlaps(results)

        # Build operators for replacement format [ENTITY_TYPE]
        from presidio_anonymizer.entities import OperatorConfig

        operators = {}
        for entity_type in set(r.entity_type for r in results):
            operators[entity_type] = OperatorConfig(
                "replace", {"new_value": f"[{entity_type}]"}
            )

        anonymized = self._anonymizer.anonymize(
            text=text,
            analyzer_results=results,
            operators=operators,
        )

        categories = sorted(set(r.entity_type for r in results))

        return ControlResult(
            control_type=self.control_type,
            detected=True,
            action="redact",
            categories=categories,
            latency_ms=latency_ms,
            modified_text=anonymized.text,
            metadata={
                "backend": self._config.backend,
                "mode": self._mode,
                "count": len(results),
            },
        )

    def close(self) -> None:
        """Release resources."""
        self._analyzer = None
        self._anonymizer = None
        self._initialized = False

    def _resolve_overlaps(self, results: list[Any]) -> list[Any]:
        """Resolve overlapping detections by keeping highest confidence."""
        if not results:
            return results

        sorted_results = sorted(
            results, key=lambda r: (r.start, -r.score, -(r.end - r.start))
        )

        merged: list[Any] = []
        for result in sorted_results:
            overlaps = False
            for kept in merged:
                if result.start < kept.end and result.end > kept.start:
                    overlaps = True
                    break

            if not overlaps:
                merged.append(result)

        return merged

    # =========================================================================
    # Pattern Recognizer Builders (from original RedactionEngine)
    # =========================================================================

    def _build_core_pattern_recognizers(self) -> list[Any]:
        """Build core pattern-based recognizers."""
        from presidio_analyzer import Pattern, PatternRecognizer

        recognizers = []

        # US SSN
        ssn_patterns = [
            Pattern(name="ssn_dashes", regex=r"\b\d{3}-\d{2}-\d{4}\b", score=0.85),
            Pattern(name="ssn_spaces", regex=r"\b\d{3}\s\d{2}\s\d{4}\b", score=0.85),
            Pattern(name="ssn_no_sep", regex=r"\b\d{9}\b", score=0.3),
        ]
        recognizers.append(
            PatternRecognizer(
                supported_entity="US_SSN",
                patterns=ssn_patterns,
                context=["ssn", "social security", "social security number"],
            )
        )

        # Email
        email_patterns = [
            Pattern(
                name="email",
                regex=r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",
                score=0.85,
            ),
        ]
        recognizers.append(
            PatternRecognizer(supported_entity="EMAIL_ADDRESS", patterns=email_patterns)
        )

        # Phone Number
        phone_patterns = [
            Pattern(
                name="phone_with_parens",
                regex=r"\b\(?\d{3}\)?[\s.-]?\d{3}[\s.-]?\d{4}\b",
                score=0.7,
            ),
            Pattern(
                name="phone_with_country",
                regex=r"\b\+?1?[\s.-]?\(?\d{3}\)?[\s.-]?\d{3}[\s.-]?\d{4}\b",
                score=0.7,
            ),
        ]
        recognizers.append(
            PatternRecognizer(
                supported_entity="PHONE_NUMBER",
                patterns=phone_patterns,
                context=["phone", "telephone", "cell", "mobile", "call", "contact"],
            )
        )

        # Credit Card
        cc_patterns = [
            Pattern(
                name="credit_card",
                regex=r"\b(?:4[0-9]{12}(?:[0-9]{3})?|5[1-5][0-9]{14}|3[47][0-9]{13}|6(?:011|5[0-9]{2})[0-9]{12})\b",
                score=0.8,
            ),
            Pattern(
                name="credit_card_spaced",
                regex=r"\b(?:\d{4}[\s-]?){3}\d{4}\b",
                score=0.5,
            ),
        ]
        recognizers.append(
            PatternRecognizer(
                supported_entity="CREDIT_CARD",
                patterns=cc_patterns,
                context=["credit card", "card number", "cc", "visa", "mastercard", "amex"],
            )
        )

        # IP Address
        ip_patterns = [
            Pattern(
                name="ipv4",
                regex=r"\b(?:(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\b",
                score=0.7,
            ),
        ]
        recognizers.append(
            PatternRecognizer(
                supported_entity="IP_ADDRESS",
                patterns=ip_patterns,
                context=["ip", "ip address", "address"],
            )
        )

        # URL
        url_patterns = [
            Pattern(
                name="url",
                regex=r"\bhttps?://[^\s<>\"{}|\\^`\[\]]+\b",
                score=0.7,
            ),
        ]
        recognizers.append(PatternRecognizer(supported_entity="URL", patterns=url_patterns))

        # US Driver License
        dl_patterns = [
            Pattern(
                name="driver_license_with_context",
                regex=r"\b(?:DL|Driver'?s?\s*License|License)[\s:#]*([A-Z0-9]{5,15})\b",
                score=0.7,
            ),
        ]
        recognizers.append(
            PatternRecognizer(
                supported_entity="US_DRIVER_LICENSE",
                patterns=dl_patterns,
                context=["driver", "license", "dl", "driving"],
            )
        )

        # Date of Birth / Dates
        date_patterns = [
            Pattern(
                name="date_mdy_full",
                regex=r"\b(?:0?[1-9]|1[0-2])[/\-.](?:0?[1-9]|[12]\d|3[01])[/\-.](?:19|20)\d{2}\b",
                score=0.7,
            ),
            Pattern(
                name="date_iso",
                regex=r"\b(?:19|20)\d{2}[/\-.](?:0?[1-9]|1[0-2])[/\-.](?:0?[1-9]|[12]\d|3[01])\b",
                score=0.7,
            ),
            Pattern(
                name="date_written",
                regex=r"\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s+(?:19|20)\d{2}\b",
                score=0.75,
            ),
            Pattern(
                name="date_written_dmy",
                regex=r"\b\d{1,2}\s+(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+(?:19|20)\d{2}\b",
                score=0.75,
            ),
        ]
        recognizers.append(
            PatternRecognizer(
                supported_entity="DATE_TIME",
                patterns=date_patterns,
                context=[
                    "dob", "birth", "born", "date of birth",
                    "birthday", "admitted", "discharged", "died",
                ],
            )
        )

        # Person names with context
        name_patterns = [
            Pattern(
                name="patient_name",
                regex=r"(?:Patient|Client|Member|Subscriber|Beneficiary)(?:\s+Name)?[\s:]+([A-Z][a-z]+\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)",
                score=0.85,
            ),
            Pattern(
                name="name_field",
                regex=r"(?<!\w)Name[\s:]+([A-Z][a-z]+\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)",
                score=0.7,
            ),
            Pattern(
                name="doctor_name",
                regex=r"(?:Dr\.?|Doctor|Physician|Provider)[\s:]+([A-Z][a-z]+\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)",
                score=0.8,
            ),
        ]
        recognizers.append(
            PatternRecognizer(
                supported_entity="PERSON",
                patterns=name_patterns,
                context=["patient", "name", "client", "member", "doctor", "physician"],
            )
        )

        return recognizers

    def _build_healthcare_recognizers(self) -> list[Any]:
        """Build healthcare-specific recognizers."""
        from presidio_analyzer import Pattern, PatternRecognizer

        recognizers = []

        # Medical Record Number (MRN)
        mrn_patterns = [
            Pattern(
                name="mrn_numeric",
                regex=r"\b(?:MRN|MR#?|Medical Record|Patient ID)[\s:#\-]*(\d{6,10})\b",
                score=0.85,
            ),
            Pattern(
                name="mrn_alphanumeric",
                regex=r"\b(?:MRN|MR#?)[\s:#\-]*([A-Z]{1,3}[\-]?\d{6,10})\b",
                score=0.85,
            ),
            Pattern(
                name="mrn_standalone",
                regex=r"\b[A-Z]{2,3}\d{7,10}\b",
                score=0.4,
            ),
        ]
        recognizers.append(
            PatternRecognizer(
                supported_entity="MEDICAL_RECORD_NUMBER",
                patterns=mrn_patterns,
                context=["mrn", "medical record", "patient id", "chart number", "hospital number"],
            )
        )

        # Health Plan Beneficiary Numbers
        health_plan_patterns = [
            Pattern(
                name="medicare_new",
                regex=r"\b[1-9][A-Z][A-Z0-9]\d-?[A-Z][A-Z0-9]\d-?[A-Z][A-Z0-9]\d{2}\b",
                score=0.85,
            ),
            Pattern(
                name="medicare_legacy",
                regex=r"\b\d{3}-?\d{2}-?\d{4}[A-Z]{1,2}\b",
                score=0.75,
            ),
            Pattern(
                name="member_id_generic",
                regex=r"\b(?:Member ID|Policy|Subscriber|Beneficiary)[\s:#]*([A-Z0-9]{9,15})\b",
                score=0.8,
            ),
            Pattern(
                name="group_number",
                regex=r"\b(?:Group|GRP)[\s:#]*([A-Z0-9]{5,12})\b",
                score=0.7,
            ),
        ]
        recognizers.append(
            PatternRecognizer(
                supported_entity="HEALTH_PLAN_BENEFICIARY",
                patterns=health_plan_patterns,
                context=[
                    "medicare", "medicaid", "member", "subscriber",
                    "beneficiary", "insurance", "policy", "group",
                    "health plan", "coverage",
                ],
            )
        )

        # NPI
        npi_patterns = [
            Pattern(
                name="npi_with_context",
                regex=r"\b(?:NPI|National Provider)[\s:#]*(\d{10})\b",
                score=0.9,
            ),
            Pattern(
                name="npi_standalone",
                regex=r"\b[12]\d{9}\b",
                score=0.5,
            ),
        ]
        recognizers.append(
            PatternRecognizer(
                supported_entity="NPI",
                patterns=npi_patterns,
                context=["npi", "national provider", "provider identifier", "prescriber"],
            )
        )

        # DEA Number
        dea_patterns = [
            Pattern(
                name="dea_with_context",
                regex=r"\b(?:DEA|Drug Enforcement)[\s:#]*([ABCDEFGHJKLMPRSTUX][A-Z9]\d{7})\b",
                score=0.9,
            ),
            Pattern(
                name="dea_standalone",
                regex=r"\b[ABCDEFGHJKLMPRSTUX][A-Z9]\d{7}\b",
                score=0.6,
            ),
        ]
        recognizers.append(
            PatternRecognizer(
                supported_entity="DEA_NUMBER",
                patterns=dea_patterns,
                context=["dea", "drug enforcement", "controlled substance", "prescriber"],
            )
        )

        # Medical License
        medical_license_patterns = [
            Pattern(
                name="medical_license_with_context",
                regex=r"\b(?:License|Lic|Medical License)[\s:#]*([A-Z]{1,2}\d{5,8})\b",
                score=0.8,
            ),
            Pattern(
                name="state_license",
                regex=r"\b(?:MD|DO|RN|NP|PA|DDS|DMD|DPM|DC|OD)[\s-]*(?:License|Lic)[\s:#]*(\d{4,8})\b",
                score=0.85,
            ),
        ]
        recognizers.append(
            PatternRecognizer(
                supported_entity="MEDICAL_LICENSE",
                patterns=medical_license_patterns,
                context=[
                    "license", "medical license", "state license",
                    "board certified", "credentials", "physician",
                    "practitioner",
                ],
            )
        )

        return recognizers

    def _build_geographic_recognizers(self) -> list[Any]:
        """Build geographic identifier recognizers."""
        from presidio_analyzer import Pattern, PatternRecognizer

        recognizers = []

        # US ZIP Code
        zip_patterns = [
            Pattern(name="zip_plus_4", regex=r"\b\d{5}-\d{4}\b", score=0.7),
            Pattern(
                name="zip_5_with_context",
                regex=r"\b(?:zip|zip code|postal)[\s:#]*(\d{5})\b",
                score=0.7,
            ),
        ]
        recognizers.append(
            PatternRecognizer(
                supported_entity="US_ZIP_CODE",
                patterns=zip_patterns,
                context=["zip", "zipcode", "zip code", "postal", "mailing"],
            )
        )

        # Street Address
        address_patterns = [
            Pattern(
                name="street_address_full",
                regex=r"\b\d{1,5}\s+(?:[A-Z][a-z]+\s+){1,3}(?:Street|St|Avenue|Ave|Road|Rd|Boulevard|Blvd|Drive|Dr|Lane|Ln|Way|Court|Ct|Circle|Cir|Place|Pl|Terrace|Ter|Highway|Hwy)\.?\b",
                score=0.7,
            ),
            Pattern(
                name="po_box",
                regex=r"\b(?:P\.?O\.?\s*Box|Post Office Box)\s*\d+\b",
                score=0.85,
            ),
            Pattern(
                name="apt_suite",
                regex=r"\b(?:Apt|Apartment|Suite|Ste|Unit|#)\s*[A-Z0-9]+\b",
                score=0.5,
            ),
        ]
        recognizers.append(
            PatternRecognizer(
                supported_entity="STREET_ADDRESS",
                patterns=address_patterns,
                context=["address", "street", "mail", "ship", "deliver", "residence", "home"],
            )
        )

        return recognizers

    def _build_identifier_recognizers(self) -> list[Any]:
        """Build vehicle, device, and other identifier recognizers."""
        from presidio_analyzer import Pattern, PatternRecognizer

        recognizers = []

        # VIN
        vin_patterns = [
            Pattern(
                name="vin_with_context",
                regex=r"\b(?:VIN|Vehicle ID)[\s:#]*([A-HJ-NPR-Z0-9]{17})\b",
                score=0.9,
            ),
            Pattern(
                name="vin_standalone",
                regex=r"\b[A-HJ-NPR-Z0-9]{17}\b",
                score=0.5,
            ),
        ]
        recognizers.append(
            PatternRecognizer(
                supported_entity="VIN",
                patterns=vin_patterns,
                context=["vin", "vehicle", "car", "truck", "automobile", "registration"],
            )
        )

        # License Plate
        plate_patterns = [
            Pattern(
                name="plate_with_context",
                regex=r"\b(?:License Plate|Plate|Tag|Plate #)[\s:#]*([A-Z0-9]{2,8})\b",
                score=0.85,
            ),
        ]
        recognizers.append(
            PatternRecognizer(
                supported_entity="LICENSE_PLATE",
                patterns=plate_patterns,
                context=["license plate", "plate number", "tag", "vehicle registration", "dmv"],
            )
        )

        # Device Serial Numbers
        serial_patterns = [
            Pattern(
                name="serial_with_context",
                regex=r"\b(?:Serial|SN|S/N)[\s:#]*([A-Z0-9]{8,20})\b",
                score=0.85,
            ),
            Pattern(
                name="serial_common",
                regex=r"\b[A-Z]{2,4}\d{6,12}[A-Z0-9]*\b",
                score=0.4,
            ),
        ]
        recognizers.append(
            PatternRecognizer(
                supported_entity="DEVICE_SERIAL",
                patterns=serial_patterns,
                context=["serial", "device", "equipment", "model", "asset"],
            )
        )

        # UDI
        udi_patterns = [
            Pattern(
                name="udi_gs1",
                regex=r"\b\(01\)\d{14}(?:\(\d{2}\)[A-Z0-9]+)*\b",
                score=0.9,
            ),
            Pattern(
                name="udi_hibcc",
                regex=r"\b\+[A-Z0-9]{4,}\/[A-Z0-9]+\b",
                score=0.85,
            ),
            Pattern(
                name="udi_with_context",
                regex=r"\b(?:UDI|Unique Device)[\s:#]*([A-Z0-9\(\)\/\+]{10,})\b",
                score=0.9,
            ),
        ]
        recognizers.append(
            PatternRecognizer(
                supported_entity="UDI",
                patterns=udi_patterns,
                context=["udi", "unique device", "medical device", "implant", "fda"],
            )
        )

        # IMEI
        imei_patterns = [
            Pattern(
                name="imei_with_context",
                regex=r"\b(?:IMEI|International Mobile)[\s:#]*(\d{15})\b",
                score=0.9,
            ),
            Pattern(
                name="imei_standalone",
                regex=r"\b\d{2}-?\d{6}-?\d{6}-?\d\b",
                score=0.6,
            ),
        ]
        recognizers.append(
            PatternRecognizer(
                supported_entity="IMEI",
                patterns=imei_patterns,
                context=["imei", "mobile", "phone", "device", "cellular"],
            )
        )

        # Fax Number
        fax_patterns = [
            Pattern(
                name="fax_with_context",
                regex=r"\b(?:Fax|Facsimile|F)[\s:#]*(?:\+?1[\s.-]?)?\(?\d{3}\)?[\s.-]?\d{3}[\s.-]?\d{4}\b",
                score=0.9,
            ),
        ]
        recognizers.append(
            PatternRecognizer(
                supported_entity="FAX_NUMBER",
                patterns=fax_patterns,
                context=["fax", "facsimile"],
            )
        )

        # Biometric Identifiers
        biometric_patterns = [
            Pattern(
                name="biometric_reference",
                regex=r"\b(?:fingerprint|retina|iris|voice\s*print|face\s*id|biometric)[\s:#]*(?:id|scan|data|template)[\s:#]*([A-Z0-9\-]{8,})\b",
                score=0.85,
            ),
        ]
        recognizers.append(
            PatternRecognizer(
                supported_entity="BIOMETRIC_ID",
                patterns=biometric_patterns,
                context=["biometric", "fingerprint", "retina", "iris", "voiceprint", "facial"],
            )
        )

        # UUID
        uuid_patterns = [
            Pattern(
                name="uuid",
                regex=r"\b[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\b",
                score=0.8,
            ),
        ]
        recognizers.append(
            PatternRecognizer(supported_entity="UUID", patterns=uuid_patterns)
        )

        return recognizers

    @property
    def mode(self) -> str:
        """Current operating mode."""
        return self._mode

    @property
    def is_spacy_available(self) -> bool:
        """Whether spaCy NLP is available."""
        self._ensure_initialized()
        return self._spacy_available
