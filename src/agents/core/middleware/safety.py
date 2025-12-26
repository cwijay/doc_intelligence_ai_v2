"""
Safety middleware for DocumentAgent.

Provides:
- PIIDetector: Detects and handles PII in text using regex patterns
"""

import hashlib
import logging
import re
from enum import Enum
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class PIIStrategy(str, Enum):
    """Strategy for handling detected PII."""
    REDACT = "redact"      # Replace with [REDACTED]
    MASK = "mask"          # Partial masking (e.g., ***@email.com)
    HASH = "hash"          # Replace with hash
    BLOCK = "block"        # Raise exception if PII found


class PIIType(str, Enum):
    """Types of PII that can be detected."""
    EMAIL = "email"
    PHONE = "phone"
    SSN = "ssn"
    CREDIT_CARD = "credit_card"
    IP_ADDRESS = "ip_address"
    DATE_OF_BIRTH = "date_of_birth"
    ADDRESS = "address"


class PIIMatch:
    """Represents a detected PII match."""

    def __init__(
        self,
        pii_type: PIIType,
        value: str,
        start: int,
        end: int
    ):
        self.pii_type = pii_type
        self.value = value
        self.start = start
        self.end = end

    def __repr__(self):
        return f"PIIMatch({self.pii_type.value}: {self.start}-{self.end})"


class PIIBlockedError(Exception):
    """Raised when PII is detected and strategy is BLOCK."""

    def __init__(self, pii_types: List[PIIType]):
        self.pii_types = pii_types
        types_str = ", ".join(t.value for t in pii_types)
        super().__init__(f"PII detected and blocked: {types_str}")


class PIIDetector:
    """
    Detects and handles PII in text using regex patterns.

    Supports multiple handling strategies: redact, mask, hash, or block.
    """

    # Regex patterns for common PII types
    PATTERNS: Dict[PIIType, str] = {
        PIIType.EMAIL: r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
        PIIType.PHONE: r'\b(?:\+?1[-.\s]?)?(?:\(?\d{3}\)?[-.\s]?)?\d{3}[-.\s]?\d{4}\b',
        PIIType.SSN: r'\b\d{3}[-\s]?\d{2}[-\s]?\d{4}\b',
        PIIType.CREDIT_CARD: r'\b(?:\d{4}[-\s]?){3}\d{4}\b',
        PIIType.IP_ADDRESS: r'\b(?:\d{1,3}\.){3}\d{1,3}\b',
        PIIType.DATE_OF_BIRTH: r'\b(?:0[1-9]|1[0-2])[-/](?:0[1-9]|[12]\d|3[01])[-/](?:19|20)\d{2}\b',
    }

    def __init__(
        self,
        strategy: PIIStrategy = PIIStrategy.REDACT,
        detect_types: Optional[List[PIIType]] = None,
        custom_patterns: Optional[Dict[str, str]] = None
    ):
        """
        Initialize PII detector.

        Args:
            strategy: How to handle detected PII
            detect_types: Which PII types to detect (None = all)
            custom_patterns: Additional custom patterns to detect
        """
        self.strategy = strategy
        self.detect_types = detect_types or list(PIIType)
        self.custom_patterns = custom_patterns or {}

        # Compile patterns for efficiency
        self._compiled_patterns: Dict[str, re.Pattern] = {}
        for pii_type in self.detect_types:
            if pii_type in self.PATTERNS:
                self._compiled_patterns[pii_type.value] = re.compile(
                    self.PATTERNS[pii_type],
                    re.IGNORECASE
                )

        for name, pattern in self.custom_patterns.items():
            self._compiled_patterns[name] = re.compile(pattern, re.IGNORECASE)

        logger.debug(
            f"PIIDetector initialized with strategy={strategy.value}, "
            f"types={[t.value for t in self.detect_types]}"
        )

    def detect(self, text: str) -> List[PIIMatch]:
        """
        Detect PII in text.

        Args:
            text: Text to scan for PII

        Returns:
            List of PIIMatch objects
        """
        matches = []

        for type_name, pattern in self._compiled_patterns.items():
            for match in pattern.finditer(text):
                try:
                    pii_type = PIIType(type_name)
                except ValueError:
                    # Custom pattern, use EMAIL as placeholder
                    pii_type = PIIType.EMAIL

                matches.append(PIIMatch(
                    pii_type=pii_type,
                    value=match.group(),
                    start=match.start(),
                    end=match.end()
                ))

        # Sort by position
        matches.sort(key=lambda m: m.start)

        if matches:
            logger.info(f"Detected {len(matches)} PII matches in text")

        return matches

    def process(self, text: str) -> Tuple[str, List[PIIMatch]]:
        """
        Process text according to the configured strategy.

        Args:
            text: Text to process

        Returns:
            Tuple of (processed_text, detected_matches)

        Raises:
            PIIBlockedError: If strategy is BLOCK and PII is detected
        """
        matches = self.detect(text)

        if not matches:
            return text, []

        if self.strategy == PIIStrategy.BLOCK:
            pii_types = list(set(m.pii_type for m in matches))
            raise PIIBlockedError(pii_types)

        # Process matches in reverse order to preserve positions
        processed_text = text
        for match in reversed(matches):
            replacement = self._get_replacement(match)
            processed_text = (
                processed_text[:match.start] +
                replacement +
                processed_text[match.end:]
            )

        return processed_text, matches

    def _get_replacement(self, match: PIIMatch) -> str:
        """
        Get replacement text for a PII match based on strategy.

        Args:
            match: PIIMatch object

        Returns:
            Replacement string
        """
        if self.strategy == PIIStrategy.REDACT:
            return f"[{match.pii_type.value.upper()}_REDACTED]"

        elif self.strategy == PIIStrategy.MASK:
            return self._mask_value(match)

        elif self.strategy == PIIStrategy.HASH:
            hash_value = hashlib.sha256(
                match.value.encode()
            ).hexdigest()[:12]
            return f"[HASH:{hash_value}]"

        return match.value

    def _mask_value(self, match: PIIMatch) -> str:
        """
        Mask a PII value partially.

        Args:
            match: PIIMatch object

        Returns:
            Masked string
        """
        value = match.value

        if match.pii_type == PIIType.EMAIL:
            # Show first char and domain: j***@email.com
            parts = value.split('@')
            if len(parts) == 2:
                return f"{parts[0][0]}***@{parts[1]}"

        elif match.pii_type == PIIType.PHONE:
            # Show last 4 digits: ***-***-1234
            digits = re.sub(r'\D', '', value)
            if len(digits) >= 4:
                return f"***-***-{digits[-4:]}"

        elif match.pii_type == PIIType.SSN:
            # Show last 4 digits: ***-**-1234
            digits = re.sub(r'\D', '', value)
            if len(digits) == 9:
                return f"***-**-{digits[-4:]}"

        elif match.pii_type == PIIType.CREDIT_CARD:
            # Show last 4 digits: ****-****-****-1234
            digits = re.sub(r'\D', '', value)
            if len(digits) >= 4:
                return f"****-****-****-{digits[-4:]}"

        elif match.pii_type == PIIType.IP_ADDRESS:
            # Mask first two octets: ***.***123.456
            parts = value.split('.')
            if len(parts) == 4:
                return f"***.***{parts[2]}.{parts[3]}"

        # Default masking: show first and last char
        if len(value) > 2:
            return f"{value[0]}{'*' * (len(value) - 2)}{value[-1]}"

        return '*' * len(value)

    def has_pii(self, text: str) -> bool:
        """
        Check if text contains any PII.

        Args:
            text: Text to check

        Returns:
            True if PII detected, False otherwise
        """
        return len(self.detect(text)) > 0

    def get_pii_summary(self, text: str) -> Dict[str, int]:
        """
        Get summary of PII types found in text.

        Args:
            text: Text to analyze

        Returns:
            Dictionary mapping PII type to count
        """
        matches = self.detect(text)
        summary: Dict[str, int] = {}

        for match in matches:
            type_name = match.pii_type.value
            summary[type_name] = summary.get(type_name, 0) + 1

        return summary
