from __future__ import annotations

import re
from typing import Any, Dict, List


# ============================================================
# Controlled technical paraphrases
# ============================================================

CLAIM_SYNONYMS = [
    (
        "supports the termination of",
        "terminates",
    ),
    (
        "supports termination of",
        "terminates",
    ),
    (
        "provides support for",
        "supports",
    ),
    (
        "is responsible for",
        "supports",
    ),
    (
        "is responsible to",
        "supports",
    ),
    (
        "handles",
        "supports",
    ),
    (
        "provides",
        "supports",
    ),
]


# ============================================================
# Technical terms that should not disappear during
# paraphrasing
# ============================================================

IMPORTANT_TECHNICAL_TERMS = {
    "amf",
    "smf",
    "upf",
    "pcf",
    "nrf",
    "udm",
    "udr",
    "ausf",
    "nef",
    "nssf",
    "nwdaf",
    "chf",
    "nas",
    "ran",
    "ue",
    "scp",
    "sepp",
    "n1",
    "n2",
    "n3",
    "n4",
    "n5",
    "n6",
    "n7",
    "n8",
    "n10",
    "n11",
    "n12",
    "n13",
    "n14",
    "n15",
    "n16",
    "n22",
    "n24",
    "n26",
    "n27",
    "n30",
    "n31",
    "n32",
    "n33",
}


# ============================================================
# Basic text normalization
# ============================================================

def normalize_claim_text(
    text: str,
) -> str:
    """
    Normalize wording differences while preserving
    important technical meaning.

    Example:

        "supports termination of the RAN CP interface (N2)"
        ->
        "terminates the ran cp interface n2"
    """

    if not text:
        return ""

    text = text.lower()

    # Normalize common punctuation.
    text = re.sub(
        r"[^a-z0-9\s/-]",
        " ",
        text,
    )

    # Normalize whitespace.
    text = " ".join(
        text.split()
    )

    # Controlled technical paraphrases.
    for source, target in CLAIM_SYNONYMS:
        text = text.replace(
            source,
            target,
        )

    # Normalize whitespace again after replacements.
    text = " ".join(
        text.split()
    )

    return text


# ============================================================
# Tokenization
# ============================================================

def tokenize(
    text: str,
) -> set[str]:
    """
    Tokenize normalized technical text.
    """

    normalized = normalize_claim_text(
        text
    )

    return set(
        re.findall(
            r"[a-z0-9]+(?:-[a-z0-9]+)*",
            normalized,
        )
    )


# ============================================================
# Important technical tokens
# ============================================================

def technical_tokens(
    text: str,
) -> set[str]:
    """
    Extract important 3GPP technical identifiers.

    These provide a safety check so that semantic
    paraphrasing does not accidentally change the
    technical subject.
    """

    tokens = tokenize(text)

    return tokens.intersection(
        IMPORTANT_TECHNICAL_TERMS
    )


# ============================================================
# Section matching
# ============================================================

def section_matches(
    claim_section: str,
    evidence_section: str,
) -> bool:
    """
    Match exact or child sections.

    Examples:

        6.2.1 == 6.2.1
        6.2.1 <- 6.2.1.1
        4.2.8.3 <- 4.2.8.3.1
    """

    claim_section = str(
        claim_section
    ).strip()

    evidence_section = str(
        evidence_section
    ).strip()

    if not claim_section or not evidence_section:
        return False

    if claim_section == evidence_section:
        return True

    if evidence_section.startswith(
        claim_section + "."
    ):
        return True

    return False


# ============================================================
# Keyword overlap
# ============================================================

def keyword_overlap(
    claim: str,
    evidence: str,
) -> float:
    """
    Calculate token overlap between claim and evidence.
    """

    claim_tokens = tokenize(
        claim
    )

    evidence_tokens = tokenize(
        evidence
    )

    if not claim_tokens:
        return 0.0

    overlap = (
        claim_tokens
        .intersection(
            evidence_tokens
        )
    )

    return (
        len(overlap)
        / len(claim_tokens)
    )


# ============================================================
# Technical-token consistency
# ============================================================

def technical_token_match(
    claim: str,
    evidence: str,
) -> bool:
    """
    Ensure critical technical identifiers in the claim
    are also present in the supporting evidence.

    If the claim contains N2 / AMF / NAS / RAN, the
    evidence should contain those same identifiers.
    """

    claim_terms = technical_tokens(
        claim
    )

    evidence_terms = technical_tokens(
        evidence
    )

    # No important technical token in the claim.
    if not claim_terms:
        return True

    return claim_terms.issubset(
        evidence_terms
    )


# ============================================================
# Claim support evaluation
# ============================================================

def validate_single_claim(
    claim: Dict[str, Any],
    evidence: List[Dict[str, Any]],
    min_overlap: float = 0.45,
) -> Dict[str, Any]:
    """
    Validate one generated claim against retrieved evidence.

    Validation requires:

        1. claim text exists
        2. claim section exists
        3. matching evidence section
        4. sufficient lexical overlap
        5. important technical identifiers preserved
    """

    claim_text = str(
        claim.get(
            "text",
            "",
        )
    ).strip()

    claim_section = str(
        claim.get(
            "section",
            "",
        )
    ).strip()

    if not claim_text:

        return {
            "claim": claim_text,
            "section": claim_section,
            "valid": False,
            "reason": "empty_claim",
        }

    if not claim_section:

        return {
            "claim": claim_text,
            "section": claim_section,
            "valid": False,
            "reason": "missing_section",
        }

    matching_evidence = []

    for result in evidence:

        evidence_section = str(
            result.get(
                "section",
                "",
            )
        ).strip()

        if not section_matches(
            claim_section,
            evidence_section,
        ):
            continue

        evidence_text = str(
            result.get(
                "content",
                "",
            )
        ).strip()

        if not evidence_text:
            continue

        matching_evidence.append(
            result
        )

    if not matching_evidence:

        return {
            "claim": claim_text,
            "section": claim_section,
            "valid": False,
            "reason": "no_matching_section",
        }

    best_overlap = 0.0
    best_result = None
    technical_match = False

    # --------------------------------------------------------
    # Evaluate every matching evidence chunk.
    # --------------------------------------------------------

    for result in matching_evidence:

        evidence_text = str(
            result.get(
                "content",
                "",
            )
        ).strip()

        overlap = keyword_overlap(
            claim_text,
            evidence_text,
        )

        if overlap > best_overlap:

            best_overlap = overlap
            best_result = result

        if technical_token_match(
            claim_text,
            evidence_text,
        ):
            technical_match = True

    # --------------------------------------------------------
    # Strong support
    # --------------------------------------------------------

    if (
        best_overlap >= min_overlap
        and technical_match
    ):

        return {
            "claim": claim_text,
            "section": claim_section,
            "valid": True,
            "reason": "supported",
        }

    # --------------------------------------------------------
    # Technical terms match but lexical overlap is lower.
    #
    # Allow a lower threshold because controlled
    # paraphrasing can legitimately change wording.
    # --------------------------------------------------------

    if (
        technical_match
        and best_overlap >= 0.30
    ):

        return {
            "claim": claim_text,
            "section": claim_section,
            "valid": True,
            "reason": "supported_paraphrase",
        }

    # --------------------------------------------------------
    # Weak evidence.
    # --------------------------------------------------------

    return {
        "claim": claim_text,
        "section": claim_section,
        "valid": False,
        "reason": "weak_evidence_overlap",
    }


# ============================================================
# Main validation function
# ============================================================

def validate_claims(
    response: Dict[str, Any] | None = None,
    evidence: List[Dict[str, Any]] | None = None,
    answer: str | None = None,
) -> Dict[str, Any]:
    """
    Validate all claims generated by the LLM.

    Preferred usage:

        validate_claims(
            response=response,
            evidence=results,
        )

    Backward-compatible usage:

        validate_claims(
            answer=answer,
            evidence=results,
        )

    When only plain answer text is provided, validation
    cannot reliably associate every sentence with a source
    section, so structured claims are preferred.
    """

    evidence = evidence or []

    # --------------------------------------------------------
    # Structured response path
    # --------------------------------------------------------

    if response is not None:

        claims = response.get(
            "claims",
            [],
        )

    # --------------------------------------------------------
    # Backward-compatible plain-answer path
    # --------------------------------------------------------

    elif answer is not None:

        # Plain answers do not carry explicit section metadata.
        # Treat them as unsupported rather than inventing a
        # section.
        return {
            "valid": False,
            "reason": "structured_claims_required",
            "total_claims": 0,
            "valid_claims": 0,
            "invalid_claims": [],
            "claims": [],
        }

    else:

        return {
            "valid": False,
            "reason": "missing_response",
            "total_claims": 0,
            "valid_claims": 0,
            "invalid_claims": [],
            "claims": [],
        }

    # --------------------------------------------------------
    # Validate claims container
    # --------------------------------------------------------

    if not isinstance(
        claims,
        list,
    ):

        return {
            "valid": False,
            "reason": "claims_not_a_list",
            "total_claims": 0,
            "valid_claims": 0,
            "invalid_claims": [],
            "claims": [],
        }

    # No claims is not acceptable for an allowed answer.
    if not claims:

        return {
            "valid": False,
            "reason": "no_claims",
            "total_claims": 0,
            "valid_claims": 0,
            "invalid_claims": [],
            "claims": [],
        }

    validated_claims = []

    invalid_claims = []

    valid_count = 0

    # --------------------------------------------------------
    # Validate each claim.
    # --------------------------------------------------------

    for claim in claims:

        if not isinstance(
            claim,
            dict,
        ):

            validation = {
                "claim": str(
                    claim
                ),
                "section": "",
                "valid": False,
                "reason": "claim_not_an_object",
            }

        else:

            validation = validate_single_claim(
                claim=claim,
                evidence=evidence,
            )

        validated_claims.append(
            validation
        )

        if validation["valid"]:

            valid_count += 1

        else:

            invalid_claims.append(
                validation
            )

    # --------------------------------------------------------
    # Entire answer is valid only when every claim passes.
    # --------------------------------------------------------

    all_valid = (
        len(invalidated_claims)
        == 0
        if False
        else len(invalid_claims) == 0
    )

    if all_valid:

        reason = "all_claims_supported"

    else:

        reason = (
            "weak_evidence_overlap"
            if any(
                claim.get(
                    "reason"
                ) == "weak_evidence_overlap"
                for claim in invalid_claims
            )
            else "claim_validation_failed"
        )

    return {
        "valid": all_valid,
        "reason": reason,
        "total_claims": len(
            claims
        ),
        "valid_claims": valid_count,
        "invalid_claims": invalid_claims,
        "claims": validated_claims,
    }