from typing import List, Dict, Tuple
import logging

logger = logging.getLogger(__name__)


def flatten_pipeline_output(pipeline_output: dict) -> Tuple[str, List[Dict], str]:
    """
    Flatten pipeline JSON into judge-ready format.

    Args:
        pipeline_output: Raw pipeline JSON with normalized_terms

    Returns:
        Tuple of (clinical_note, flattened_terms_list, formatted_terms_string)
    """
    logger.info(f"Flattening pipeline output for note_id: {pipeline_output.get('note_id', 'unknown')}")

    # Extract clinical note
    clinical_note = pipeline_output.get("api_response", {}).get("raw_text", "")

    # Extract normalized terms
    normalized_terms = pipeline_output.get("api_response", {}).get("normalized_terms", [])

    if not normalized_terms:
        logger.warning("No normalized_terms found in pipeline output")
        return clinical_note, [], ""

    # Flatten each term
    flattened_terms = []
    for term_data in normalized_terms:
        term_text = term_data.get("term", "")
        normalize_payload = term_data.get("normalize_payload", {})

        # Extract default lexical info
        default_lexical_title = normalize_payload.get("default_lexical_title", "")
        default_lexical_code = normalize_payload.get("default_lexical_code", "")

        # Extract concept code and title (for judge validation)
        concept_code = normalize_payload.get("concept_code", "")
        concept_title = normalize_payload.get("title", "")

        # Extract IMO code
        imo_code = normalize_payload.get("code", "")

        # Extract ICD-10 codes
        icd10_codes = []
        mappings = normalize_payload.get("metadata", {}).get("mappings", {})
        icd10_data = mappings.get("icd10cm", {}).get("codes", [])
        for code_item in icd10_data:
            icd10_codes.append({
                "code": code_item.get("code", ""),
                "title": code_item.get("title", "")
            })

        # Extract SNOMED codes
        snomed_codes = []
        snomed_data = mappings.get("snomedInternational", {}).get("codes", [])
        for code_item in snomed_data:
            snomed_codes.append({
                "code": code_item.get("code", ""),
                "title": code_item.get("title", "")
            })

        flattened_term = {
            "term": term_text,
            "concept_code": concept_code,
            "concept_title": concept_title,
            "default_lexical_title": default_lexical_title,
            "default_lexical_code": default_lexical_code,
            "imo_code": imo_code,
            "icd10": icd10_codes,
            "snomed": snomed_codes
        }

        flattened_terms.append(flattened_term)

    logger.info(f"Flattened {len(flattened_terms)} terms")

    # Format terms for prompt injection
    formatted_terms = format_terms_for_prompt(flattened_terms)

    return clinical_note, flattened_terms, formatted_terms


def format_terms_for_prompt(flattened_terms: List[Dict]) -> str:
    """
    Format flattened terms for prompt injection.

    Args:
        flattened_terms: List of flattened term dictionaries

    Returns:
        Formatted string for prompt injection
    """
    formatted_parts = []

    for idx, term in enumerate(flattened_terms, 1):
        term_text = term.get("default_lexical_title", term.get("term", ""))
        icd10_codes = term.get("icd10", [])
        snomed_codes = term.get("snomed", [])

        # Build term section
        section = f"TERM {idx}:\n"
        section += f"- Term: {term_text}\n"

        # Add ICD-10 codes
        for code in icd10_codes:
            section += f"  ICD-10: {code['code']} - {code['title']}\n"

        # Add SNOMED codes
        for code in snomed_codes:
            section += f"  SNOMED: {code['code']} - {code['title']}\n"

        formatted_parts.append(section)

    formatted_string = "\n".join(formatted_parts)

    logger.debug(f"Formatted terms string length: {len(formatted_string)}")

    return formatted_string
