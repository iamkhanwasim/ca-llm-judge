# LLM Judge — FastAPI Application

## Claude Code Prompt File

---

## 1. Project Overview

Build a FastAPI application that serves as a reference-free LLM judge for evaluating clinical NLP pipeline output. The pipeline takes clinical notes and produces IMO lexical terms with mapped ICD-10 and SNOMED codes. The judge evaluates whether the pipeline output is clinically correct by reading only the clinical note and the predicted terms — no gold standard is used at runtime (except for the gold_evaluation endpoint).

The primary use case is diabetes-related clinical notes, but the system should support other conditions as well.

**Evaluation level:** The judge receives **all `normalized_terms` in a single prompt** per judge and scores **each term independently** within that response. All 4 scoring dimensions are applied per term. This means one LLM call per judge per note (not per term). Note-level verdicts are derived by aggregating term-level scores.

The judge evaluates per term:
- Default lexical term text (e.g., "Diabetes mellitus")
- ICD-10 code descriptions (e.g., "Type 2 diabetes mellitus without complications")
- SNOMED code descriptions (e.g., "Diabetes mellitus")

The judge does NOT evaluate IMO codes directly (proprietary system, LLM cannot assess).

**Diabetes term construction rules:**
The pipeline constructs normalized terms following a specific component ordering. The judge should evaluate whether the term follows this structure when components are present (no penalty if the clinical note lacks evidence for a component):

1. **Control Status** — "Controlled" or "Uncontrolled" (only if note has evidence for control status). Precedes diabetes type.
2. **Diabetes Type** — "Type 1" or "Type 2" (only if note specifies type)
3. **Complications** — natural word order: "diabetes mellitus with retinopathy" NOT "retinopathy with diabetes mellitus"
4. **Insulin Status** — "with long-term current use of insulin" (only if note has evidence of insulin use)
5. **Oral Medication** — "with oral medication" (only if note has evidence)

Example well-formed term: `Uncontrolled Type 2 diabetes mellitus with retinopathy with long-term current use of insulin`

These rules are embedded in the judge prompt templates.

---

## 2. Folder Structure

```
llm-judge/
├── app/
│   ├── __init__.py
│   ├── main.py                         # FastAPI app entry point
│   ├── config.py                       # Load and validate config
│   ├── routers/
│   │   ├── __init__.py
│   │   ├── evaluation.py               # /evaluate (single note)
│   │   ├── batch_evaluation.py         # /batch_evaluate (multiple notes)
│   │   ├── gold_evaluation.py          # /gold_evaluate (gold standard P/R/F1)
│   │   ├── health.py                   # /health
│   │   └── models.py                   # /models (list available models)
│   ├── services/
│   │   ├── __init__.py
│   │   ├── input_loader.py             # Parse and flatten pipeline JSON
│   │   ├── judge.py                    # Core LLM judge logic
│   │   ├── judge_registry.py           # Model provider abstraction
│   │   ├── aggregator.py               # Multi-judge score aggregation
│   │   ├── threshold_gate.py           # Configurable threshold logic
│   │   ├── report_generator.py         # Per-note and aggregate reports
│   │   └── gold_evaluator.py           # Gold standard P/R/F1 computation
│   ├── providers/
│   │   ├── __init__.py
│   │   ├── base.py                     # Abstract base provider
│   │   ├── bedrock_provider.py         # AWS Bedrock (Claude, Nova)
│   │   ├── azure_openai_provider.py    # Azure OpenAI (GPT-5)
│   │   └── ollama_provider.py          # Ollama local models
│   ├── schemas/
│   │   ├── __init__.py
│   │   ├── requests.py                 # Request models (Pydantic)
│   │   ├── responses.py                # Response models (Pydantic)
│   │   └── config_schema.py            # Config validation schema
│   └── utils/
│       ├── __init__.py
│       └── helpers.py                  # Shared utilities
├── config/
│   └── config.yaml                     # Single config file
├── prompts/
│   ├── prompt_a_single.txt             # Prompt A: all dimensions at once
│   └── prompt_b_cot.txt                # Prompt B: chain-of-thought
├── data/
│   ├── gold_standard/
│   │   └── gold_standard.json          # 20 annotated gold notes
│   └── pipeline_output/
│       └── pipeline_output.json        # Pipeline output for gold notes
├── tests/
│   ├── __init__.py
│   ├── test_evaluation.py
│   ├── test_batch_evaluation.py
│   ├── test_gold_evaluation.py
│   └── test_providers.py
├── requirements.txt
├── .env                                # Azure OpenAI credentials (not committed)
├── .env.example                        # Template for .env
├── Dockerfile
└── README.md
```

---

## 3. Config File (`config/config.yaml`)

Single config file for all settings:

```yaml
# --- Judge Models ---
judges:
  - provider: bedrock
    model: claude-3.7-sonnet
    enabled: true
  - provider: bedrock
    model: nova-premier
    enabled: true
  - provider: azure_openai
    model: gpt-5
    enabled: true
  - provider: ollama
    model: qwen3:1.7b
    endpoint: http://localhost:11434
    enabled: false

# --- Azure OpenAI ---
azure_openai:
  endpoint: ${AZURE_OPENAI_ENDPOINT}
  api_key: ${AZURE_OPENAI_API_KEY}
  deployment: ${AZURE_OPENAI_DEPLOYMENT}
  api_version: ${AZURE_OPENAI_API_VERSION}

# --- Metrics and Thresholds ---
metrics:
  - clinical_correctness
  - completeness
  - specificity
  - component_coverage

thresholds:
  clinical_correctness: 0.80
  completeness: 0.75
  specificity: 0.80
  component_coverage: 0.70

# --- Execution ---
execution:
  batch_size: 5

# --- Gold Standard Paths ---
gold_standard:
  gold_file_path: data/gold_standard/gold_standard.json
  pipeline_output_path: data/pipeline_output/pipeline_output.json

# --- Report ---
flag_for_review: true
```

---

## 4. API Endpoints

### 4.1 `POST /evaluate` — Single Note Evaluation

Evaluates a single clinical note's pipeline output.

**Request body:**
```json
{
  "pipeline_output": {
    "note_id": "note_4",
    "docname": "Note 4 - DiabetesMellitusSOAPNote2",
    "api_response": {
      "raw_text": "SUBJECTIVE: I am asked to see the patient...",
      "normalized_terms": [
        {
          "term": "Improving Unspecified diabetes mellitus",
          "normalize_payload": {
            "code": "48686997",
            "title": "diabetes mellitus, unspecified",
            "default_lexical_code": "29688",
            "default_lexical_title": "Diabetes mellitus",
            "metadata": {
              "mappings": {
                "icd10cm": { "codes": [...] },
                "snomedInternational": { "codes": [...] }
              }
            }
          }
        }
      ]
    }
  },
  "judges": ["claude-3.7-sonnet", "nova-premier"],
  "prompt_template": "prompt_a"
}
```

**Required fields:**
- `pipeline_output`: full pipeline JSON for one note (same structure as pipeline produces)
- `judges`: list of model names (minimum 1). Must match model names in config. Only enabled models in config are available.
- `prompt_template`: `"prompt_a"` or `"prompt_b"`

**Response:**
```json
{
  "note_id": "note_4",
  "verdict": "FAIL",
  "term_results": [
    {
      "term": "Improving Unspecified diabetes mellitus",
      "default_lexical_title": "Diabetes mellitus",
      "scores": {
        "clinical_correctness": {
          "aggregated": 0.9,
          "per_judge": { "claude-3.7-sonnet": 0.9, "nova-premier": 0.85 }
        },
        "completeness": {
          "aggregated": 0.5,
          "per_judge": { "claude-3.7-sonnet": 0.5, "nova-premier": 0.5 }
        },
        "specificity": {
          "aggregated": 0.5,
          "per_judge": { "claude-3.7-sonnet": 0.5, "nova-premier": 0.55 }
        },
        "component_coverage": {
          "aggregated": 0.4,
          "per_judge": { "claude-3.7-sonnet": 0.4, "nova-premier": 0.4 }
        }
      },
      "failed_dimensions": ["completeness", "specificity", "component_coverage"],
      "justifications": {
        "claude-3.7-sonnet": {
          "clinical_correctness": "Term is supported by the note...",
          "completeness": "Term does not capture hypoglycemia finding...",
          "specificity": "Term is too generic, missing complications...",
          "component_coverage": "Missing: hypoglycemia complication, control status not specified despite evidence of problematic glucose control..."
        }
      },
      "suggested_corrections": [
        {
          "judge": "claude-3.7-sonnet",
          "issue": "Missing hypoglycemia and control status",
          "current": "Diabetes mellitus",
          "suggested": "Uncontrolled diabetes mellitus with hypoglycemia"
        }
      ]
    },
    {
      "term": "Improving Unspecified diabetes mellitus with long-term current use of insulin",
      "default_lexical_title": "Long-term current use of insulin for diabetes mellitus",
      "scores": {
        "clinical_correctness": {
          "aggregated": 0.9,
          "per_judge": { "claude-3.7-sonnet": 0.9, "nova-premier": 0.9 }
        },
        "completeness": {
          "aggregated": 0.7,
          "per_judge": { "claude-3.7-sonnet": 0.7, "nova-premier": 0.7 }
        },
        "specificity": {
          "aggregated": 0.8,
          "per_judge": { "claude-3.7-sonnet": 0.8, "nova-premier": 0.8 }
        },
        "component_coverage": {
          "aggregated": 0.6,
          "per_judge": { "claude-3.7-sonnet": 0.6, "nova-premier": 0.6 }
        }
      },
      "failed_dimensions": ["component_coverage"],
      "justifications": { "..." : "..." },
      "suggested_corrections": []
    }
  ],
  "note_summary": {
    "total_terms": 2,
    "terms_passed": 0,
    "terms_failed": 2,
    "avg_scores": {
      "clinical_correctness": 0.9,
      "completeness": 0.6,
      "specificity": 0.65,
      "component_coverage": 0.5
    }
  },
  "flagged_for_review": true
}
```

---

### 4.2 `POST /batch_evaluate` — Batch Evaluation

Evaluates multiple notes at once. Same logic as `/evaluate` but for a list.

**Request body:**
```json
{
  "pipeline_outputs": [
    { "note_id": "note_4", "docname": "...", "api_response": { "..." : "..." } },
    { "note_id": "note_5", "docname": "...", "api_response": { "..." : "..." } }
  ],
  "judges": ["claude-3.7-sonnet"],
  "prompt_template": "prompt_b"
}
```

**Response:**
```json
{
  "results": [
    {
      "note_id": "note_4",
      "verdict": "FAIL",
      "term_results": [ "...per-term scores as in /evaluate..." ],
      "note_summary": { "total_terms": 2, "terms_passed": 0, "terms_failed": 2, "avg_scores": { "..." : "..." } }
    },
    {
      "note_id": "note_5",
      "verdict": "PASS",
      "term_results": [ "..." ],
      "note_summary": { "..." : "..." }
    }
  ],
  "aggregate": {
    "total_notes": 2,
    "pass_count": 1,
    "fail_count": 1,
    "pass_rate": 0.50,
    "total_terms_evaluated": 5,
    "terms_passed": 3,
    "terms_failed": 2,
    "avg_scores": {
      "clinical_correctness": 0.85,
      "completeness": 0.65,
      "specificity": 0.70,
      "component_coverage": 0.60
    },
    "most_common_failures": ["component_coverage", "completeness"],
    "worst_performing_notes": ["note_4"]
  }
}
```

---

### 4.3 `POST /gold_evaluate` — Gold Standard Evaluation

Runs LLM judge on the gold standard notes and computes P/R/F1 to validate the judge itself.

**Flow:**
1. Read gold standard JSON from config path (`gold_standard.gold_file_path`)
2. Read pipeline output JSON from config path (`gold_standard.pipeline_output_path`)
3. Match by note_id (gold `id` field = pipeline `note_id` field)
4. For each matched note: run LLM judge (reference-free) on clinical note + pipeline output
5. Judge produces scores + suggested corrections
6. Compare pipeline predicted codes against gold standard expected codes
7. Compute Precision, Recall, F1 per code system

**Request body:**
```json
{
  "judges": ["claude-3.7-sonnet", "nova-premier"],
  "prompt_template": "prompt_a"
}
```

**Response:**
```json
{
  "total_gold_notes": 20,
  "per_note_results": [
    {
      "note_id": "note_4",
      "verdict": "FAIL",
      "scores": { "..." : "..." },
      "suggested_corrections": [],
      "gold_expected": {
        "title": "Diabetes mellitus with hypoglycemia, with long-term current use of insulin",
        "code": "64855057"
      },
      "pipeline_predicted": [
        { "default_lexical_title": "Diabetes mellitus", "default_lexical_code": "29688" },
        { "default_lexical_title": "Long-term current use of insulin for diabetes mellitus", "default_lexical_code": "57821885" }
      ]
    }
  ],
  "judge_validation_metrics": {
    "imo": {
      "precision": 0.0,
      "recall": 0.0,
      "f1": 0.0
    },
    "icd10": {
      "precision": 0.85,
      "recall": 0.70,
      "f1": 0.77
    },
    "snomed": {
      "precision": 0.80,
      "recall": 0.65,
      "f1": 0.72
    }
  },
  "aggregate": {
    "pass_rate": 0.45,
    "avg_scores": {
      "clinical_correctness": 0.82,
      "completeness": 0.60,
      "specificity": 0.68,
      "component_coverage": 0.55
    }
  }
}
```

**P/R/F1 computation logic:**
- For each note: collect all predicted codes (from pipeline `normalized_terms`) and all gold expected codes
- Precision = |predicted ∩ gold| / |predicted|
- Recall = |predicted ∩ gold| / |gold|
- F1 = 2 × (P × R) / (P + R)
- Computed separately for IMO codes, ICD-10 codes, SNOMED codes
- Aggregated across all notes (micro-average)

---

### 4.4 `GET /health` — Health Check

**Response:**
```json
{
  "status": "healthy",
  "config_loaded": true,
  "available_judges": ["claude-3.7-sonnet", "nova-premier"],
  "ollama_reachable": false
}
```

---

### 4.5 `GET /models` — List Available Models

**Response:**
```json
{
  "models": [
    { "name": "claude-3.7-sonnet", "provider": "bedrock", "enabled": true },
    { "name": "nova-premier", "provider": "bedrock", "enabled": true },
    { "name": "gpt-5", "provider": "azure_openai", "enabled": true },
    { "name": "qwen3:1.7b", "provider": "ollama", "enabled": false }
  ]
}
```

---

## 5. Prompt Template Files

### 5.1 `prompts/prompt_a_single.txt` — Single Prompt

```
You are a clinical coding quality evaluator. You will be given a clinical
note and a set of predicted medical terms with their associated ICD-10
and SNOMED code descriptions. Evaluate how well EACH term individually
captures the relevant clinical content from the note. Score each term
separately.

DIABETES TERM CONSTRUCTION RULES:
For diabetes-related terms, the expected component ordering is:
1. Control Status — "Controlled" or "Uncontrolled" (only if the note
   provides evidence of control status). Precedes diabetes type.
2. Diabetes Type — "Type 1" or "Type 2" (only if the note specifies type)
3. Complications — use natural word order: "diabetes mellitus with
   retinopathy" NOT "retinopathy with diabetes mellitus"
4. Insulin Status — "with long-term current use of insulin" (only if the
   note evidences insulin use)
5. Oral Medication — "with oral medication" (only if the note evidences it)

Example: "Uncontrolled Type 2 diabetes mellitus with retinopathy with
long-term current use of insulin"

Do NOT penalize missing components if the clinical note lacks evidence
for them. Only penalize if the note has evidence and the term omits it,
or if components are present but in the wrong order.

For EACH predicted term, score these 4 dimensions on a scale of 0.0 to 1.0:

CLINICAL_CORRECTNESS: Is this predicted term and its associated ICD-10
and SNOMED descriptions clinically supported by the note?
- 1.0 = fully supported by the note
- 0.5 = partially supported or ambiguous
- 0.0 = contradicted or unsupported

COMPLETENESS: Does this term fully capture the clinical finding it
represents? Are relevant aspects of that finding included?
- 1.0 = the finding is fully represented
- 0.5 = the core finding is captured but some aspects are missing
- 0.0 = the term barely represents the finding

SPECIFICITY: Is this term precise enough? Not too broad, not hallucinated?
- 1.0 = appropriately specific
- 0.5 = too generic (e.g., "diabetes" instead of "diabetes with
  hypoglycemia" when hypoglycemia is evidenced in the note)
- 0.0 = vague or fabricated

COMPONENT_COVERAGE: Does this term capture all relevant clinical modifiers
and qualifiers evidenced in the note? For diabetes terms, are the
components present in the correct order per the construction rules above?
- 1.0 = all evidenced modifiers captured in correct order
- 0.5 = some evidenced modifiers missing or ordering incorrect
- 0.0 = most evidenced modifiers missing

For any term with any dimension scoring below 1.0, suggest specific corrections.

OUTPUT FORMAT (respond ONLY with this JSON, no other text):
{
  "term_evaluations": [
    {
      "term": "the predicted term text",
      "scores": {
        "clinical_correctness": { "score": 0.0-1.0, "justification": "..." },
        "completeness": { "score": 0.0-1.0, "justification": "..." },
        "specificity": { "score": 0.0-1.0, "justification": "..." },
        "component_coverage": { "score": 0.0-1.0, "justification": "..." }
      },
      "suggested_corrections": [
        {
          "issue": "description of the problem",
          "current": "current predicted term or description",
          "suggested": "what it should be"
        }
      ],
      "verdict": "PASS or FAIL"
    }
  ]
}

---

CLINICAL NOTE:
{clinical_note}

PREDICTED TERMS:
{predicted_terms}
```

### 5.2 `prompts/prompt_b_cot.txt` — Chain-of-Thought

```
You are a clinical coding quality evaluator. You will be given a clinical
note and a set of predicted medical terms with their associated ICD-10
and SNOMED code descriptions. Evaluate how well EACH term individually
captures the relevant clinical content from the note. Score each term
separately.

DIABETES TERM CONSTRUCTION RULES:
For diabetes-related terms, the expected component ordering is:
1. Control Status — "Controlled" or "Uncontrolled" (only if the note
   provides evidence of control status). Precedes diabetes type.
2. Diabetes Type — "Type 1" or "Type 2" (only if the note specifies type)
3. Complications — use natural word order: "diabetes mellitus with
   retinopathy" NOT "retinopathy with diabetes mellitus"
4. Insulin Status — "with long-term current use of insulin" (only if the
   note evidences insulin use)
5. Oral Medication — "with oral medication" (only if the note evidences it)

Example: "Uncontrolled Type 2 diabetes mellitus with retinopathy with
long-term current use of insulin"

Do NOT penalize missing components if the clinical note lacks evidence
for them. Only penalize if the note has evidence and the term omits it,
or if components are present but in the wrong order.

Follow these steps in order:

STEP 1 - ANALYZE THE NOTE:
List all clinically significant findings, diagnoses, conditions, and
relevant status indicators from the clinical note. For diabetes notes,
specifically identify: control status evidence, diabetes type evidence,
complications, insulin use evidence, oral medication evidence.

STEP 2 - ASSESS EACH PREDICTED TERM:
For each predicted term and its associated code descriptions:
- Is it supported by the clinical note?
- Is it specific enough or too generic?
- Does it capture all relevant modifiers evidenced in the note?
- Are the ICD-10 and SNOMED descriptions consistent with the note?
- For diabetes terms: are components in the correct order?

STEP 3 - IDENTIFY GAPS PER TERM:
For each term, list any clinically significant aspects of the finding
that the term should capture but does not.

STEP 4 - SCORE EACH TERM:
For EACH term, score 4 dimensions on a scale of 0.0 to 1.0:

CLINICAL_CORRECTNESS: Is this predicted term and its associated ICD-10
and SNOMED descriptions clinically supported by the note?
- 1.0 = fully supported
- 0.5 = partially supported or ambiguous
- 0.0 = contradicted or unsupported

COMPLETENESS: Does this term fully capture the clinical finding it
represents?
- 1.0 = the finding is fully represented
- 0.5 = core finding captured but some aspects missing
- 0.0 = term barely represents the finding

SPECIFICITY: Is this term precise enough? Not too broad, not hallucinated?
- 1.0 = appropriately specific
- 0.5 = too generic
- 0.0 = vague or fabricated

COMPONENT_COVERAGE: Does this term capture all relevant clinical modifiers
evidenced in the note? For diabetes terms, are components in correct order?
- 1.0 = all evidenced modifiers captured in correct order
- 0.5 = some evidenced modifiers missing or ordering incorrect
- 0.0 = most evidenced modifiers missing

STEP 5 - SUGGEST CORRECTIONS:
For each term with any dimension scoring below 1.0, suggest specific
corrections.

OUTPUT FORMAT (respond ONLY with this JSON, no other text):
{
  "chain_of_thought": {
    "clinical_findings": ["list all findings from the note"],
    "diabetes_components_evidenced": {
      "control_status": "evidence or null",
      "diabetes_type": "evidence or null",
      "complications": ["list or empty"],
      "insulin_use": "evidence or null",
      "oral_medication": "evidence or null"
    }
  },
  "term_evaluations": [
    {
      "term": "the predicted term text",
      "assessment": {
        "icd10_descriptions": ["list"],
        "snomed_descriptions": ["list"],
        "supported": true/false,
        "modifiers_present": ["list"],
        "modifiers_missing": ["list"],
        "ordering_correct": true/false,
        "explanation": "brief assessment"
      },
      "scores": {
        "clinical_correctness": { "score": 0.0-1.0, "justification": "..." },
        "completeness": { "score": 0.0-1.0, "justification": "..." },
        "specificity": { "score": 0.0-1.0, "justification": "..." },
        "component_coverage": { "score": 0.0-1.0, "justification": "..." }
      },
      "suggested_corrections": [
        {
          "issue": "description of the problem",
          "current": "current predicted term or description",
          "suggested": "what it should be"
        }
      ],
      "verdict": "PASS or FAIL"
    }
  ]
}

---

CLINICAL NOTE:
{clinical_note}

PREDICTED TERMS:
{predicted_terms}
```

---

## 6. Implementation Notes

### 6.1 Input Loader (`services/input_loader.py`)

Flattens ALL pipeline JSON `normalized_terms` into judge-ready format for a single prompt:

```python
# Input: raw pipeline JSON with multiple normalized_terms
# Output: list of flattened term structures

[
    {
        "term": "Improving Unspecified diabetes mellitus",
        "default_lexical_title": "Diabetes mellitus",
        "default_lexical_code": "29688",
        "imo_code": "48686997",
        "icd10": [
            {"code": "E11.9", "title": "Type 2 diabetes mellitus without complications"}
        ],
        "snomed": [
            {"code": "73211009", "title": "Diabetes mellitus"}
        ]
    },
    { ... next term ... }
]
```

For prompt injection, ALL terms formatted together:
```
TERM 1:
- Term: Diabetes mellitus
  ICD-10: E11.9 - Type 2 diabetes mellitus without complications
  SNOMED: 73211009 - Diabetes mellitus

TERM 2:
- Term: Long-term current use of insulin for diabetes mellitus
  ICD-10: Z79.4 - Long term (current) use of insulin
  ICD-10: E11.9 - Type 2 diabetes mellitus without complications
  SNOMED: 710815001 - Long-term current use of insulin
```

The judge sees descriptions, not just codes. IMO codes are included in the flattened struct for reporting but NOT passed to the judge prompt.

### 6.2 Judge Registry (`services/judge_registry.py`)

- Reads config, builds registry of available models
- Validates requested judges against enabled models in config
- Returns appropriate provider instance per model
- If a requested judge is not enabled in config, return error

### 6.3 Providers (`providers/`)

Abstract base class:
```python
class BaseProvider:
    async def evaluate(
        self,
        clinical_note: str,
        predicted_terms: str,
        prompt_template: str
    ) -> dict:
        """Send prompt to model with all terms, return parsed JSON response with per-term scores."""
        raise NotImplementedError
```

Provider implementations:
- `bedrock_provider.py` — uses boto3 bedrock-runtime for Claude 3.7 Sonnet, Nova Premier
- `azure_openai_provider.py` — uses openai Python SDK with Azure config for GPT-5. Reads credentials from environment variables (AZURE_OPENAI_API_KEY, AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_DEPLOYMENT, AZURE_OPENAI_API_VERSION). Uses `openai.AzureOpenAI` client.
- `ollama_provider.py` — calls Ollama REST API at configured endpoint (default: http://localhost:11434)

Each provider:
1. Loads the selected prompt template file (prompt_a or prompt_b)
2. Injects clinical_note and ALL predicted_terms into template
3. Sends single request to model
4. Parses JSON response containing per-term scores array (handle malformed JSON gracefully)
5. Returns structured scores dict with scores for each term

**Azure OpenAI provider example:**
```python
from openai import AzureOpenAI

client = AzureOpenAI(
    api_key=config.azure_openai.api_key,
    api_version=config.azure_openai.api_version,
    azure_endpoint=config.azure_openai.endpoint
)

response = client.chat.completions.create(
    model=config.azure_openai.deployment,  # "gpt-5"
    messages=[
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ],
    temperature=0.0,
    response_format={"type": "json_object"}
)
```

### 6.4 Judge (`services/judge.py`)

Core orchestration:
1. Receive evaluation request
2. Call input_loader to flatten ALL pipeline terms
3. Get enabled judges from judge_registry
4. **For each judge:** single LLM call with clinical_note + ALL terms → returns per-term scores in one response
5. Collect all judge responses
6. Pass to aggregator (aggregates per-term scores across judges)
7. Apply threshold gate per term
8. Derive note-level verdict: if at least ONE term passes → note PASSES (only if ALL terms fail → note FAILS)

**API call count:** 1 call per judge per note. A note with 2 judges = 2 API calls total, regardless of term count.

### 6.5 Aggregator (`services/aggregator.py`)

- If single judge: passthrough, no aggregation
- If multiple judges: compute average (or median) per dimension across judges
- Output: aggregated scores + per-judge breakdown

### 6.6 Threshold Gate (`services/threshold_gate.py`)

- Read thresholds from config
- **Applied per term:** For each dimension of each term, check if aggregated score >= threshold
- All dimensions pass for a term → term verdict = "PASS"
- Any dimension fails for a term → term verdict = "FAIL", record which dimensions failed
- **Note-level verdict:** If at least ONE term passes → note verdict = "PASS" (only if ALL terms fail → note verdict = "FAIL")
- Attach flag_for_review based on config

### 6.7 Report Generator (`services/report_generator.py`)

Per-note report:
- note-level verdict, per-term results (scores, failed_dimensions, justifications, suggested_corrections per term), note_summary (avg scores, terms passed/failed), flagged_for_review

Aggregate report (batch/gold only):
- total_notes, pass_count, fail_count, pass_rate, total_terms_evaluated, terms_passed, terms_failed, avg_scores, most_common_failures, worst_performing_notes

### 6.8 Gold Evaluator (`services/gold_evaluator.py`)

Flow:
1. Read gold standard JSON from config path
2. Read pipeline output JSON from config path
3. Match by note_id (gold `id` field = pipeline `note_id` field)
4. For each matched note:
   a. Run LLM judge (same as /evaluate)
   b. Collect judge scores + suggested corrections
5. Compare pipeline predicted codes against gold standard expected codes:
   - Extract predicted: IMO codes, ICD-10 codes, SNOMED codes from pipeline `normalized_terms`
   - Extract gold: IMO code from gold `golds[].code`
   - For ICD-10/SNOMED: lookup from gold IMO code mappings if available
6. Compute P/R/F1:
   - Per code system (IMO, ICD-10, SNOMED)
   - Precision = |predicted ∩ gold| / |predicted|
   - Recall = |predicted ∩ gold| / |gold|
   - F1 = 2 × (P × R) / (P + R)
   - Micro-averaged across all notes

**Gold standard JSON structure:**
```json
{
  "id": "note_4",
  "golds": [
    {
      "id": 1,
      "code": "64855057",
      "title": "Diabetes mellitus with hypoglycemia, with long-term current use of insulin",
      "evidence_groups": ["diabetes_type", "hypoglycemia", "insulin_use"]
    }
  ]
}
```

**Pipeline output JSON structure:**
```json
{
  "note_id": "note_4",
  "api_response": {
    "normalized_terms": [
      {
        "term": "...",
        "normalize_payload": {
          "code": "...",
          "default_lexical_code": "...",
          "default_lexical_title": "...",
          "metadata": {
            "mappings": {
              "icd10cm": { "codes": [...] },
              "snomedInternational": { "codes": [...] }
            }
          }
        }
      }
    ]
  }
}
```

---

## 7. Pydantic Schemas

### 7.1 Request Schemas (`schemas/requests.py`)

```python
from pydantic import BaseModel, Field
from typing import List, Literal

class EvaluateRequest(BaseModel):
    pipeline_output: dict
    judges: List[str] = Field(..., min_length=1)
    prompt_template: Literal["prompt_a", "prompt_b"]

class BatchEvaluateRequest(BaseModel):
    pipeline_outputs: List[dict] = Field(..., min_length=1)
    judges: List[str] = Field(..., min_length=1)
    prompt_template: Literal["prompt_a", "prompt_b"]

class GoldEvaluateRequest(BaseModel):
    judges: List[str] = Field(..., min_length=1)
    prompt_template: Literal["prompt_a", "prompt_b"]
```

### 7.2 Response Schemas (`schemas/responses.py`)

```python
from pydantic import BaseModel
from typing import Dict, List, Literal, Optional

class DimensionScore(BaseModel):
    aggregated: float
    per_judge: Dict[str, float]

class SuggestedCorrection(BaseModel):
    judge: str
    issue: str
    current: str
    suggested: str

class TermResult(BaseModel):
    term: str
    default_lexical_title: str
    scores: Dict[str, DimensionScore]
    failed_dimensions: List[str]
    justifications: Dict[str, Dict[str, str]]
    suggested_corrections: List[SuggestedCorrection]
    verdict: Literal["PASS", "FAIL"]

class NoteSummary(BaseModel):
    total_terms: int
    terms_passed: int
    terms_failed: int
    avg_scores: Dict[str, float]

class EvaluateResponse(BaseModel):
    note_id: str
    verdict: Literal["PASS", "FAIL"]
    term_results: List[TermResult]
    note_summary: NoteSummary
    flagged_for_review: bool

class AggregateStats(BaseModel):
    total_notes: int
    pass_count: int
    fail_count: int
    pass_rate: float
    total_terms_evaluated: int
    terms_passed: int
    terms_failed: int
    avg_scores: Dict[str, float]
    most_common_failures: List[str]
    worst_performing_notes: List[str]

class BatchEvaluateResponse(BaseModel):
    results: List[EvaluateResponse]
    aggregate: AggregateStats

class CodeSystemMetrics(BaseModel):
    precision: float
    recall: float
    f1: float

class GoldNoteResult(BaseModel):
    note_id: str
    verdict: Literal["PASS", "FAIL"]
    term_results: List[TermResult]
    note_summary: NoteSummary
    gold_expected: List[dict]
    pipeline_predicted: List[dict]

class GoldEvaluateResponse(BaseModel):
    total_gold_notes: int
    per_note_results: List[GoldNoteResult]
    judge_validation_metrics: Dict[str, CodeSystemMetrics]
    aggregate: AggregateStats
```

---

## 8. Error Handling

- Requested judge not enabled in config → 400 with available models list
- Ollama endpoint unreachable → 503 with message
- LLM returns malformed JSON → retry once, then return partial result with error flag
- Gold standard file not found → 404 with config path in message
- Pipeline output has no normalized_terms → 422 with field-level error
- note_id mismatch between gold and pipeline → 400 listing unmatched note_ids

---

## 9. Dependencies (`requirements.txt`)

```
fastapi>=0.104.0
uvicorn>=0.24.0
pydantic>=2.0.0
pyyaml>=6.0
boto3>=1.34.0
openai>=1.0.0
httpx>=0.25.0
python-dotenv>=1.0.0
```

---

## 10. How to Run

```bash
# Set up environment variables
cp .env.example .env
# Edit .env with your credentials:
# AZURE_OPENAI_API_KEY=your_key
# AZURE_OPENAI_DEPLOYMENT=gpt-5
# AZURE_OPENAI_ENDPOINT=https://your-endpoint.openai.azure.com/
# AZURE_OPENAI_API_VERSION=2024-12-01-preview

# Install dependencies
pip install -r requirements.txt

# Start server
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload

# Endpoints
# POST http://localhost:8000/evaluate
# POST http://localhost:8000/batch_evaluate
# POST http://localhost:8000/gold_evaluate
# GET  http://localhost:8000/health
# GET  http://localhost:8000/models
```

---

## 11. Key Design Decisions

1. **Per-term scoring in single prompt**: All terms sent in one LLM call per judge. The LLM scores each term independently within its response. This gives granular per-term diagnostics with minimal API calls (1 call per judge per note). Note-level verdict: if at least one term passes, note passes (only if all terms fail, note fails).

2. **Reference-free evaluation**: LLM judge reads only the clinical note + pipeline output. No gold standard at runtime. This allows the judge to scale beyond the 20 annotated notes.

3. **Diabetes-specific term construction rules**: The judge prompt includes diabetes term ordering rules (control status → type → complications → insulin → oral medication). Components are only penalized when the clinical note provides evidence for them. Rules are embedded in prompt templates for easy modification.

4. **IMO codes excluded from LLM evaluation**: IMO is proprietary. LLM cannot assess IMO code correctness. IMO accuracy is only measured in gold_evaluation via P/R/F1.

5. **Prompt template selected per request**: Allows A/B testing between Prompt A (single) and Prompt B (chain-of-thought) to determine which performs better.

6. **Multi-provider support**: Bedrock (Claude, Nova), Azure OpenAI (GPT-5), Ollama (local models). Single or multi-judge configurable. Aggregator skipped for single judge.

7. **Configurable thresholds**: Per-dimension thresholds in config. All dimensions must meet threshold per term for PASS. Values TBD after initial runs.

8. **Gold evaluation validates the judge**: Runs LLM judge on gold notes, computes P/R/F1 per code system. This measures how well the judge + pipeline perform against known correct answers.

9. **Azure OpenAI for GPT-5**: GPT-5 accessed via Azure OpenAI endpoint. Credentials stored in environment variables, referenced in config. Uses openai Python SDK with AzureOpenAI client.

---

## 12. Test Client (HTML UI)

A standalone HTML test client (`test-client.html`) is provided for easy API testing without needing external tools like Postman.

**Features:**
- Single-page application with tabs for each API endpoint
- Configurable API base URL (defaults to `http://localhost:8000`)
- Raw JSON input via textareas
- **Hierarchical tree display** for readable output (not raw JSON)
- Color-coded verdicts (PASS = green, FAIL = red)
- Detailed term-by-term breakdown showing:
  - Scores per dimension (aggregated + per-judge)
  - Failed dimensions
  - Justifications per judge per metric
  - Suggested corrections with judge attribution
- Health status indicators
- Models table with enabled/disabled status
- Aggregate statistics for batch and gold evaluations
- P/R/F1 metrics display for gold evaluation

**Usage:**
```bash
# Open in browser (FastAPI server must be running)
# For Windows:
start test-client.html

# For Mac/Linux:
open test-client.html
# or
xdg-open test-client.html
```

**Output Format (Hierarchical Tree):**
```
📄 Note: note_4 | Verdict: FAIL ✗
├─ Term 1: Diabetes mellitus
│  ├─ Default Lexical: Diabetes mellitus
│  ├─ Verdict: FAIL ✗
│  ├─ Scores:
│  │  ├─ Clinical Correctness: 0.90 (qwen3:1.7b: 0.90)
│  │  ├─ Completeness: 0.50 (qwen3:1.7b: 0.50)
│  │  ├─ Specificity: 0.50 (qwen3:1.7b: 0.50)
│  │  └─ Component Coverage: 0.40 (qwen3:1.7b: 0.40)
│  ├─ Failed Dimensions: completeness, specificity, component_coverage
│  ├─ Justifications:
│  │  └─ qwen3:1.7b:
│  │     ├─ Clinical Correctness: "Term is supported by the note..."
│  │     ├─ Completeness: "Missing hypoglycemia finding..."
│  │     └─ ...
│  └─ Suggested Corrections:
│     └─ [qwen3:1.7b] Missing hypoglycemia and control status
│        ├─ Current: Diabetes mellitus
│        └─ Suggested: Uncontrolled diabetes mellitus with hypoglycemia
└─ Summary:
   ├─ Total Terms: 2
   ├─ Passed: 0
   ├─ Failed: 2
   └─ Average Scores: clin=0.9, comp=0.6, spec=0.65, comp=0.5
```

**Tabs:**
1. **Health** - Check service health and available judges
2. **Models** - List all configured models with their status
3. **Evaluate** - Single note evaluation with full term details
4. **Batch Evaluate** - Multiple notes with aggregate statistics
5. **Gold Evaluate** - Gold standard validation with P/R/F1 metrics

The test client makes it easy to:
- Quickly verify API is running
- Test different judge models
- Compare prompt templates (prompt_a vs prompt_b)
- Review detailed evaluation results in readable format
- Debug pipeline output issues
