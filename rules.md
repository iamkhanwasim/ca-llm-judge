Business Logic (NLP) — Diabetes Entity Construction + Normalization/KG Routing 

This is a practical rule-spec to implement as post-processing over extracted entities (from NER + relation/attribute extraction). It produces: 

Final extracted entity (display/clinical statement): richer, keeps type/control/complications/therapy wording. 

Entity to send to Normalization/KG: optimized for matching (removes/adjusts attributes per rules, with fallback retry). 

 

1) Inputs (from extraction) 

Assume the NLP pipeline outputs a structured frame per mention/window (sentence/section-level aggregation): 

Diabetes → e.g., "diabetes mellitus" 

Diabetes_type → e.g., type 1 | type 2 | gestational | juvenile | other 

Diabetes_control_status → e.g., uncontrolled | poorly controlled | suboptimal control | ... 

Glycemic_status → e.g., hyperglycemia | hypoglycemia (often considered a diabetes complication/manifestation) 

Complications[] → list of findings (gastroparesis, CKD, neuropathy, etc.) + evidence flags (explicitly due-to diabetes vs not) 

Therapy.insulin_use → active insulin use (exclude noncompliance) 

Therapy.insulin_pump → pump present (external/internal) 

Therapy.oral_hypoglycemic_use → active oral meds (exclude noncompliance), optionally with specific drug names (e.g., metformin) 

Refer to: 

Dictionaries 

DM_TYPE_DICT 

INSULIN_DICT (to detect insulin) 

NON-INSULIN_INJECTABLE_DICT 

ORAL_HYPOGLYCEMIC_DICT 

DM_COMPLICATIONS_IMPLICIT_DICT (allowed to be causally linked even if not explicitly stated) 

DM_COMPLICATIONS_EXPLICIT_ONLY_DICT (only link if explicitly “due to/from/secondary to/diabetic…”) 

Normalizer returning: 

best_match.concept_id 

best_match.lexical 

best_match.confidence 

 

2) Output Objects 

For each diabetes “case” generate: 

final_entity_text (what you store/show as extracted) 

normalize_query_text (what you send to normalizer/KG) 

normalize_result (concept + confidence)  

Optional: fallback_attempts[] (what was tried) 

 

3) Rule Set A — Diabetes Type Handling 

A1. Always retain diabetes type in the final extracted entity 

If Diabetes.type present (e.g., Type 1, Type 2, gestational, juvenile), include it in the final entity string. 

A2. If diabetes type is not mentioned 

Use “diabetes mellitus” as the base (final and normalize query), i.e., treat “diabetes” as “diabetes mellitus”. 

A3. Special rule: Type 2 “drop type” fallback for normalization (strictly Type 2) 

If ALL are true: 

Diabetes.type == "type 2" OR 

Diabetes.type is not specified as “type 2” in text OR there is a complicating attribute like: 

complications present, and/or 

oral hypoglycemic medication use present 

AND normalization returns no direct match or confidence < TYPE2_MATCH_THRESHOLD 

Then: 

Keep the final extracted entity as the richer Type 2 statement (don’t degrade display text). 

But remove “type 2” from the normalization query and re-query using a more matchable phrase. 

Example  

Extracted: 

Diabetes: diabetes mellitus  

Diabetes_type: type 2  

Oral_hypoglycemic_use: metformin 

Final extracted entity: 

type 2 diabetes mellitus treated with oral medication 

Normalize/KG query (fallback if low confidence/no match): 

diabetes mellitus treated with oral medication 

 

4) Rule Set B — Control Status Handling 

B1. Control status is retained in the final extracted entity 

If control status is present, prepend/attach it: 

“uncontrolled type 2 diabetes mellitus” 

“poorly controlled type 2 diabetes mellitus” 

B2. Control status is removed from the normalize/KG query (default) 

Remove these control status variants when building the normalize query: 

uncontrolled 

not in control 

suboptimal control / suboptimally controlled 

Example 

Final: 

uncontrolled type 2 diabetes mellitus 

Normalize query: 

type 2 diabetes mellitus 

B3. Exception: uncontrolled + (hyperglycemia or hypoglycemia) → keep “uncontrolled” in normalize query 

If: 

Diabetes.control_status == "uncontrolled" AND 

Glycemic_status in {"hyperglycemia","hypoglycemia"} (either extracted as glycemic_status or as a recognized complication) 

Then keep “uncontrolled” in the normalize query because the combined concept often exists as a single normalized concept (or maps better). 

Example 

Final: 

uncontrolled type 2 diabetes mellitus with hyperglycemia 

Normalize query: 

uncontrolled type 2 diabetes mellitus with hyperglycemia 

B4. “Poorly controlled” special fallback logic 

For Diabetes.control_status == "poorly controlled": 

First try normalize with “poorly controlled …” included only when a direct concept match exist (or just always try it first). 

If normalizer returns no match or confidence < POOR_CONTROLLED_THRESHOLD: 

Remove “poorly controlled” and retry. 

Additionally, send diabetes + poorly controlled as a separate normalization request (so you still capture the attribute somewhere). 

Example 

Extracted: 

type 2 DM 

poorly controlled 

gastroparesis 

CKD 

Final entities (display): 

poorly controlled type 2 diabetes mellitus with gastroparesis 

poorly controlled type 2 diabetes mellitus with chronic kidney disease 

Normalize/KG queries: 

Try: poorly controlled type 2 diabetes mellitus with gastroparesis (keep poorly controlled if it matches well) 

For CKD: type 2 diabetes mellitus with chronic kidney disease (drop poorly controlled if no good direct match) 

Additional fallback note: 

If only CKD exists (no complication that matches well with “poorly controlled”): 

Send two queries: 

poorly controlled type 2 diabetes mellitus (attribute capture) 

type 2 diabetes mellitus with chronic kidney disease (complication capture) 

 

5) Rule Set C — Complications & Causal Linking 

C1. Complications allowed to link to diabetes even if not explicitly stated (implicit causal link) 

If a complication appears and it is in DM_COMPLICATIONS_IMPLICIT_DICT, you may construct: 

“diabetes mellitus with ” even when the note doesn’t say “due to diabetes”. 

Typical examples: diabetic neuropathy, retinopathy, nephropathy, diabetic gastroparesis, diabetic foot ulcer, etc. 

Implementation rule 

If complication.name in DM_COMPLICATIONS_IMPLICIT_DICT: 

Link it to diabetes by default (unless negated/ruled-out). 

C2. Complications that can be linked only if explicitly stated as due to diabetes (explicit-only) 

If complication is in DM_COMPLICATIONS_EXPLICIT_ONLY_DICT, only link when there is explicit attribution evidence, such as: 

“due to diabetes”, “secondary to diabetes”, “diabetic ”, “because of DM” 

Or a relation extractor outputs: cause(diabetes, complication) = true 

Implementation rule 

If complication.name in DM_COMPLICATIONS_EXPLICIT_ONLY_DICT: 

Link to diabetes only if complication.explicit_due_to_diabetes == true 

Otherwise, keep the complication as a separate problem (don’t attach “with …” to diabetes) 

C3. Building final entities with complications 

For each eligible linked complication: 

Create a separate final entity variant: 

<control_status?> <type?> diabetes mellitus with <complication> This yields multiple final entities when multiple complications exist. 

C4. Normalization query construction with complications 

Start with: <type?> diabetes mellitus with <complication> 

Apply control-status removal rules (Section 4), except the uncontrolled + glycemic-status exception. 

 

6) Rule Set D — Insulin Use 

D1. When insulin use is documented (active use; exclude noncompliance) 

Add therapy concept: 

“Long term (current) use of insulin” except when the patient has type 1/juvenile diabetes mellitus. 

Interpretation 

If Diabetes.type in {"type 1","juvenile"}: 

Do not add the “long term current use…” add-on (since insulin is expected/less discriminative). 

Else: 

Add it as a separate entity (recommended) and/or as a modifier: 

Separate entity is usually cleaner for KG: Long term (current) use of insulin 

Diabetes entity remains: type 2 diabetes mellitus (plus complications as applicable) 

D2. Insulin pump 

If insulin pump documented: 

Use: “Presence of insulin pump (external)” or “Presence of insulin pump (internal)” based on extraction. 

This is typically best as a separate normalized entity (device), not embedded into the diabetes string. 

 

7) Rule Set E — Oral Hypoglycemic Use 

E1. When oral hypoglycemic is documented (active use; exclude noncompliance) 

If no complication present: 

KG/normalize query can be combined: 

(<type?>) diabetes mellitus treated with oral medication 

If any complication present: 

KG/normalize queries should be: 

(<type?>) diabetes mellitus with <complication> (e.g., type 2 diabetes mellitus with chronic kidney disease) 

plus separately: Long term (current) use of oral hypoglycemic drugs 

Example: 

DM + CKD + metformin → send 

type 2 diabetes mellitus with chronic kidney disease 

Long term (current) use of oral hypoglycemic drugs 

DM + metformin only → send 

type 2 diabetes mellitus treated with oral medication 

 

8) Implementation Notes (to avoid common errors) 

Negation/noncompliance filtering must happen before adding insulin/oral therapy entities. 

Keep display entities clinically rich; keep normalize queries conservative. 

Generate multiple diabetes entities when multiple complications are linkable; this improves KG structure (one edge per complication). 

Do not auto-link explicit-only complications unless you have clear attribution evidence.