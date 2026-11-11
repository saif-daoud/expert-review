import json


def annotate_knowledge_graph_prompt(
    domain: str,
    system: str,
    user: str,
    interaction_unit: str,
    schema: dict,
    current_graph: dict,
    phase: dict,
    phase_utterances: list,
) -> str:
    """Prompt for a longitudinal graph transaction at one phase boundary."""

    node_descriptions = {
        n.get("type"): n.get("description", "")
        for n in schema.get("nodes", [])
        if isinstance(n, dict) and n.get("type")
    }

    input_obj = {
        "allowed_node_types": node_descriptions,
        "allowed_schema_edges": schema.get("edges", []),
        "current_graph": current_graph,
        "phase": phase,
        "phase_utterances": phase_utterances,
    }

    return f"""You maintain a LONGITUDINAL patient knowledge graph from {domain} dialogue.

You are given:
1. the fixed knowledge-graph schema;
2. the current populated graph before this update;
3. one completed macro phase of dialogue.

Your task is NOT to rewrite the graph. Propose only the smallest set of graph
operations needed to incorporate information supported by this phase.

Return ONLY valid JSON:
{{
  "operations": [ ... ]
}}

SUPPORTED OPERATIONS

1) ADD_NODE
Use when a genuinely new patient-specific fact/entity is not already represented.
{{
  "op": "ADD_NODE",
  "temp_id": "new_1",
  "node_type": "<exact allowed node type>",
  "content": "<concise patient-specific fact>",
  "confidence": 0.0,
  "provenance_kind": "explicit | supported_inference",
  "evidence_utterance_ids": ["<id from phase_utterances>"]
}}

2) UPDATE_NODE
Use when an existing node remains the same underlying fact but this phase refines,
strengthens, weakens, or changes its current status. Refer to the exact existing id.
Allowed changes: content, status, confidence, provenance_kind.
{{
  "op": "UPDATE_NODE",
  "node_id": "kg_...",
  "changes": {{"confidence": 0.92}},
  "reason": "optional short reason",
  "evidence_utterance_ids": ["..."]
}}

3) ADD_EDGE
Use to connect two existing nodes or new temp_ids. The node-type/relation/node-type
combination must be permitted by allowed_schema_edges.
{{
  "op": "ADD_EDGE",
  "source": "kg_... or new_1",
  "relation": "<schema relation>",
  "target": "kg_... or new_2",
  "confidence": 0.0,
  "evidence_utterance_ids": ["..."]
}}

4) UPDATE_EDGE
Use only to change status/confidence or add evidence to an existing edge.
Do NOT use it to change endpoints or relation; deactivate the old edge and add a
new one instead.
{{
  "op": "UPDATE_EDGE",
  "edge_id": "ke_...",
  "changes": {{"confidence": 0.88}},
  "reason": "optional short reason",
  "evidence_utterance_ids": ["..."]
}}

5) DEACTIVATE_NODE
Use when an existing fact is no longer currently true, was explicitly corrected,
was resolved, or should no longer be treated as active. NEVER hard-delete history.
Allowed statuses: historical, resolved, retracted, uncertain, inactive.
{{
  "op": "DEACTIVATE_NODE",
  "node_id": "kg_...",
  "status": "historical",
  "reason": "why it is no longer active",
  "evidence_utterance_ids": ["..."]
}}

6) DEACTIVATE_EDGE
Use when an existing relation should no longer be treated as active.
{{
  "op": "DEACTIVATE_EDGE",
  "edge_id": "ke_...",
  "status": "historical",
  "reason": "why the relation changed",
  "evidence_utterance_ids": ["..."]
}}

7) MERGE_NODES
Use ONLY for clear duplicate nodes of the SAME node type that refer to the same
underlying fact/entity. Keep the best existing node and merge duplicates into it.
{{
  "op": "MERGE_NODES",
  "keep_node_id": "kg_...",
  "merge_node_ids": ["kg_..."],
  "reason": "duplicate representations of the same fact"
}}

STATUS SEMANTICS
- active: currently supported/current.
- historical: previously true/relevant but not current.
- resolved: a prior problem/issue has been resolved.
- retracted: prior graph content was wrong or explicitly contradicted.
- uncertain: evidence is currently ambiguous or weakened.
- inactive: generic non-current state when none of the above fits.

IMPORTANT RULES
- Use EXACT node types from allowed_node_types.
- Use relations compatible with allowed_schema_edges.
- Reuse existing node ids whenever the same underlying fact is already represented.
- Do not create a paraphrase duplicate merely because wording differs.
- `Patient` is already represented in current_graph; never ADD_NODE for Patient.
- For a new relation involving Patient, use its existing patient node id.
- Every ADD_NODE that is referenced later in this transaction should have a unique
  `temp_id` such as new_1, new_2, ... .
- New facts should normally be connected into the graph with one or more ADD_EDGE
  operations when the schema permits it. Avoid isolated nodes when a supported
  relation is evident.
- Do not invent diagnoses, causes, history, or relationships not supported by the
  supplied phase dialogue.
- Prefer `explicit` provenance for directly stated facts. Use `supported_inference`
  only for conservative, strongly supported interpretations.
- Confidence is in [0, 1]. It is epistemic confidence in the extracted fact/relation,
  not symptom severity.
- A temporal change is NOT a deletion. Preserve the old node/edge and deactivate it,
  then add/update the new current fact if needed.
- If a previous extraction is clearly wrong, mark it `retracted`; do not erase it.
- If nothing in this phase changes the graph, return {{"operations": []}}.
- Evidence ids must come only from phase_utterances.
- Do not include explanations outside the JSON.

INPUT
{json.dumps(input_obj, ensure_ascii=False, indent=2)}

Return ONLY valid JSON.
"""

