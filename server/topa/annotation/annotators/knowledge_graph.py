from __future__ import annotations

import hashlib
import json
import os
import os.path as osp
import re
import time
from collections import defaultdict
from copy import deepcopy
from difflib import get_close_matches
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import pandas as pd
from tqdm import tqdm

from .base import BaseAnnotator
from ..prompts.knowledge_graph_prompt import annotate_knowledge_graph_prompt
from ...utils import build_logger, parse_json, run_llm_query

logger = build_logger()


class KnowledgeGraphAnnotator(BaseAnnotator):
    """Longitudinal, transaction-based patient KG population.

    One LLM call is made at each completed macro-phase boundary. The LLM does not
    rewrite the graph. It proposes a small list of graph operations; Python
    validates and applies each operation independently.

    Supported operations:
      * ADD_NODE
      * UPDATE_NODE
      * ADD_EDGE
      * UPDATE_EDGE
      * DEACTIVATE_NODE
      * DEACTIVATE_EDGE
      * MERGE_NODES

    There is intentionally no semantic hard-delete operation. Facts that are no
    longer current are preserved with a longitudinal status such as historical,
    resolved, retracted, or uncertain.

    Longitudinal semantics:
      * phase k consumes the dialogue observed during that phase and updates G_k;
      * the resulting graph becomes the initializer for phase k+1;
      * session N starts from the final graph of session N-1 for the same patient.
    """

    FORMAT_VERSION = 3
    ALLOWED_STATUSES = {
        "active",
        "historical",
        "resolved",
        "retracted",
        "uncertain",
        "inactive",
    }
    INACTIVE_STATUSES = {"historical", "resolved", "retracted", "inactive"}
    ALLOWED_PROVENANCE = {"explicit", "supported_inference"}
    OP_TYPES = {
        "ADD_NODE",
        "UPDATE_NODE",
        "ADD_EDGE",
        "UPDATE_EDGE",
        "DEACTIVATE_NODE",
        "DEACTIVATE_EDGE",
        "MERGE_NODES",
    }

    def __init__(self, **kwargs):
        _kwargs = deepcopy(kwargs)
        n_procs = int(_kwargs.get("num_processes", 1) or 1)
        super().__init__(**_kwargs)
        if n_procs > 1:
            logger.warning(
                "[KG] knowledge_graph is sequential within each patient; "
                "session-level num_processes=%s is ignored.",
                n_procs,
            )

        self.schema: Optional[dict] = None
        self.node_types: set[str] = set()
        self._type_casefold: Dict[str, str] = {}
        self.allowed_edges: set[Tuple[str, str, str]] = set()
        self._relations_by_pair: Dict[Tuple[str, str], List[str]] = defaultdict(list)

    # ------------------------------------------------------------------
    # BaseAnnotator API (this annotator uses a custom longitudinal runner)
    # ------------------------------------------------------------------
    def build_prompt(self, input_content: str) -> str:
        return input_content

    def sanity_check(self, outputs, data) -> bool:
        return True

    def postprocess(self, outputs, data):
        return outputs

    def fallback(self, last_outputs, data):
        return {"operations": []}

    def annotate_single_session(self, *args, **kwargs):
        raise RuntimeError(
            "KnowledgeGraphAnnotator uses its custom longitudinal annotate() method; "
            "annotate_single_session() must not be called directly."
        )

    # ------------------------------------------------------------------
    # Schema handling
    # ------------------------------------------------------------------
    def _canonical_type(self, value: Any) -> Optional[str]:
        if not isinstance(value, str):
            return None
        value = value.strip()
        if not value:
            return None
        if value in self.node_types:
            return value
        hit = self._type_casefold.get(value.casefold())
        if hit is not None:
            return hit
        matches = get_close_matches(value, sorted(self.node_types), n=1, cutoff=0.84)
        return matches[0] if matches else None

    def _canonical_relation(self, source_type: str, relation: Any, target_type: str) -> Optional[str]:
        allowed = self._relations_by_pair.get((source_type, target_type), [])
        if not allowed:
            return None

        if isinstance(relation, str):
            relation = relation.strip()
            if relation in allowed:
                return relation
            by_case = {r.casefold(): r for r in allowed}
            if relation.casefold() in by_case:
                return by_case[relation.casefold()]

        # The schema, not the LLM wording, is authoritative. When a source/target
        # pair permits exactly one relation, use it directly.
        if len(allowed) == 1:
            return allowed[0]

        if isinstance(relation, str) and relation.strip():
            matches = get_close_matches(relation.strip(), allowed, n=1, cutoff=0.72)
            if matches:
                return matches[0]
        return None

    def _prepare_schema(self, schema: dict) -> dict:
        if (
            not isinstance(schema, dict)
            or not isinstance(schema.get("nodes"), list)
            or not isinstance(schema.get("edges"), list)
        ):
            raise ValueError("knowledge_graph.json must contain top-level 'nodes' and 'edges' lists")

        clean = deepcopy(schema)
        self.node_types = {
            str(n["type"]).strip()
            for n in clean["nodes"]
            if isinstance(n, dict) and isinstance(n.get("type"), str) and n["type"].strip()
        }
        if "Patient" not in self.node_types:
            raise ValueError("knowledge_graph.json must define a 'Patient' node type")
        self._type_casefold = {x.casefold(): x for x in self.node_types}

        def schema_type(name: str) -> str:
            canonical = self._canonical_type(name)
            if canonical is None:
                raise ValueError(f"Unknown KG schema node type in edge: {name!r}")
            if canonical != name:
                logger.info("[KG] Canonicalized schema endpoint %r -> %r", name, canonical)
            return canonical

        self.allowed_edges = set()
        self._relations_by_pair = defaultdict(list)
        for edge in clean["edges"]:
            if not isinstance(edge, dict):
                raise ValueError(f"Invalid KG edge specification: {edge!r}")
            source = schema_type(str(edge.get("source", "")).strip())
            target = schema_type(str(edge.get("target", "")).strip())
            relation = str(edge.get("relation", "")).strip()
            if not relation:
                raise ValueError(f"KG schema edge has empty relation: {edge!r}")

            edge["source"] = source
            edge["target"] = target
            edge["relation"] = relation
            self.allowed_edges.add((source, relation, target))
            pair = (source, target)
            if relation not in self._relations_by_pair[pair]:
                self._relations_by_pair[pair].append(relation)

        self.schema = clean
        return clean

    def _schema_for_prompt(self) -> dict:
        assert self.schema is not None
        return {
            "nodes": [
                {"type": n.get("type"), "description": n.get("description", "")}
                for n in self.schema["nodes"]
            ],
            "edges": [
                {"source": e["source"], "relation": e["relation"], "target": e["target"]}
                for e in self.schema["edges"]
            ],
        }

    # ------------------------------------------------------------------
    # Parsing / operation sanitation
    # ------------------------------------------------------------------
    @staticmethod
    def _extract_json_fallback(raw: str):
        text = (raw or "").strip()
        if not text:
            return None
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s*```$", "", text)
        starts = [i for i in (text.find("{"), text.find("[")) if i >= 0]
        if not starts:
            return None
        start = min(starts)
        for end_char in ("}", "]"):
            end = text.rfind(end_char)
            if end > start:
                try:
                    return json.loads(text[start : end + 1])
                except Exception:
                    pass
        return None

    def _parse_llm_output(self, raw: str):
        try:
            return parse_json(raw)
        except Exception:
            return self._extract_json_fallback(raw)

    @staticmethod
    def _clip_confidence(value: Any, default: float = 1.0) -> float:
        try:
            value = float(value)
        except Exception:
            return float(default)
        return max(0.0, min(1.0, value))

    @staticmethod
    def _first(d: dict, *keys, default=None):
        for key in keys:
            if key in d and d[key] is not None:
                return d[key]
        return default

    def _sanitize_status(self, value: Any, default: str = "active") -> str:
        if not isinstance(value, str):
            return default
        value = value.strip().casefold()
        return value if value in self.ALLOWED_STATUSES else default

    def _sanitize_provenance(self, value: Any, default: str = "explicit") -> str:
        if not isinstance(value, str):
            return default
        value = value.strip().casefold()
        return value if value in self.ALLOWED_PROVENANCE else default

    @staticmethod
    def _sanitize_evidence(value: Any, valid_ids: set[str]) -> List[str]:
        if isinstance(value, str):
            value = [value]
        if not isinstance(value, list):
            return []
        out = []
        seen = set()
        for x in value:
            if isinstance(x, str) and x in valid_ids and x not in seen:
                out.append(x)
                seen.add(x)
        return out

    def _sanitize_operations(
        self,
        outputs: Any,
        valid_utterance_ids: Iterable[str],
    ) -> Tuple[List[dict], List[dict]]:
        """Normalize operation records without failing the whole phase.

        Structural/schema-dependent validation that needs the current graph is
        performed by _apply_operations. Here we only normalize the requested
        operation shape and evidence ids.
        """
        if isinstance(outputs, list):
            raw_ops = outputs
        elif isinstance(outputs, dict):
            raw_ops = outputs.get("operations", [])
        else:
            raw_ops = []
        if not isinstance(raw_ops, list):
            raw_ops = []

        valid_ids = set(valid_utterance_ids)
        accepted: List[dict] = []
        rejected: List[dict] = []

        for idx, raw in enumerate(raw_ops):
            if not isinstance(raw, dict):
                rejected.append({"index": idx, "reason": "not_an_object", "raw": raw})
                continue

            op_name = self._first(raw, "op", "operation", "type")
            if not isinstance(op_name, str):
                rejected.append({"index": idx, "reason": "missing_operation", "raw": raw})
                continue
            op_name = op_name.strip().upper()
            if op_name not in self.OP_TYPES:
                rejected.append({"index": idx, "reason": "unknown_operation", "raw": raw})
                continue

            op = deepcopy(raw)
            op["op"] = op_name
            op.pop("operation", None)
            if op.get("type") == op_name:
                op.pop("type", None)

            evidence = self._first(
                raw,
                "evidence_utterance_ids",
                "evidence",
                "utterance_ids",
                default=[],
            )
            op["evidence_utterance_ids"] = self._sanitize_evidence(evidence, valid_ids)

            if "confidence" in op:
                op["confidence"] = self._clip_confidence(op["confidence"])
            if "provenance_kind" in op:
                op["provenance_kind"] = self._sanitize_provenance(op["provenance_kind"])
            if "status" in op:
                op["status"] = self._sanitize_status(op["status"])
            if isinstance(op.get("changes"), dict):
                changes = deepcopy(op["changes"])
                if "confidence" in changes:
                    changes["confidence"] = self._clip_confidence(changes["confidence"])
                if "status" in changes:
                    changes["status"] = self._sanitize_status(changes["status"])
                if "provenance_kind" in changes:
                    changes["provenance_kind"] = self._sanitize_provenance(changes["provenance_kind"])
                op["changes"] = changes

            accepted.append(op)

        return accepted, rejected

    # ------------------------------------------------------------------
    # Graph state helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _new_graph(user_idx: int) -> dict:
        patient_id = f"patient_{user_idx}"
        return {
            "format_version": KnowledgeGraphAnnotator.FORMAT_VERSION,
            "user_idx": int(user_idx),
            "patient_node_id": patient_id,
            "nodes": [
                {
                    "id": patient_id,
                    "type": "Patient",
                    "content": "Patient",
                    "status": "active",
                    "confidence": 1.0,
                    "provenance_kind": "explicit",
                    "evidence": [],
                    "first_seen": None,
                    "last_updated": None,
                    "history": [],
                }
            ],
            "edges": [],
        }

    @staticmethod
    def _compact_graph_for_prompt(graph: dict) -> dict:
        return {
            "nodes": [
                {
                    "id": n.get("id"),
                    "type": n.get("type"),
                    "content": n.get("content"),
                    "status": n.get("status", "active"),
                    "confidence": n.get("confidence", 1.0),
                }
                for n in graph.get("nodes", [])
            ],
            "edges": [
                {
                    "id": e.get("id"),
                    "source": e.get("source"),
                    "relation": e.get("relation"),
                    "target": e.get("target"),
                    "status": e.get("status", "active"),
                    "confidence": e.get("confidence", 1.0),
                }
                for e in graph.get("edges", [])
            ],
        }

    @staticmethod
    def _normalize_content(text: str) -> str:
        return re.sub(r"\s+", " ", text.strip().casefold())

    def _stable_node_id(self, user_idx: int, node_type: str, content: str) -> str:
        raw = f"{user_idx}|node|{node_type}|{self._normalize_content(content)}".encode("utf-8")
        return "kg_" + hashlib.sha1(raw).hexdigest()[:16]

    def _stable_edge_id(self, user_idx: int, source: str, relation: str, target: str) -> str:
        raw = f"{user_idx}|edge|{source}|{relation}|{target}".encode("utf-8")
        return "ke_" + hashlib.sha1(raw).hexdigest()[:16]

    @staticmethod
    def _merge_unique(old: Iterable[str], new: Iterable[str]) -> List[str]:
        out, seen = [], set()
        for x in list(old) + list(new):
            if x not in seen:
                seen.add(x)
                out.append(x)
        return out

    @staticmethod
    def _boundary(session_idx: int, phase_idx: int) -> dict:
        return {"session_idx": int(session_idx), "phase_idx": int(phase_idx)}

    @staticmethod
    def _qualify_evidence(session_idx: int, phase_idx: int, evidence_ids: Iterable[str]) -> List[str]:
        return [f"s{session_idx}:p{phase_idx}:{x}" for x in evidence_ids]

    @staticmethod
    def _append_history(record: dict, boundary: dict, event: str, before: dict, reason: Optional[str] = None) -> None:
        item = {"at": deepcopy(boundary), "event": event, "before": before}
        if reason:
            item["reason"] = reason
        record.setdefault("history", []).append(item)

    def _find_existing_node(self, graph: dict, node_type: str, content: str) -> Optional[dict]:
        wanted = self._normalize_content(content)
        for node in graph.get("nodes", []):
            if node.get("type") == node_type and self._normalize_content(str(node.get("content", ""))) == wanted:
                return node
        return None

    @staticmethod
    def _node_index(graph: dict) -> Dict[str, dict]:
        return {n["id"]: n for n in graph.get("nodes", []) if isinstance(n, dict) and n.get("id")}

    @staticmethod
    def _edge_index(graph: dict) -> Dict[str, dict]:
        return {e["id"]: e for e in graph.get("edges", []) if isinstance(e, dict) and e.get("id")}

    @staticmethod
    def _edge_key_index(graph: dict) -> Dict[Tuple[str, str, str], dict]:
        return {
            (e.get("source"), e.get("relation"), e.get("target")): e
            for e in graph.get("edges", [])
            if isinstance(e, dict)
        }

    def _resolve_ref(self, ref: Any, temp_map: Dict[str, str], node_by_id: Dict[str, dict]) -> Optional[str]:
        if not isinstance(ref, str) or not ref.strip():
            return None
        ref = ref.strip()
        if ref in temp_map:
            ref = temp_map[ref]
        return ref if ref in node_by_id else None

    def _validate_edge_schema(self, node_by_id: Dict[str, dict], source_id: str, relation: Any, target_id: str) -> Optional[str]:
        source_type = node_by_id[source_id].get("type")
        target_type = node_by_id[target_id].get("type")
        return self._canonical_relation(source_type, relation, target_type)

    def _merge_duplicate_edges(self, graph: dict, user_idx: int, boundary: dict, stats: dict) -> None:
        """Collapse duplicate edges after endpoint redirection during node merge."""
        merged: Dict[Tuple[str, str, str], dict] = {}
        out: List[dict] = []
        for edge in graph.get("edges", []):
            key = (edge.get("source"), edge.get("relation"), edge.get("target"))
            if key not in merged:
                edge["id"] = self._stable_edge_id(user_idx, *key)
                merged[key] = edge
                out.append(edge)
                continue
            keep = merged[key]
            keep["evidence"] = self._merge_unique(keep.get("evidence", []), edge.get("evidence", []))
            keep["confidence"] = max(
                self._clip_confidence(keep.get("confidence", 1.0)),
                self._clip_confidence(edge.get("confidence", 1.0)),
            )
            keep.setdefault("history", []).extend(edge.get("history", []))
            keep["history"].append({
                "at": deepcopy(boundary),
                "event": "MERGED_DUPLICATE_EDGE",
                "merged_edge_id": edge.get("id"),
            })
            keep["last_updated"] = deepcopy(boundary)
            stats["edges_merged"] += 1
        graph["edges"] = out

    # ------------------------------------------------------------------
    # Transaction application
    # ------------------------------------------------------------------
    def _apply_operations(
        self,
        graph: dict,
        operations: List[dict],
        *,
        user_idx: int,
        session_idx: int,
        phase_idx: int,
        pre_rejected: Optional[List[dict]] = None,
    ) -> Tuple[dict, dict, List[dict]]:
        graph = deepcopy(graph)
        graph["format_version"] = self.FORMAT_VERSION
        boundary = self._boundary(session_idx, phase_idx)
        qualified_for = lambda ev: self._qualify_evidence(session_idx, phase_idx, ev)

        stats = {
            "operations_proposed": len(operations),
            "operations_applied": 0,
            "operations_rejected": len(pre_rejected or []),
            "nodes_added": 0,
            "nodes_reused": 0,
            "nodes_updated": 0,
            "nodes_deactivated": 0,
            "nodes_merged": 0,
            "edges_added": 0,
            "edges_reused": 0,
            "edges_updated": 0,
            "edges_deactivated": 0,
            "edges_merged": 0,
        }
        rejected: List[dict] = list(pre_rejected or [])
        temp_map: Dict[str, str] = {}

        node_by_id = self._node_index(graph)
        edge_by_id = self._edge_index(graph)
        edge_by_key = self._edge_key_index(graph)

        def reject(op_index: int, op: dict, reason: str, detail: Optional[str] = None):
            item = {"index": op_index, "reason": reason, "op": op}
            if detail:
                item["detail"] = detail
            rejected.append(item)
            stats["operations_rejected"] += 1

        # ---- Pass 1: ADD_NODE. This establishes all temp ids first. ----
        for i, op in enumerate(operations):
            if op.get("op") != "ADD_NODE":
                continue

            node_type = self._canonical_type(self._first(op, "node_type", "type"))
            content = self._first(op, "content", "text", "value")
            temp_id = self._first(op, "temp_id", "ref", "alias")
            if node_type is None:
                reject(i, op, "unknown_node_type")
                continue
            if node_type == "Patient":
                node_id = graph["patient_node_id"]
                if isinstance(temp_id, str) and temp_id.strip():
                    temp_map[temp_id.strip()] = node_id
                stats["nodes_reused"] += 1
                stats["operations_applied"] += 1
                continue
            if not isinstance(content, str) or not content.strip():
                reject(i, op, "missing_node_content")
                continue

            existing = self._find_existing_node(graph, node_type, content)
            evidence = qualified_for(op.get("evidence_utterance_ids", []))
            if existing is not None:
                existing["evidence"] = self._merge_unique(existing.get("evidence", []), evidence)
                existing["last_updated"] = deepcopy(boundary)
                existing["confidence"] = max(
                    self._clip_confidence(existing.get("confidence", 1.0)),
                    self._clip_confidence(op.get("confidence", 1.0)),
                )
                node_id = existing["id"]
                stats["nodes_reused"] += 1
            else:
                node_id = self._stable_node_id(user_idx, node_type, content)
                # Collision defense for extremely unlikely hash collision / legacy id.
                if node_id in node_by_id and (
                    node_by_id[node_id].get("type") != node_type
                    or self._normalize_content(str(node_by_id[node_id].get("content", ""))) != self._normalize_content(content)
                ):
                    suffix = hashlib.sha1(f"{time.time_ns()}|{content}".encode("utf-8")).hexdigest()[:6]
                    node_id = f"{node_id}_{suffix}"

                node = {
                    "id": node_id,
                    "type": node_type,
                    "content": content.strip(),
                    "status": self._sanitize_status(op.get("status", "active")),
                    "confidence": self._clip_confidence(op.get("confidence", 1.0)),
                    "provenance_kind": self._sanitize_provenance(op.get("provenance_kind", "explicit")),
                    "evidence": evidence,
                    "first_seen": deepcopy(boundary),
                    "last_updated": deepcopy(boundary),
                    "history": [],
                }
                graph["nodes"].append(node)
                node_by_id[node_id] = node
                stats["nodes_added"] += 1

            if isinstance(temp_id, str) and temp_id.strip():
                temp_map[temp_id.strip()] = node_id
            stats["operations_applied"] += 1

        # ---- Pass 2: UPDATE_NODE / DEACTIVATE_NODE ----
        for i, op in enumerate(operations):
            op_name = op.get("op")
            if op_name not in {"UPDATE_NODE", "DEACTIVATE_NODE"}:
                continue

            ref = self._first(op, "node_id", "id", "node", "ref")
            node_id = self._resolve_ref(ref, temp_map, node_by_id)
            if node_id is None:
                reject(i, op, "unknown_node_id")
                continue
            if node_id == graph["patient_node_id"] and op_name == "DEACTIVATE_NODE":
                reject(i, op, "cannot_deactivate_patient_node")
                continue

            node = node_by_id[node_id]
            evidence = qualified_for(op.get("evidence_utterance_ids", []))

            if op_name == "UPDATE_NODE":
                changes = op.get("changes", {})
                if not isinstance(changes, dict):
                    reject(i, op, "changes_must_be_object")
                    continue
                allowed = {"content", "status", "confidence", "provenance_kind"}
                changes = {k: v for k, v in changes.items() if k in allowed}
                if not changes and not evidence:
                    reject(i, op, "empty_node_update")
                    continue

                before = {k: deepcopy(node.get(k)) for k in changes}
                if "content" in changes:
                    if not isinstance(changes["content"], str) or not changes["content"].strip():
                        reject(i, op, "invalid_updated_content")
                        continue
                    changes["content"] = changes["content"].strip()
                if "status" in changes:
                    changes["status"] = self._sanitize_status(changes["status"], node.get("status", "active"))
                if "confidence" in changes:
                    changes["confidence"] = self._clip_confidence(changes["confidence"], node.get("confidence", 1.0))
                if "provenance_kind" in changes:
                    changes["provenance_kind"] = self._sanitize_provenance(
                        changes["provenance_kind"], node.get("provenance_kind", "explicit")
                    )

                if changes:
                    self._append_history(node, boundary, "UPDATE_NODE", before, op.get("reason"))
                    node.update(changes)
                node["evidence"] = self._merge_unique(node.get("evidence", []), evidence)
                node["last_updated"] = deepcopy(boundary)
                stats["nodes_updated"] += 1
                stats["operations_applied"] += 1
                continue

            # DEACTIVATE_NODE
            new_status = self._sanitize_status(op.get("status", "historical"), "historical")
            if new_status == "active":
                new_status = "historical"
            before = {"status": node.get("status", "active")}
            self._append_history(node, boundary, "DEACTIVATE_NODE", before, op.get("reason"))
            node["status"] = new_status
            node["evidence"] = self._merge_unique(node.get("evidence", []), evidence)
            node["last_updated"] = deepcopy(boundary)
            stats["nodes_deactivated"] += 1
            stats["operations_applied"] += 1

            # Keep the current-view graph internally consistent: incident active
            # edges inherit the node's non-active status. Their history is retained.
            for edge in graph.get("edges", []):
                if edge.get("source") == node_id or edge.get("target") == node_id:
                    if edge.get("status", "active") == "active":
                        self._append_history(
                            edge,
                            boundary,
                            "CASCADE_FROM_NODE_DEACTIVATION",
                            {"status": "active"},
                            f"node {node_id} became {new_status}",
                        )
                        edge["status"] = new_status
                        edge["last_updated"] = deepcopy(boundary)
                        stats["edges_deactivated"] += 1

        # ---- Pass 3: MERGE_NODES ----
        for i, op in enumerate(operations):
            if op.get("op") != "MERGE_NODES":
                continue

            keep_ref = self._first(op, "keep_node_id", "keep", "target_node_id")
            merge_refs = self._first(op, "merge_node_ids", "merge", "source_node_ids", default=[])
            if isinstance(merge_refs, str):
                merge_refs = [merge_refs]
            if not isinstance(merge_refs, list) or not merge_refs:
                reject(i, op, "missing_merge_node_ids")
                continue

            keep_id = self._resolve_ref(keep_ref, temp_map, node_by_id)
            if keep_id is None:
                reject(i, op, "unknown_keep_node_id")
                continue
            if keep_id == graph["patient_node_id"]:
                reject(i, op, "cannot_merge_into_patient_node")
                continue

            merge_ids = []
            bad_ref = False
            for ref in merge_refs:
                mid = self._resolve_ref(ref, temp_map, node_by_id)
                if mid is None:
                    bad_ref = True
                    break
                if mid != keep_id and mid not in merge_ids:
                    merge_ids.append(mid)
            if bad_ref:
                reject(i, op, "unknown_merge_node_id")
                continue
            if not merge_ids:
                reject(i, op, "nothing_to_merge")
                continue

            keep = node_by_id[keep_id]
            if any(node_by_id[mid].get("type") != keep.get("type") for mid in merge_ids):
                reject(i, op, "merge_requires_same_node_type")
                continue
            if graph["patient_node_id"] in merge_ids:
                reject(i, op, "cannot_merge_patient_node")
                continue

            merged_snapshots = []
            for mid in merge_ids:
                other = node_by_id[mid]
                merged_snapshots.append({
                    "id": mid,
                    "content": other.get("content"),
                    "status": other.get("status", "active"),
                    "confidence": other.get("confidence", 1.0),
                })
                keep["evidence"] = self._merge_unique(keep.get("evidence", []), other.get("evidence", []))
                keep["confidence"] = max(
                    self._clip_confidence(keep.get("confidence", 1.0)),
                    self._clip_confidence(other.get("confidence", 1.0)),
                )
                keep.setdefault("history", []).extend(other.get("history", []))

                for edge in graph.get("edges", []):
                    if edge.get("source") == mid:
                        edge["source"] = keep_id
                    if edge.get("target") == mid:
                        edge["target"] = keep_id

                graph["nodes"] = [n for n in graph.get("nodes", []) if n.get("id") != mid]
                node_by_id.pop(mid, None)
                for alias, mapped in list(temp_map.items()):
                    if mapped == mid:
                        temp_map[alias] = keep_id
                stats["nodes_merged"] += 1

            keep.setdefault("history", []).append({
                "at": deepcopy(boundary),
                "event": "MERGE_NODES",
                "merged_nodes": merged_snapshots,
                **({"reason": op.get("reason")} if op.get("reason") else {}),
            })
            keep["last_updated"] = deepcopy(boundary)
            self._merge_duplicate_edges(graph, user_idx, boundary, stats)
            edge_by_id = self._edge_index(graph)
            edge_by_key = self._edge_key_index(graph)
            stats["operations_applied"] += 1

        # Refresh indexes after node operations / merges.
        node_by_id = self._node_index(graph)
        edge_by_id = self._edge_index(graph)
        edge_by_key = self._edge_key_index(graph)

        # ---- Pass 4: ADD_EDGE ----
        for i, op in enumerate(operations):
            if op.get("op") != "ADD_EDGE":
                continue

            source_ref = self._first(op, "source", "source_id", "source_node_id")
            target_ref = self._first(op, "target", "target_id", "target_node_id")
            source_id = self._resolve_ref(source_ref, temp_map, node_by_id)
            target_id = self._resolve_ref(target_ref, temp_map, node_by_id)
            if source_id is None or target_id is None:
                reject(i, op, "unknown_edge_endpoint")
                continue

            relation = self._validate_edge_schema(node_by_id, source_id, op.get("relation"), target_id)
            if relation is None:
                reject(
                    i,
                    op,
                    "schema_edge_not_allowed",
                    f"{node_by_id[source_id].get('type')} -> {node_by_id[target_id].get('type')}",
                )
                continue

            key = (source_id, relation, target_id)
            evidence = qualified_for(op.get("evidence_utterance_ids", []))
            if key in edge_by_key:
                edge = edge_by_key[key]
                edge["evidence"] = self._merge_unique(edge.get("evidence", []), evidence)
                edge["confidence"] = max(
                    self._clip_confidence(edge.get("confidence", 1.0)),
                    self._clip_confidence(op.get("confidence", 1.0)),
                )
                edge["last_updated"] = deepcopy(boundary)
                stats["edges_reused"] += 1
                stats["operations_applied"] += 1
                continue

            edge_id = self._stable_edge_id(user_idx, source_id, relation, target_id)
            edge = {
                "id": edge_id,
                "source": source_id,
                "relation": relation,
                "target": target_id,
                "status": self._sanitize_status(op.get("status", "active")),
                "confidence": self._clip_confidence(op.get("confidence", 1.0)),
                "evidence": evidence,
                "first_seen": deepcopy(boundary),
                "last_updated": deepcopy(boundary),
                "history": [],
            }
            graph["edges"].append(edge)
            edge_by_id[edge_id] = edge
            edge_by_key[key] = edge
            stats["edges_added"] += 1
            stats["operations_applied"] += 1

        # ---- Pass 5: UPDATE_EDGE / DEACTIVATE_EDGE ----
        edge_by_id = self._edge_index(graph)
        for i, op in enumerate(operations):
            op_name = op.get("op")
            if op_name not in {"UPDATE_EDGE", "DEACTIVATE_EDGE"}:
                continue

            edge_ref = self._first(op, "edge_id", "id", "edge")
            if not isinstance(edge_ref, str) or edge_ref not in edge_by_id:
                reject(i, op, "unknown_edge_id")
                continue
            edge = edge_by_id[edge_ref]
            evidence = qualified_for(op.get("evidence_utterance_ids", []))

            if op_name == "UPDATE_EDGE":
                changes = op.get("changes", {})
                if not isinstance(changes, dict):
                    reject(i, op, "changes_must_be_object")
                    continue
                # Topology changes are deliberately disallowed here. If the
                # relationship itself changes, deactivate the old edge and add a new one.
                allowed = {"status", "confidence"}
                changes = {k: v for k, v in changes.items() if k in allowed}
                if not changes and not evidence:
                    reject(i, op, "empty_edge_update")
                    continue
                before = {k: deepcopy(edge.get(k)) for k in changes}
                if "status" in changes:
                    changes["status"] = self._sanitize_status(changes["status"], edge.get("status", "active"))
                if "confidence" in changes:
                    changes["confidence"] = self._clip_confidence(changes["confidence"], edge.get("confidence", 1.0))
                if changes:
                    self._append_history(edge, boundary, "UPDATE_EDGE", before, op.get("reason"))
                    edge.update(changes)
                edge["evidence"] = self._merge_unique(edge.get("evidence", []), evidence)
                edge["last_updated"] = deepcopy(boundary)
                stats["edges_updated"] += 1
                stats["operations_applied"] += 1
                continue

            new_status = self._sanitize_status(op.get("status", "historical"), "historical")
            if new_status == "active":
                new_status = "historical"
            self._append_history(
                edge,
                boundary,
                "DEACTIVATE_EDGE",
                {"status": edge.get("status", "active")},
                op.get("reason"),
            )
            edge["status"] = new_status
            edge["evidence"] = self._merge_unique(edge.get("evidence", []), evidence)
            edge["last_updated"] = deepcopy(boundary)
            stats["edges_deactivated"] += 1
            stats["operations_applied"] += 1

        return graph, stats, rejected

    # ------------------------------------------------------------------
    # Phase construction
    # ------------------------------------------------------------------
    @staticmethod
    def _macro_annotation_path(annotation_path: str) -> str:
        candidates = [
            osp.join(annotation_path, "annotations_macro_actions.csv"),
            osp.join(annotation_path, "annotations_macro_actions_no_rules.csv"),
        ]
        for p in candidates:
            if osp.exists(p):
                return p
        raise FileNotFoundError(
            "Knowledge-graph population requires macro-action annotations first. "
            f"None of these files exists: {candidates}"
        )

    def _load_macro_annotations(self, annotation_path: str) -> pd.DataFrame:
        path = self._macro_annotation_path(annotation_path)
        df = pd.read_csv(path, encoding="utf-8")
        required = {
            "user_idx",
            "session_idx",
            "phase_idx",
            "from_id",
            "to_id",
            "from_idx",
            "to_idx",
            "selected_macro_action",
        }
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"Macro annotation file is missing columns: {sorted(missing)}")
        for c in ("user_idx", "session_idx", "phase_idx", "from_idx", "to_idx"):
            df[c] = df[c].astype(int)
        return df.sort_values(["user_idx", "session_idx", "phase_idx"]).reset_index(drop=True)

    def _phase_payloads(
        self,
        user_idx: int,
        session_idx: int,
        dialogue: List[Dict],
        macro_source,
    ) -> List[dict]:
        # macro_source can be either the original DataFrame or a pre-indexed
        # {(user_idx, session_idx): [phase_rows...]} mapping. The latter avoids
        # rescanning the full macro-action table for every session.
        if isinstance(macro_source, dict):
            rows_list = macro_source.get((int(user_idx), int(session_idx)), [])
        else:
            rows = macro_source[
                (macro_source["user_idx"] == user_idx)
                & (macro_source["session_idx"] == session_idx)
            ].sort_values("phase_idx")
            rows_list = rows.to_dict("records")

        if not rows_list:
            raise ValueError(f"No macro-action phases for user={user_idx}, session={session_idx}")

        utterances, _, _ = self.process_dialogue(dialogue)
        payloads = []

        # Consume all dialogue observed until the next planning boundary. This
        # includes the user's response after the final system turn of the phase.
        start = 0
        for i, row in enumerate(rows_list):
            if i + 1 < len(rows_list):
                end = int(rows_list[i + 1]["from_idx"]) - 1
            else:
                end = len(utterances) - 1
            end = max(end, start - 1)
            phase_utts = utterances[start : end + 1] if end >= start else []
            payloads.append(
                {
                    "phase_idx": int(row["phase_idx"]),
                    "from_id": row["from_id"],
                    "to_id": row["to_id"],
                    "selected_macro_action": row["selected_macro_action"],
                    "utterances": phase_utts,
                }
            )
            start = end + 1

        return payloads

    # ------------------------------------------------------------------
    # Persistence / resume
    # ------------------------------------------------------------------
    @staticmethod
    def _kg_root(annotation_path: str) -> Path:
        return Path(annotation_path) / "knowledge_graph_states"

    @classmethod
    def _phase_snapshot_path(
        cls, annotation_path: str, user_idx: int, session_idx: int, phase_idx: int
    ) -> Path:
        return (
            cls._kg_root(annotation_path)
            / f"user_{user_idx}"
            / f"session_{session_idx}"
            / f"phase_{phase_idx}.json"
        )

    @classmethod
    def _phase_delta_path(
        cls, annotation_path: str, user_idx: int, session_idx: int, phase_idx: int
    ) -> Path:
        return (
            cls._kg_root(annotation_path)
            / f"user_{user_idx}"
            / f"session_{session_idx}"
            / f"phase_{phase_idx}_delta.json"
        )

    @classmethod
    def _session_final_path(cls, annotation_path: str, user_idx: int, session_idx: int) -> Path:
        return cls._kg_root(annotation_path) / f"user_{user_idx}" / f"session_{session_idx}" / "final.json"

    @classmethod
    def _user_latest_path(cls, annotation_path: str, user_idx: int) -> Path:
        return cls._kg_root(annotation_path) / f"user_{user_idx}" / "latest.json"

    @staticmethod
    def _load_json_if_exists(path: Path) -> Optional[dict]:
        if not path.exists():
            return None
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)

    @staticmethod
    def _write_json(path: Path, obj: dict) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        with tmp.open("w", encoding="utf-8") as f:
            json.dump(obj, f, ensure_ascii=False, indent=2)
        os.replace(tmp, path)

    @staticmethod
    def _index_path(annotation_path: str) -> Path:
        return Path(annotation_path) / "annotations_knowledge_graph.csv"

    @staticmethod
    def _usage_path(annotation_path: str) -> Path:
        return Path(annotation_path) / "annotations_knowledge_graph_usage.json"

    def _load_index(self, annotation_path: str) -> pd.DataFrame:
        path = self._index_path(annotation_path)
        if not path.exists() or path.stat().st_size == 0:
            return pd.DataFrame()
        return pd.read_csv(path, encoding="utf-8")

    def _save_index(self, annotation_path: str, rows: List[dict]) -> str:
        path = self._index_path(annotation_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        df = pd.DataFrame(rows)
        if not df.empty:
            df = df.sort_values(["user_idx", "session_idx", "phase_idx"]).drop_duplicates(
                ["user_idx", "session_idx", "phase_idx"], keep="last"
            )
        df.to_csv(path, index=False, encoding="utf-8")
        return str(path)

    # ------------------------------------------------------------------
    # One phase annotation with JSON-parse retries
    # ------------------------------------------------------------------
    def annotate_single_phase(
        self,
        *,
        user_idx: int,
        session_idx: int,
        phase: dict,
        current_graph: dict,
    ) -> Tuple[dict, dict]:
        valid_ids = [u["utterance_id"] for u in phase["utterances"]]
        if not phase["utterances"]:
            return {"operations": [], "rejected": []}, {"inp_t": 0.0, "out_t": 0.0}

        phase_meta = {
            "session_idx": session_idx,
            "phase_idx": phase["phase_idx"],
            "selected_macro_action": phase["selected_macro_action"],
            "from_id": phase["from_id"],
            "to_id": phase["to_id"],
        }
        prompt = annotate_knowledge_graph_prompt(
            domain=self.domain_params["adj"],
            user=self.domain_params["user"],
            system=self.domain_params["system"],
            interaction_unit=self.domain_params["interaction_unit"],
            schema=self._schema_for_prompt(),
            current_graph=self._compact_graph_for_prompt(current_graph),
            phase=phase_meta,
            phase_utterances=phase["utterances"],
        )

        usage = {"inp_t": 0.0, "out_t": 0.0}

        # Initial generation + 6 additional generations only when the returned
        # text cannot be parsed as JSON. run_llm_query() still owns transport/API
        # retries (429, transient service errors, etc.) for each generation.
        json_parse_retries = 6
        parsed = None
        last_raw = None

        for parse_attempt in range(json_parse_retries + 1):
            try:
                raw, inp_t, out_t = run_llm_query(prompt, self.api_key)
                usage["inp_t"] += inp_t / 1_000_000
                usage["out_t"] += out_t / 1_000_000
            except Exception as e:
                # IMPORTANT: do not convert a failed API call into an empty
                # successful transaction. The phase must remain unfinished.
                logger.error(
                    "[KG] LLM call failed for user=%s session=%s phase=%s "
                    "during JSON generation attempt %s/%s: %s. "
                    "Phase remains unfinished and will be retried on the next run.",
                    user_idx,
                    session_idx,
                    phase["phase_idx"],
                    parse_attempt + 1,
                    json_parse_retries + 1,
                    e,
                )
                raise RuntimeError(
                    f"KG annotation LLM call failed for user={user_idx}, "
                    f"session={session_idx}, phase={phase['phase_idx']}: {e}"
                ) from e

            last_raw = raw
            parsed = self._parse_llm_output(raw)
            if parsed is not None:
                if parse_attempt > 0:
                    logger.info(
                        "[KG] JSON parse succeeded for user=%s session=%s phase=%s "
                        "after %s retry/retries.",
                        user_idx,
                        session_idx,
                        phase["phase_idx"],
                        parse_attempt,
                    )
                break

            if parse_attempt < json_parse_retries:
                logger.warning(
                    "[KG] Could not parse JSON for user=%s session=%s phase=%s "
                    "(generation %s/%s). Retrying generation (%s retry/retries left).",
                    user_idx,
                    session_idx,
                    phase["phase_idx"],
                    parse_attempt + 1,
                    json_parse_retries + 1,
                    json_parse_retries - parse_attempt,
                )

        if parsed is None:
            # All parse attempts failed. Keep the graph unchanged and continue.
            preview = (last_raw or "").strip().replace("\n", " ")[:1000]
            logger.warning(
                "[KG] JSON parse failed after initial generation + %s retries for "
                "user=%s session=%s phase=%s. Keeping the previous graph and "
                "continuing to the next phase. Last raw response preview: %r",
                json_parse_retries,
                user_idx,
                session_idx,
                phase["phase_idx"],
                preview,
            )
            return {"operations": [], "rejected": []}, usage

        operations, rejected = self._sanitize_operations(parsed, valid_ids)
        if rejected:
            logger.warning(
                "[KG] user=%s session=%s phase=%s: accepted %s operation record(s), "
                "dropped %s malformed record(s) before graph validation.",
                user_idx,
                session_idx,
                phase["phase_idx"],
                len(operations),
                len(rejected),
            )
        else:
            logger.info(
                "[KG] user=%s session=%s phase=%s: received %s operation(s).",
                user_idx,
                session_idx,
                phase["phase_idx"],
                len(operations),
            )

        return {"operations": operations, "rejected": rejected}, usage

    # ------------------------------------------------------------------
    # Longitudinal execution
    # ------------------------------------------------------------------
    def annotate(
        self, data, component, component_name, annotation_path,
        session_filter: set | None = None, *, session_ranges=None, macro_annotations=None,
    ):
        if component_name != "knowledge_graph":
            raise ValueError(
                f"KnowledgeGraphAnnotator expected component_name='knowledge_graph', got {component_name!r}"
            )
        if session_filter is not None:
            logger.warning(
                "[KG] session_filter is ignored because longitudinal KG carryover "
                "requires complete ordered sessions per patient."
            )

        self._prepare_schema(component)
        macro_df = (
            self._load_macro_annotations(annotation_path)
            if macro_annotations is None else macro_annotations
        )
        Path(annotation_path).mkdir(parents=True, exist_ok=True)

        # Pre-index macro phases once. This changes repeated O(num_macro_rows)
        # DataFrame filtering per session into O(1) dictionary lookup.
        macro_by_session: Dict[Tuple[int, int], List[dict]] = {}
        for (u, s), group in macro_df.groupby(["user_idx", "session_idx"], sort=False):
            macro_by_session[(int(u), int(s))] = (
                group.sort_values("phase_idx").to_dict("records")
            )
        logger.info(
            "[KG] Indexed %s macro-action phases across %s sessions.",
            len(macro_df),
            len(macro_by_session),
        )

        existing_index = self._load_index(annotation_path)
        index_rows = existing_index.to_dict("records") if not existing_index.empty else []

        # Fast completion discovery: annotations_knowledge_graph.csv already stores
        # the format version. At startup we only check that the snapshot exists;
        # loading every full KG snapshot here is unnecessarily expensive.
        completed_phases = set()
        for row in index_rows:
            try:
                if int(row.get("format_version", 0) or 0) != self.FORMAT_VERSION:
                    continue
                key = (
                    int(row["user_idx"]),
                    int(row["session_idx"]),
                    int(row["phase_idx"]),
                )
                if self._phase_snapshot_path(annotation_path, *key).exists():
                    completed_phases.add(key)
            except Exception:
                continue

        logger.info(
            "[KG] Loaded %s previous KG annotation rows; %s completed phase snapshots found.",
            len(index_rows),
            len(completed_phases),
        )

        if not existing_index.empty:
            versions = set()
            if "format_version" in existing_index.columns:
                for v in existing_index["format_version"].dropna().tolist():
                    try:
                        versions.add(int(v))
                    except Exception:
                        pass
            if versions and versions != {self.FORMAT_VERSION}:
                logger.warning(
                    "[KG] Existing KG annotations include older format version(s) %s. "
                    "They will be regenerated as transaction-based v%s snapshots.",
                    sorted(versions),
                    self.FORMAT_VERSION,
                )
            elif "format_version" not in existing_index.columns:
                logger.warning(
                    "[KG] Existing KG annotations have no format_version and will be regenerated."
                )

        usage = {"inp_t": 0.0, "out_t": 0.0}
        previous_runtime = 0.0
        usage_path = self._usage_path(annotation_path)
        if usage_path.exists():
            try:
                obj = json.loads(usage_path.read_text(encoding="utf-8"))
                usage["inp_t"] = float(obj.get("usage", {}).get("inp_t", 0.0) or 0.0)
                usage["out_t"] = float(obj.get("usage", {}).get("out_t", 0.0) or 0.0)
                previous_runtime = float(obj.get("runtime", 0.0) or 0.0)
            except Exception as e:
                logger.warning("[KG] Could not load previous usage file: %s", e)

        started = time.time()

        for user_idx, user_obj in enumerate(data):
            if session_ranges is not None and user_idx not in session_ranges:
                continue
            sessions = user_obj.get("sessions", [])
            current_graph = self._new_graph(user_idx)

            # Explicit ranges retain original dataset indices and start with a
            # fresh graph. The default longitudinal runner keeps its existing cap.
            if session_ranges is None:
                first_session, last_session = 0, min(10, len(sessions)) - 1
            else:
                first_session, last_session = session_ranges[user_idx]
                if not 0 <= first_session <= last_session < len(sessions):
                    raise ValueError(f"Invalid KG session range for user={user_idx}")
                current_graph["window_start_session_idx"] = first_session
                current_graph["window_target_session_idx"] = last_session

            for session_idx in tqdm(
                range(first_session, last_session + 1), desc=f"KG user {user_idx}"
            ):
                session_obj = sessions[session_idx]
                phases = self._phase_payloads(
                    user_idx=user_idx,
                    session_idx=session_idx,
                    dialogue=session_obj["dialogue"],
                    macro_source=macro_by_session,
                )

                phase_keys = [
                    (user_idx, session_idx, int(phase["phase_idx"]))
                    for phase in phases
                ]
                session_final = self._session_final_path(
                    annotation_path, user_idx, session_idx
                )

                # ----------------------------------------------------------
                # FAST PATH 1: entire session is already complete.
                # Load final.json once instead of every phase snapshot.
                # ----------------------------------------------------------
                session_complete = bool(phase_keys) and all(
                    key in completed_phases for key in phase_keys
                )
                if session_complete and session_final.exists():
                    loaded_final = self._load_json_if_exists(session_final)
                    if (
                        loaded_final is not None
                        and int(loaded_final.get("format_version", 0) or 0)
                        == self.FORMAT_VERSION
                    ):
                        current_graph = loaded_final
                        continue

                # ----------------------------------------------------------
                # Session initializer. Session 0 starts from an empty patient
                # graph; later sessions inherit session N-1 final.json.
                # ----------------------------------------------------------
                if session_idx != first_session:
                    prev_final = self._session_final_path(
                        annotation_path, user_idx, session_idx - 1
                    )
                    loaded_prev = self._load_json_if_exists(prev_final)
                    if (
                        loaded_prev is None
                        or int(loaded_prev.get("format_version", 0) or 0)
                        != self.FORMAT_VERSION
                    ):
                        raise RuntimeError(
                            f"Cannot annotate user={user_idx}, session={session_idx}: "
                            f"previous session final KG v{self.FORMAT_VERSION} is missing: {prev_final}"
                        )
                    current_graph = loaded_prev

                # ----------------------------------------------------------
                # FAST PATH 2: partially completed session. Completed phases
                # are expected to form a prefix. Restore only the LAST snapshot
                # in that prefix, then continue from the first unfinished phase.
                # This avoids loading phase_0.json, phase_1.json, ... one by one.
                # ----------------------------------------------------------
                resume_from = 0
                completed_prefix_len = 0
                for key in phase_keys:
                    if key not in completed_phases:
                        break
                    snap = self._phase_snapshot_path(annotation_path, *key)
                    if not snap.exists():
                        # Defensive repair: CSV says complete but file is absent.
                        completed_phases.discard(key)
                        break
                    completed_prefix_len += 1

                # Load at most ONE phase snapshot for a partially completed
                # session. If the newest prefix snapshot is corrupt, walk backward
                # until a valid checkpoint is found, then recompute from there.
                for checkpoint_idx in range(completed_prefix_len - 1, -1, -1):
                    checkpoint_key = phase_keys[checkpoint_idx]
                    checkpoint_path = self._phase_snapshot_path(
                        annotation_path, *checkpoint_key
                    )
                    try:
                        loaded = self._load_json_if_exists(checkpoint_path)
                    except Exception as e:
                        logger.warning(
                            "[KG] Could not load checkpoint %s: %s. "
                            "Will resume from an earlier checkpoint.",
                            checkpoint_path,
                            e,
                        )
                        loaded = None

                    if (
                        loaded is not None
                        and int(loaded.get("format_version", 0) or 0)
                        == self.FORMAT_VERSION
                    ):
                        current_graph = loaded
                        resume_from = checkpoint_idx + 1
                        break

                    # This checkpoint cannot be trusted as completed.
                    completed_phases.discard(checkpoint_key)

                for phase in phases[resume_from:]:
                    key = (user_idx, session_idx, int(phase["phase_idx"]))
                    snap_path = self._phase_snapshot_path(annotation_path, *key)

                    # Non-contiguous completed rows are unusual (normally phases
                    # are a prefix). Keep the old safe behavior for that case.
                    if key in completed_phases:
                        loaded = self._load_json_if_exists(snap_path)
                        if (
                            loaded is not None
                            and int(loaded.get("format_version", 0) or 0)
                            == self.FORMAT_VERSION
                        ):
                            current_graph = loaded
                            continue
                        completed_phases.discard(key)

                    result, phase_usage = self.annotate_single_phase(
                        user_idx=user_idx,
                        session_idx=session_idx,
                        phase=phase,
                        current_graph=current_graph,
                    )
                    usage["inp_t"] += phase_usage["inp_t"]
                    usage["out_t"] += phase_usage["out_t"]

                    current_graph, stats, rejected = self._apply_operations(
                        current_graph,
                        result["operations"],
                        user_idx=user_idx,
                        session_idx=session_idx,
                        phase_idx=int(phase["phase_idx"]),
                        pre_rejected=result.get("rejected", []),
                    )

                    if stats["operations_rejected"]:
                        logger.warning(
                            "[KG] user=%s session=%s phase=%s: applied %s/%s proposed operation(s); "
                            "%s record(s) rejected in total.",
                            user_idx,
                            session_idx,
                            phase["phase_idx"],
                            stats["operations_applied"],
                            stats["operations_proposed"],
                            stats["operations_rejected"],
                        )
                    else:
                        logger.info(
                            "[KG] user=%s session=%s phase=%s: applied %s operation(s).",
                            user_idx,
                            session_idx,
                            phase["phase_idx"],
                            stats["operations_applied"],
                        )

                    delta_path = self._phase_delta_path(annotation_path, *key)
                    self._write_json(
                        delta_path,
                        {
                            "format_version": self.FORMAT_VERSION,
                            "user_idx": user_idx,
                            "session_idx": session_idx,
                            "phase_idx": int(phase["phase_idx"]),
                            "from_id": phase["from_id"],
                            "to_id": phase["to_id"],
                            "selected_macro_action": phase["selected_macro_action"],
                            "operations": result["operations"],
                            "rejected": rejected,
                            "apply_stats": stats,
                        },
                    )
                    self._write_json(snap_path, current_graph)

                    row = {
                        "format_version": self.FORMAT_VERSION,
                        "user_idx": user_idx,
                        "session_idx": session_idx,
                        "phase_idx": int(phase["phase_idx"]),
                        "from_id": phase["from_id"],
                        "to_id": phase["to_id"],
                        "selected_macro_action": phase["selected_macro_action"],
                        "delta_path": str(delta_path),
                        "snapshot_path": str(snap_path),
                        "num_nodes": len(current_graph["nodes"]),
                        "num_edges": len(current_graph["edges"]),
                        **stats,
                    }
                    index_rows.append(row)
                    completed_phases.add(key)
                    self._save_index(annotation_path, index_rows)

                    usage_obj = {
                        "runtime": previous_runtime + (time.time() - started),
                        "usage": usage,
                    }
                    self._write_json(usage_path, usage_obj)

                # Save/refresh the session checkpoint only for sessions that were
                # incomplete when this run entered them. Fully completed sessions
                # take the fast path above and are not rewritten.
                self._write_json(session_final, current_graph)
                self._write_json(
                    self._user_latest_path(annotation_path, user_idx), current_graph
                )

        usage_obj = {
            "runtime": previous_runtime + (time.time() - started),
            "usage": usage,
        }
        self._write_json(usage_path, usage_obj)
        return self._save_index(annotation_path, index_rows)

