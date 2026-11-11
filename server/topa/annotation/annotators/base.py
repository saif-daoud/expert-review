from abc import ABC, abstractmethod
from typing import Any, Dict, List, Tuple
import time
import json
import os
import os.path as osp
import pandas as pd
from tqdm import tqdm
import multiprocessing as mp

from ...utils import build_logger, run_llm_query, parse_json

logger = build_logger()

# Multiprocessing helpers
_G_ANNOTATOR = None
_G_COMPONENT = None
_G_REQUIRED = None

def _mp_init_worker(annotator_obj, component, required_annotations):
    global _G_ANNOTATOR, _G_COMPONENT, _G_REQUIRED
    _G_ANNOTATOR = annotator_obj
    _G_COMPONENT = component
    _G_REQUIRED = required_annotations

def _mp_process_task_batch(task_batch):
    """Process a list of (user_idx, session_idx, dialogue) tasks in one worker."""
    anns = []
    usage = {"inp_t": 0.0, "out_t": 0.0}
    n_tasks = 0
    for (user_idx, session_idx, dialogue, session_metadata) in task_batch:
        session_annotations, u = _G_ANNOTATOR.annotate_single_session(
            user_idx, session_idx, dialogue, session_metadata, _G_COMPONENT, _G_REQUIRED
        )
        anns.extend(session_annotations)
        usage["inp_t"] += u["inp_t"]
        usage["out_t"] += u["out_t"]
        n_tasks += 1
    return anns, usage, n_tasks

def _estimate_task_weight(dialogue, component_name):
    """Assign weight to balance the tasks."""
    total_words = sum(len(turn["text"].split()) for turn in dialogue)
    if component_name == "macro_actions":
        return total_words
    else:
        return total_words + 20 * len(dialogue)

def _greedy_balance(tasks_with_w, n_bins):
    """Greedy bin packing to make workers finish at similar time."""
    bins = [[] for _ in range(n_bins)]
    bin_w = [0 for _ in range(n_bins)]
    # heavier first
    for task, w in sorted(tasks_with_w, key=lambda x: x[1], reverse=True):
        j = min(range(n_bins), key=lambda i: bin_w[i])
        bins[j].append(task)
        bin_w[j] += w
    # drop empties
    return [b for b in bins if b]


class BaseAnnotator(ABC):
    """
    Parent class that contains all shared logic for LLM-based annotators.
    Child classes override sanity checks, postprocessing and prompt building.
    """

    def __init__(self, domain: str, domain_params: Dict, api_key: str, max_retries: int = 3, inp_cost: float = 3.0, out_cost: float = 12.0, num_processes: int = 1):
        self.domain = domain
        self.domain_params = domain_params
        self.api_key = api_key
        self.max_retries = max_retries
        self.inp_cost = inp_cost
        self.out_cost = out_cost
        self.num_processes = num_processes

        # Default: some annotators expose/use annotation rules, others don't.
        # Avoid attribute errors in save_output.
        self.use_annotation_rules = False

    @abstractmethod
    def build_prompt(self, input_content: str) -> str:
        """Return final text prompt to give to the LLM."""
        pass

    @abstractmethod
    def sanity_check(self, outputs: List[Dict], data: Dict) -> bool:
        """Check whether the parsed LLM output is acceptable."""
        pass

    @abstractmethod
    def annotate_single_session(self, user_idx: int, session_idx: int, session: List[Dict], session_metadata: dict, 
                                component: Any, required_annotations: pd.DataFrame) -> List:
        """Annotate one session with the compenent."""
        pass

    @abstractmethod
    def fallback(self, last_outputs: List[Dict], data: Dict) -> List[Dict]:
        """
        If max retries fail, define how to fix or fill missing annotations.
        """
        pass

    def postprocess(self, outputs: List[Dict], data: Dict) -> List[Dict]:
        """Optional (default no-op)."""
        return outputs

    def process_dialogue(self, dialogue: List[Dict]) -> List[Dict]:
        """
        Process Dialogue for a single session
        """
        system = self.domain_params["system"]
        user = self.domain_params["user"]
        utterances = []
        system_counter = 0
        user_counter = 0

        for turn in dialogue:
            if turn["speaker"] == "system":
                utt_id = f"{system}_{system_counter}"
                system_counter += 1
                speaker = system
            elif turn["speaker"] == "user":
                utt_id = f"{user}_{user_counter}"
                user_counter += 1
                speaker = user
            else:
                raise Exception(f"Unkown speaker {turn['speaker']} - Utterance : {turn}")

            utt = {
                "utterance_id": utt_id,
                "speaker": speaker,
                "utterance": turn["text"],
            }
            utterances.append(utt)
        return utterances, system_counter, user_counter
    
    def has_valid_system_ids(self, utterance_ids: List[str], sys_pos: Dict[str, int], context: str = "system utterance ids") -> bool:
        """Return False when any predicted system utterance id does not exist in the session."""
        missing_ids = [utt_id for utt_id in utterance_ids if utt_id not in sys_pos]
        if missing_ids:
            logger.warning(f"[SANITY CHECK] Unknown {context}: {missing_ids}")
            return False
        return True

    def get_output_paths(self, output_path: str, component_name: str) -> Tuple[str, str]:
        """Resolve the canonical CSV + usage paths for one component."""
        if component_name in ["macro_actions", "micro_actions", "micro_actions_direct"]:
            os_csv = f"annotations_{component_name}{'_no_rules' if not self.use_annotation_rules else ''}.csv"
        else:
            os_csv = f"annotations_{component_name}.csv"
        os_usage = f"annotations_{component_name}_usage.json"
        return osp.join(output_path, os_csv), osp.join(output_path, os_usage)

    def load_existing_output(self, output_path: str, component_name: str) -> Tuple[List[Dict], set, Dict[str, float], float, str]:
        """Load existing annotations/usage so interrupted runs can resume."""
        csv_path, usage_path = self.get_output_paths(output_path, component_name)

        all_annotations = []
        completed_sessions = set()
        running_usage = {"inp_t": 0.0, "out_t": 0.0}
        previous_runtime = 0.0

        if osp.exists(csv_path) and osp.getsize(csv_path) > 0:
            existing_df = pd.read_csv(csv_path, encoding="utf-8")
            if not existing_df.empty:
                all_annotations = existing_df.to_dict("records")
                if {"user_idx", "session_idx"}.issubset(existing_df.columns):
                    pairs = (
                        existing_df[["user_idx", "session_idx"]]
                        .dropna()
                        .drop_duplicates()
                        .itertuples(index=False, name=None)
                    )
                    completed_sessions = {(int(user_idx), int(session_idx)) for user_idx, session_idx in pairs}

        if osp.exists(usage_path) and osp.getsize(usage_path) > 0:
            with open(usage_path, "r", encoding="utf-8") as f:
                cost_data = json.load(f)
            usage_data = cost_data.get("usage", {})
            running_usage["inp_t"] = float(usage_data.get("inp_t", 0.0) or 0.0)
            running_usage["out_t"] = float(usage_data.get("out_t", 0.0) or 0.0)
            previous_runtime = float(cost_data.get("runtime", 0.0) or 0.0)

        return all_annotations, completed_sessions, running_usage, previous_runtime, csv_path

    def build_cost_data(self, start_time: float, previous_runtime: float, usage: Dict[str, float]) -> Dict:
        """Accumulate runtime/usage when resuming from an existing annotation directory."""
        return {"runtime": previous_runtime + (time.time() - start_time), "usage": usage}

    def run_with_retries(self, prompt: str, data) -> Tuple[List[Dict], dict]:
        """Run LLM + JSON parsing + retries + sanity."""
        last_outputs = None
        usage = {"inp_t": 0.0, "out_t": 0.0}

        for attempt in range(self.max_retries):
            raw, inp_t, out_t = run_llm_query(prompt, self.api_key)

            # update cost
            usage["inp_t"] += (inp_t / 1_000_000)
            usage["out_t"] += (out_t / 1_000_000)
            # Parse JSON
            try:
                outputs = parse_json(raw)
            except Exception as e:
                try:
                    outputs = parse_json(raw.replace("'", '"'))
                except:
                    logger.warning(f"[WARNING] JSON parsing failed at attempt {attempt+1}, error: {e} retrying...")
                    continue
            
            outputs = self.postprocess(outputs, data)
            last_outputs = outputs
            if self.sanity_check(outputs, data):
                return outputs, usage
            logger.warning(f"[WARNING] Sanity check failed at attempt {attempt+1}, retrying...")

        logger.warning("[ERROR] Max retries reached. Using fallback.")
        fixed_outputs = self.fallback(last_outputs, data)
        return fixed_outputs, usage

    def annotate(self, data, component, component_name, annotation_path, session_filter: set | None = None):
        """
        Run Annotation (optionally in parallel over sessions).
        """
        start_time = time.time()
        required_annotations = self.load_required_annotations(annotation_path, component_name)
        all_annotations, completed_sessions, running_usage, previous_runtime, annotations_path = self.load_existing_output(
            annotation_path, component_name
        )

        if completed_sessions:
            logger.info(
                f"[TOPA] Resuming {component_name}: found {len(completed_sessions)} completed sessions "
                f"and {len(all_annotations)} existing rows."
            )

        n_procs = self.num_processes

        # -------- sequential --------
        if n_procs <= 1:
            for user_idx, user in enumerate(data):
                for session_idx, session in tqdm(
                    enumerate(user["sessions"]),
                    total=len(user["sessions"]),
                    desc=f"User {user_idx}",
                ):
                    if session_filter is not None and (user_idx, session_idx) not in session_filter:
                        continue
                    if (user_idx, session_idx) in completed_sessions:
                        continue

                    # session_annotations, usage = self.annotate_single_session(
                    #     user_idx, session_idx, session["dialogue"], session["session_metadata"], component, required_annotations
                    # )

                    try:
                        session_annotations, usage = self.annotate_single_session(
                            user_idx,
                            session_idx,
                            session["dialogue"],
                            session["session_metadata"],
                            component,
                            required_annotations,
                        )
                    except RuntimeError as e:
                        if "content_filter" in str(e) or "LLM call failed" in str(e):
                            logger.warning(
                                f"[SKIP] Skipping user_idx={user_idx}, session_idx={session_idx} "
                                f"for component={component_name} because LLM failed: {e}"
                            )
                            continue
                        raise

                    all_annotations.extend(session_annotations)
                    running_usage["inp_t"] += usage["inp_t"]
                    running_usage["out_t"] += usage["out_t"]

                    cost_data = self.build_cost_data(start_time, previous_runtime, running_usage)
                    annotations_path = self.save_output(all_annotations, cost_data, annotation_path, component_name)

            return annotations_path

        # -------- parallel over (user_idx, session_idx) tasks --------
        # Flatten nested loops to tasks
        tasks_with_w = []
        total_tasks = 0
        for user_idx, user in enumerate(data):
            for session_idx, session in enumerate(user["sessions"]):
                if session_filter is not None and (user_idx, session_idx) not in session_filter:
                    continue
                if (user_idx, session_idx) in completed_sessions:
                    continue
                dialogue = session["dialogue"]
                t = (user_idx, session_idx, dialogue, session["session_metadata"])
                tasks_with_w.append((t, _estimate_task_weight(dialogue, component_name)))
                total_tasks += 1

        if total_tasks == 0:
            return annotations_path

        # Balance batches so processes finish around the same time
        batches = _greedy_balance(tasks_with_w, n_procs)
        if n_procs in [2, 3]:
            batches[1] = batches[1][::-1] # for faster processing

        # Use spawn to clean process state
        ctx = mp.get_context("spawn")
        pbar = tqdm(total=total_tasks, desc="Annotating sessions (parallel)")
        annotations_path = None

        with ctx.Pool(
            processes=n_procs,
            initializer=_mp_init_worker,
            initargs=(self, component, required_annotations),
        ) as pool:
            for anns, usage, n_done in pool.imap_unordered(_mp_process_task_batch, batches, chunksize=1):
                all_annotations.extend(anns)
                running_usage["inp_t"] += usage["inp_t"]
                running_usage["out_t"] += usage["out_t"]
                pbar.update(int(n_done))

                cost_data = self.build_cost_data(start_time, previous_runtime, running_usage)
                annotations_path = self.save_output(all_annotations, cost_data, annotation_path, component_name)

        pbar.close()
        return annotations_path

    def load_required_annotations(self, annotation_path: str, component_name: str):
        """Load annotated data from other components if it is required to annotate with the current component."""
        if component_name in ["micro_actions", "conv_state", "intrinsic_reward"]:
            return pd.read_csv(osp.join(annotation_path, "annotations_macro_actions.csv"), encoding="utf-8")
        if component_name == "user_profile":
            macro_path = osp.join(annotation_path, "annotations_macro_actions.csv")
            if osp.exists(macro_path):
                return pd.read_csv(macro_path, encoding="utf-8")
            return None
        if component_name in ["knowledge_graph_retrieval", "kg_retrieval"]:
            return pd.read_csv(osp.join(annotation_path, "annotations_knowledge_graph.csv"), encoding="utf-8")
        return 

    def save_output(self, annotations: List, cost_data: Dict, output_path: str, component_name: str) -> str:
        """Save the annotations and the cost."""
        csv_path, usage_path = self.get_output_paths(output_path, component_name)

        if annotations:
            all_annotations = pd.DataFrame(annotations)
        else:
            all_annotations = pd.DataFrame(columns=["user_idx", "session_idx"])
        all_annotations.to_csv(csv_path, encoding="utf-8", index=False)

        with open(usage_path, "w", encoding="utf-8") as f:
            json.dump(cost_data, f, indent=2, ensure_ascii=False)
        return csv_path
