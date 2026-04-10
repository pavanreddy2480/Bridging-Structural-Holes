# src/stage0_seed_curation.py
# Plan Section 5.4 — v7.0 implementation.
# Validates 15 seed algorithms and writes seed_algorithms.json.

import json
import logging
from config.settings import SEED_ALGORITHMS, OGBN_LABEL_TO_CATEGORY

log = logging.getLogger(__name__)


def run_stage0() -> list:
    """
    v7.0 Changes:
    - 'home_label' (single int) replaced by 'established_labels' (list of ints)
    - Validator cross-checks every label in established_labels against OGBN map
    - Validator confirms established_domains strings match their label integers
      (this catches the silent v6.0 bug automatically)
    """
    required_keys = {
        "name", "established_labels", "established_domains",
        "canonical_terms", "problem_structure_terms", "exclusion_strings"
    }

    validated = []
    for seed in SEED_ALGORITHMS:
        missing = required_keys - set(seed.keys())
        if missing:
            raise ValueError(f"Seed '{seed.get('name', 'UNKNOWN')}' is missing fields: {missing}")

        # Validate ALL established labels exist in OGBN map
        for label in seed["established_labels"]:
            if label not in OGBN_LABEL_TO_CATEGORY:
                raise ValueError(
                    f"Seed '{seed['name']}' has established_label={label} "
                    f"not in OGBN_LABEL_TO_CATEGORY."
                )

        # Validate parallel array lengths
        if len(seed["established_labels"]) != len(seed["established_domains"]):
            raise ValueError(
                f"Seed '{seed['name']}': established_labels and established_domains "
                f"must have the same length."
            )

        # Cross-validate label integers against domain strings (catches v6.0 silent bug)
        for lbl, dom in zip(seed["established_labels"], seed["established_domains"]):
            expected = OGBN_LABEL_TO_CATEGORY[lbl]
            if expected != dom:
                raise ValueError(
                    f"Seed '{seed['name']}': label {lbl} maps to '{expected}' "
                    f"but established_domains says '{dom}'. Fix the mismatch."
                )

        if len(seed["canonical_terms"]) < 3:
            log.warning(f"Seed '{seed['name']}': only {len(seed['canonical_terms'])} canonical terms.")

        if len(seed["problem_structure_terms"]) < 5:
            log.warning(f"Seed '{seed['name']}': only {len(seed['problem_structure_terms'])} ps_terms.")

        validated.append(seed)

    log.info(f"\n── Stage 0: {len(validated)} seed algorithms validated ──")
    for s in validated:
        primary_cat = OGBN_LABEL_TO_CATEGORY.get(s["established_labels"][0], "UNKNOWN")
        established = ", ".join(s["established_domains"])
        log.info(
            f"  [{s['name']}] primary={primary_cat} | "
            f"established in: [{established}] | "
            f"{len(s['canonical_terms'])} canonical | "
            f"{len(s['problem_structure_terms'])} ps_terms | "
            f"{len(s['exclusion_strings'])} exclusion_strings"
        )

    with open("data/stage0_output/seed_algorithms.json", "w") as f:
        json.dump(validated, f, indent=2)
    log.info("Saved to data/stage0_output/seed_algorithms.json")
    return validated


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    run_stage0()
