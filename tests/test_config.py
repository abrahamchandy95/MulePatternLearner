"""The configuration schema: unknown keys, defaults, retired keys and override files."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from mule_pattern_learner.config import (
    DEFAULT_RUN,
    OPERATIONAL_DEFAULTS,
    LiveConfig,
    run_config,
    validate_config,
)
from mule_pattern_learner.contract.sampler_plan import SamplerPlan
from mule_pattern_learner.paths import load_config
from mule_pattern_learner.testing.builders import unit_config


def test_config_schema_rejects_unknown_keys_and_applies_defaults(
    tmp_path: Path,
) -> None:
    base = unit_config(tmp_path)
    result = validate_config(base)
    # Absent keys take their built-in or operational value; the optional ones stay absent.
    for key, value in {**DEFAULT_RUN, **OPERATIONAL_DEFAULTS}.items():
        assert result[key] == base.get(key, value), key
    for key in ("prepared_id", "cohort_seed", "reveal_salt"):
        assert key not in result
    assert result["sampler"] == base["sampler"] and base == unit_config(tmp_path)
    with pytest.raises(ValueError, match="Unknown configuration key.*learnig_rate"):
        validate_config({**base, "learnig_rate": 0.1})
    bad = [
        ({"fanouts": [True, 4]}, "fanouts"),
        ({"fanouts": [8]}, "fanouts"),
        ({"query_concurrency": 17}, "query_concurrency"),
        ({"request_batch_size": 65}, "request_batch_size"),
        ({"deterministic": "yes"}, "deterministic"),
        ({"prepared_id": "../escape"}, "prepared_id"),
        ({"sampler": {"recnt": 2}}, "sampler.recnt"),
        ({"sampler": {"children": {"older": 99}}}, "sampler.children.older"),
        ({"dates": {**base["dates"], "train": ["not a date"]}}, "dates"),
        ({"feature_groups": ["entity_meta", "no_such_group"]}, "no_such_group"),
    ]  # fmt: skip
    for change, name in bad:
        with pytest.raises(ValueError, match=name):
            validate_config({**base, **change})
    assert result["scope_unowned"] == "linked" and result["max_outage_s"] == 900
    for change, name in [
        ({"scope_unowned": "all"}, "scope_unowned"),
        ({"max_outage_s": -1}, "max_outage_s"),
        ({"max_outage_s": 1.5}, "max_outage_s"),
    ]:
        with pytest.raises(ValueError, match=name):
            validate_config({**base, **change})
    for policy in ("independent", "shared", "linked"):
        assert validate_config({**base, "scope_unowned": policy})["scope_unowned"] == policy
    # Checkpoint configurations used for scoring need not carry preparation keys.
    assert validate_config({"hidden": 16})["query_concurrency"] == 16
    assert validate_config({**base, "deterministic": "strict"})["deterministic"] == "strict"
    for weight in ("prior", "balanced", 0.5):
        assert validate_config({**base, "positive_weight": weight})["positive_weight"] == weight
    for weight in ("equal", 1.0, 0.0):
        with pytest.raises(ValueError, match="positive_weight"):
            validate_config({**base, "positive_weight": weight})


def test_configurations_saved_before_the_restructure_still_validate() -> None:
    # Saved models and prepared cohorts hold keys of removed paths, with the one value
    # that remains; validation drops them. Another value names a removed path.
    saved = {
        **run_config(),
        "context_storage": "stream",
        "evaluation_protocol": "strict_inductive",
        "label_policy": "graph_observed",
        "sampler": {**run_config()["sampler"], "policy": "resample"},
    }
    assert validate_config(saved) == run_config()
    with pytest.raises(ValueError, match="context_storage = 'sqlite' is no longer supported"):
        validate_config({**saved, "context_storage": "sqlite"})
    with pytest.raises(ValueError, match="evaluation_protocol = 'shared_history'"):
        validate_config({**saved, "evaluation_protocol": "shared_history"})
    with pytest.raises(ValueError, match="label_policy = 'observed' is no longer supported"):
        validate_config({**saved, "label_policy": "observed"})
    with pytest.raises(ValueError, match="observed_labels = 'x.parquet' is no longer supported$"):
        validate_config({**saved, "observed_labels": "x.parquet"})
    # The model variants became settings.
    built_in = run_config()
    assert validate_config({**saved, "variant": "temporal"}) == built_in
    assert validate_config({**saved, "variant": "tabular"}) == {
        **built_in,
        "architecture": "summary",
    }
    no_fourier = validate_config({**saved, "variant": "no_fourier"})
    groups = built_in["feature_groups"]
    assert no_fourier["feature_groups"] == [g for g in groups if g != "time_encoding"]
    assert "extraction_groups" not in no_fourier
    # Extraction groups only widened the request; the source asks for the model's groups.
    wider = {**saved, "extraction_groups": [*groups, "rolling_windows"]}
    assert validate_config(wider) == built_in
    with pytest.raises(ValueError, match="variant = 'wide' is no longer supported"):
        validate_config({**saved, "variant": "wide"})


def test_built_in_run_validates_and_only_run_config_applies_the_schema(tmp_path: Path) -> None:
    config = run_config()
    assert config == validate_config(config) and config["request_batch_size"] == 8
    assert {key: config[key] for key in DEFAULT_RUN} == DEFAULT_RUN
    # The field defaults are the operational defaults validate_config fills in.
    for key, value in OPERATIONAL_DEFAULTS.items():
        assert LiveConfig.model_fields[key].default == value, key
    toml = tmp_path / "c.toml"
    toml.write_text('dataset_id = "x"\nstage = "offline-only key"\n')
    assert load_config(toml)["stage"] == "offline-only key"
    with pytest.raises(ValueError, match="Unknown configuration key.*stage"):
        run_config(toml)


def test_override_tables_merge_into_the_built_in_run(tmp_path: Path) -> None:
    def overridden(text: str) -> dict[str, Any]:
        path = tmp_path / "overrides.toml"
        path.write_text(text)
        return run_config(path)

    default = run_config()
    torch_only = overridden('[sampler]\nbackend = "torch"\n')
    assert torch_only["sampler"] == {**DEFAULT_RUN["sampler"], "backend": "torch"}
    assert SamplerPlan.from_config(torch_only).fingerprint() == (
        SamplerPlan.from_config(default).fingerprint()
    )
    # A pool setting keeps every other sampler key.
    fewer = overridden("[sampler]\nrecent = 4\n[sampler.children]\nolder = 1\n")
    assert fewer["sampler"]["recent"] == 4 and fewer["sampler"]["older"] == 4
    assert fewer["sampler"]["children"] == {**DEFAULT_RUN["sampler"]["children"], "older": 1}
    assert fewer["sampler"]["relation_fanouts"] == DEFAULT_RUN["sampler"]["relation_fanouts"]
    # Only the resample sampler remains.
    with pytest.raises(ValueError, match="sampler.policy = 'recent' is no longer supported"):
        overridden('[sampler]\npolicy = "recent"\nrecent = 4\n')
    dates = overridden('[dates]\ntrain = ["2024-05-01", "2024-07-01"]\n')
    assert dates["dates"] == {**DEFAULT_RUN["dates"], "train": ["2024-05-01", "2024-07-01"]}
    # Lists and scalars replace the default.
    assert overridden("fanouts = [8, 2]\nepochs = 3\n")["fanouts"] == [8, 2]
    assert run_config() == default
