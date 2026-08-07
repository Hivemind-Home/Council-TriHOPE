"""Fail fast, loudly, and all at once — before a GPU-hour is spent.

Every failure mode below has already cost this project something:

* ``code_recurrent`` asked for 8 buckets from a corpus with 3. The error
  came from :mod:`hivemind.data.stream` at schedule-construction time, named
  one phase, and appeared only after the whole split had downloaded.
* ``teachers.teacher_ids`` did not match the published ``teacher_id``
  values, so every sample routed to teacher 0 — with no error at all.
* Exact match ran on 3 probes because gold rows are sparse and the eval
  loader is capped for speed. Also no error.
* A load failure used to be a ``logger.warning`` on an unconfigured root
  logger, i.e. a silently missing domain.

So this module collects **every** problem and raises once, with counts and
copy-pasteable values, rather than surfacing the first one and hiding the
rest behind it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional, Sequence

from .hf_loader import normalize_domain_specs


class PreflightError(RuntimeError):
    """One or more configuration/data mismatches, reported together."""


@dataclass
class PreflightReport:
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    facts: dict[str, Any] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return not self.errors

    def render(self) -> str:
        out: list[str] = []
        if self.errors:
            out.append(f"{len(self.errors)} blocking problem(s):")
            out += [f"  {i}. {e}" for i, e in enumerate(self.errors, 1)]
        if self.warnings:
            out.append(f"{len(self.warnings)} warning(s):")
            out += [f"  - {w}" for w in self.warnings]
        return "\n".join(out)


def _phase_domains(phase: dict) -> list[str]:
    doms = list(phase.get("domains") or [])
    if not doms and phase.get("domain"):
        doms = [phase["domain"]]
    return [str(d) for d in doms]


def validate(
    data_cfg: dict,
    stream_cfg: Optional[dict] = None,
    teachers_cfg: Optional[dict] = None,
    eval_cfg: Optional[dict] = None,
    model_cfg: Optional[dict] = None,
    dataset: Any = None,
    tokenizer: Any = None,
) -> PreflightReport:
    """Check a resolved config (and, when given, the loaded dataset).

    ``dataset`` is optional so the same checks can run in a metadata-only
    mode that never downloads a split. The data-dependent checks are simply
    skipped in that case, and the report says so.
    """
    rep = PreflightReport()
    stream_cfg = stream_cfg or {}
    teachers_cfg = teachers_cfg or {}
    eval_cfg = eval_cfg or {}
    model_cfg = model_cfg or {}

    specs = normalize_domain_specs(
        list(data_cfg.get("domains", [])),
        data_cfg.get("repo_prefix", "hivemind-research"),
    )
    declared = {s.name for s in specs}
    rep.facts["domains"] = sorted(declared)

    if not specs:
        rep.errors.append("data.domains is empty — nothing to train on.")
        return rep

    # -- config-only checks ------------------------------------------------

    teacher_ids = dict(teachers_cfg.get("teacher_ids") or {})
    for spec in specs:
        if spec.name not in teacher_ids:
            rep.warnings.append(
                f"teachers.teacher_ids has no entry for domain {spec.name!r}; "
                "MetadataRouter can then only resolve it by domain name."
            )

    if str(teachers_cfg.get("mode", "cache")) == "live":
        live = dict(teachers_cfg.get("pretrained") or {})
        for spec in specs:
            if spec.name not in live:
                rep.errors.append(
                    f"teachers.mode=live but domain {spec.name!r} has no "
                    "teachers.pretrained entry. It would not be registered, so its "
                    "samples would fall through to teacher index 0 and distil "
                    "through the wrong expert."
                )

    phases = list(stream_cfg.get("phases") or [])
    for phase in phases:
        for name in _phase_domains(phase):
            if name not in declared:
                rep.errors.append(
                    f"stream phase {phase.get('name')!r} uses domain {name!r}, which "
                    f"data.domains does not declare (declared: {sorted(declared)})."
                )
        weights = phase.get("weights")
        if weights is not None and len(weights) != len(_phase_domains(phase)):
            rep.errors.append(
                f"stream phase {phase.get('name')!r} has {len(weights)} weights for "
                f"{len(_phase_domains(phase))} domains."
            )

    if tokenizer is not None and model_cfg.get("vocab_size"):
        tok_v = int(getattr(tokenizer, "vocab_size", 0) or 0)
        mod_v = int(model_cfg["vocab_size"])
        if tok_v and tok_v != mod_v:
            rep.warnings.append(
                f"tokenizer.vocab_size={tok_v} != model.vocab_size={mod_v}. The "
                "collator sizes cached teacher logits to the model vocab; a "
                "mismatch here means any NPZ cache built at the tokenizer's "
                "number would be rejected by batch_teacher_forward's shape check."
            )

    if dataset is None:
        rep.facts["mode"] = "metadata-only"
        return rep
    rep.facts["mode"] = "full"

    # -- data-dependent checks --------------------------------------------

    loaded = set(getattr(dataset, "domains", []))
    for spec in specs:
        if spec.name not in loaded:
            rep.errors.append(
                f"domain {spec.name!r} is declared but did not load "
                f"(loaded: {sorted(loaded)})."
            )

    batch_size = int(data_cfg.get("batch_size", 1))
    per_domain_claimed: dict[str, int] = {name: 0 for name in loaded}

    for phase in phases:
        mode = str(phase.get("mode", "random"))
        pname = phase.get("name")
        doms = _phase_domains(phase)
        if mode == "recurrent":
            rec = dict(phase.get("recurrence") or {})
            if rec.get("reuse_from"):
                continue
            want_n = int(rec.get("num_buckets", 8))
            want_rows = int(rec.get("samples_per_bucket", 16))
            domain = doms[0] if doms else None
            if domain not in loaded:
                continue
            buckets = dataset.indices_by_bucket(domain)
            eligible = [k for k, v in buckets.items() if len(v) >= want_rows]
            rep.facts.setdefault("buckets", {})[domain] = {
                "total": len(buckets),
                f"with_ge_{want_rows}_rows": len(eligible),
            }
            if len(eligible) < want_n:
                rep.errors.append(
                    f"stream phase {pname!r}: needs {want_n} buckets with "
                    f">={want_rows} rows in domain {domain!r}, but only "
                    f"{len(eligible)} of {len(buckets)} qualify. Either lower "
                    f"num_buckets to {len(eligible)}, or give the domain a finer "
                    "bucket key via data.domains[].bucket_columns "
                    "(e.g. [subdomain, difficulty, source])."
                )
            per_domain_claimed[domain] += want_n * want_rows
            steps = int(phase.get("steps", 0))
            novel_frac = float(rec.get("novel_fraction", 0.0) or 0.0)
            if novel_frac > 0:
                period = int(rec.get("revisit_period") or want_n)
                bg_steps = steps * max(0, period - want_n) / max(1, period)
                per_domain_claimed[domain] += int(bg_steps * novel_frac * batch_size)
        elif mode == "novel":
            domain = doms[0] if doms else None
            if domain in loaded:
                per_domain_claimed[domain] += int(phase.get("steps", 0)) * batch_size

    for domain, claimed in per_domain_claimed.items():
        available = len(dataset.indices_for_domain(domain))
        rep.facts.setdefault("rows", {})[domain] = available
        if claimed > available:
            rep.errors.append(
                f"domain {domain!r}: the stream needs ~{claimed} single-use rows but "
                f"only {available} are loaded. Raise data.max_rows_per_domain or "
                "shorten the novel phases."
            )
        elif claimed > 0.5 * available:
            rep.warnings.append(
                f"domain {domain!r}: the stream claims ~{claimed} of {available} rows "
                "(>50%); novel batches may run short."
            )

    # teacher_id values actually present in the data
    for spec in specs:
        if spec.name not in loaded or spec.name not in teacher_ids:
            continue
        actual = _sample_teacher_ids(dataset, spec.name)
        if actual and teacher_ids[spec.name] not in actual:
            rep.errors.append(
                f"teachers.teacher_ids.{spec.name} = "
                f"{teacher_ids[spec.name]!r} does not appear in the data. Observed: "
                f"{sorted(actual)}. Routing would fall back to the domain name."
            )

    # exact-match probe availability
    em = dict(eval_cfg.get("exact_match") or {})
    if em.get("enabled"):
        want = int(em.get("num_samples", 64))
        for domain in [str(d) for d in em.get("domains", [])]:
            if domain not in loaded:
                rep.warnings.append(
                    f"eval.exact_match.domains lists {domain!r}, which is not loaded."
                )
                continue
            try:
                gold = len(dataset.gold_indices(domain))
            except (KeyError, AttributeError):
                continue
            rep.facts.setdefault("gold_rows", {})[domain] = gold
            if gold == 0:
                rep.errors.append(
                    f"eval.exact_match: domain {domain!r} has no has_gold_label rows "
                    "in the scanned window, so it would contribute no EM column."
                )
            elif gold < want:
                rep.warnings.append(
                    f"eval.exact_match: domain {domain!r} has {gold} gold rows for "
                    f"num_samples={want}; EM would be quantized to 1/{gold}. Raise "
                    "eval.exact_match.max_scan_rows."
                )

    logits_root = data_cfg.get("teacher_logits_root")
    if logits_root:
        rep.warnings.append(
            f"data.teacher_logits_root={logits_root!r} is set; verify the NPZ files "
            "exist, otherwise KD stays inert and only CE trains."
        )

    return rep


def _sample_teacher_ids(dataset: Any, domain: str, limit: int = 64) -> set[str]:
    """Distinct teacher_id values from the head of a domain (cheap probe)."""
    try:
        idxs = dataset.indices_for_domain(domain)
    except (KeyError, AttributeError):
        return set()
    out: set[str] = set()
    for i in list(idxs)[:limit]:
        tid = dataset[i].get("teacher_id")
        if tid:
            out.add(str(tid))
    return out


def validate_or_raise(*args: Any, **kwargs: Any) -> PreflightReport:
    """:func:`validate`, but raise :class:`PreflightError` on any error."""
    rep = validate(*args, **kwargs)
    if rep.warnings:
        print("[preflight] " + rep.render())
    if not rep.ok:
        raise PreflightError("preflight failed.\n" + rep.render())
    return rep


def validate_config(cfg: Any, dataset: Any = None, tokenizer: Any = None) -> PreflightReport:
    """Convenience wrapper taking a whole (Hydra) config object."""
    from omegaconf import OmegaConf

    def sub(key: str) -> dict:
        node = cfg.get(key, {}) if hasattr(cfg, "get") else {}
        if node is None:
            return {}
        return (
            OmegaConf.to_container(node, resolve=True)
            if not isinstance(node, dict)
            else node
        )

    return validate(
        data_cfg=sub("data"),
        stream_cfg=sub("stream"),
        teachers_cfg=sub("teachers"),
        eval_cfg=sub("eval"),
        model_cfg=sub("model"),
        dataset=dataset,
        tokenizer=tokenizer,
    )


def domains_from(specs: Sequence[Any]) -> list[str]:
    return [s.name for s in normalize_domain_specs(list(specs))]
