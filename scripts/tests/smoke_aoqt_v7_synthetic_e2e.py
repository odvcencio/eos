#!/usr/bin/env python3
"""Synthetic AOQT V7 cross-language E2E smoke.

This is a non-decision integration smoke. It uses deterministic synthetic
train/dev data only, writes outputs under a caller-provided scratch directory,
and asserts hash/fold/provenance bindings across:

  Python split build+validate -> Go fixture materializer -> Python fold
  materializer -> Go/native V7 dev runner -> toy dev proxy -> Python canary.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import stat
import subprocess
import sys
import tempfile
from pathlib import Path
from unittest import mock


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = REPO_ROOT / "scripts"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import aoqt_v7_movement_canary_gate as canary  # noqa: E402
import build_aoqt_stage2_calibration as stage2  # noqa: E402
import build_aoqt_v7_dev_split as splitter  # noqa: E402
import materialize_aoqt_v7_fold_train as fold_materializer  # noqa: E402


DOMAINS = ("fiqa", "nfcorpus", "scifact")
DIM = 384
QUERY_COUNT_PER_DOMAIN = 18
TOPOLOGY_SEED = 191
TURBOQUANT_SEED = 5581486560434873699


def sha256_json(payload: object) -> str:
    data = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def hex64(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def write_json(path: Path, payload: object) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def write_jsonl(path: Path, rows: list[object]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n" for row in rows), encoding="utf-8")
    return path


def unit_vec(index: int) -> list[float]:
    vec = [0.0] * DIM
    vec[index % DIM] = 1.0
    return vec


def run(cmd: list[str], *, cwd: Path = REPO_ROOT) -> subprocess.CompletedProcess[str]:
    proc = subprocess.run(cmd, cwd=cwd, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    if proc.returncode != 0:
        raise RuntimeError(
            "command failed "
            f"rc={proc.returncode}: {' '.join(cmd)}\nstdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
        )
    return proc


def make_official_registry(qids_by_dataset: dict[str, list[str]], source_by_file: dict[str, str]) -> dict:
    return {
        "schema": splitter.heldout_gate.TRUSTED_WORKLOAD_REGISTRY_SCHEMA,
        "status": "synthetic-non-decision",
        "domains": {
            domain: {
                "qrels_path_suffix": f"datasets/manta-embed-v1/raw/{domain}/{domain}/qrels/test.tsv",
                "qrels_sha256": source_by_file[f"datasets/manta-embed-v1/raw/{domain}/{domain}/qrels/test.tsv"],
                "qid_set_sha256": splitter.sha256_json(qids_by_dataset[domain]),
                "query_count": len(qids_by_dataset[domain]),
            }
            for domain in DOMAINS
        },
    }


def make_fixture_inputs(root: Path) -> dict[str, object]:
    source = root / "source"
    anchor = {
        "schema": stage2.ANCHOR_SCHEMA,
        "package_path": "synthetic-anchor.mll",
        "package_sha256": hex64("synthetic-anchor-package"),
        "embedding_dim": DIM,
        "topology": stage2.TOPOLOGY,
        "legal_scope": stage2.LEGAL_SCOPE,
    }
    anchor_path = write_json(source / "anchor.json", anchor)
    anchor_manifest_sha = sha256_file(anchor_path)
    space = "eos-d384-synthetic-anchor"

    official_qids = {domain: [f"{domain}-official-synthetic"] for domain in DOMAINS}
    official_source_by_file = {
        f"datasets/manta-embed-v1/raw/{domain}/{domain}/qrels/test.tsv": hex64(f"{domain}-official-qrels-synthetic")
        for domain in DOMAINS
    }
    official_payload = {
        "schema": stage2.EXCLUSION_SCHEMA,
        "name": "official-test",
        "qids_by_dataset": official_qids,
        "source_sha256": sha256_json(dict(sorted(official_source_by_file.items()))),
        "source_sha256_by_file": official_source_by_file,
    }
    official_path = write_json(source / "exclusions" / "official-test.json", official_payload)

    exclusion_paths: list[Path] = []
    exclusion_qids: dict[str, dict[str, list[str]]] = {
        "dev4": {domain: [f"{domain}-dev4-excluded"] for domain in DOMAINS},
        "reserve4": {domain: [f"{domain}-reserve4-excluded"] for domain in DOMAINS},
        "official-test": official_qids,
    }
    for name in ("dev4", "reserve4"):
        payload = {
            "schema": stage2.EXCLUSION_SCHEMA,
            "name": name,
            "qids_by_dataset": exclusion_qids[name],
            "source_sha256": hex64(f"{name}-source-synthetic"),
        }
        exclusion_paths.append(write_json(source / "exclusions" / f"{name}.json", payload))
    exclusion_paths.append(official_path)

    vectors_path = source / "vectors.jsonl"
    vector_rows = []
    vector_manifest_paths: list[Path] = []
    score_paths: list[Path] = []
    qrels_paths: list[Path] = []
    dataset_dirs: dict[str, Path] = {}
    anchor_docs: dict[str, Path] = {}
    anchor_queries: dict[str, Path] = {}

    for domain_index, domain in enumerate(DOMAINS):
        qids = [f"{domain}-train-{i:02d}" for i in range(1, QUERY_COUNT_PER_DOMAIN + 1)]
        doc_ids = [f"{domain}-d{i:03d}" for i in range(1, 121)]
        qrels_path = source / "qrels" / f"{domain}.qrels.jsonl"
        qrel_rows = []
        for qid in qids:
            vector_rows.append({
                "dataset": domain,
                "role": "query",
                "id": qid,
                "vector_id": f"vec-{qid}",
                "embedding_space_id": space,
                "vector": unit_vec(domain_index),
            })
            qrel_rows.append({"dataset": domain, "qid": qid, "doc_id": f"{domain}-d001", "gain": 1})
            if domain == "nfcorpus":
                qrel_rows.append({"dataset": domain, "qid": qid, "doc_id": f"{domain}-d080", "gain": 1})
        write_jsonl(qrels_path, qrel_rows)
        qrels_paths.append(qrels_path)
        qrels_sha = sha256_file(qrels_path)

        for i, doc_id in enumerate(doc_ids, start=1):
            vec = unit_vec((domain_index + i) % DIM)
            if i == 1 or (domain == "nfcorpus" and i == 80):
                vec = unit_vec(domain_index)
            vector_rows.append({
                "dataset": domain,
                "role": "doc",
                "id": doc_id,
                "vector_id": f"vec-{doc_id}",
                "embedding_space_id": space,
                "vector": vec,
            })

        for role, ids in (("query", qids), ("doc", doc_ids)):
            component = "query_vector" if role == "query" else "doc_vector"
            source_sha = hex64(f"{domain}-{component}-source")
            provenance = {component: source_sha}
            payload = {
                "schema": stage2.VECTOR_CACHE_SCHEMA,
                "dataset": domain,
                "split": "train",
                "role": role,
                "anchor": {"package_sha256": anchor["package_sha256"], "manifest_sha256": anchor_manifest_sha},
                "topology": {"id": "aoqt_givens_v1", "sha256": sha256_json(anchor["topology"])},
                "legal_scope": stage2.LEGAL_SCOPE,
                "source_sha256": source_sha,
                "qrels_sha256": qrels_sha,
                "dataset_source_provenance": provenance,
                "dataset_source_provenance_sha256": sha256_json(provenance),
                "ids": ids,
                "cache_sha256": hex64(f"{domain}-{role}-cache"),
            }
            vector_manifest_paths.append(write_json(source / "vector-manifests" / f"{domain}.{role}.json", payload))

        for bits in (3, 5):
            query_manifest_sha = sha256_file(source / "vector-manifests" / f"{domain}.query.json")
            doc_manifest_sha = sha256_file(source / "vector-manifests" / f"{domain}.doc.json")
            query_source = hex64(f"{domain}-query_vector-source")
            doc_source = hex64(f"{domain}-doc_vector-source")
            score_source = hex64(f"{domain}-q{bits}_score-source")
            provenance = {
                "query_vector": query_source,
                "doc_vector": doc_source,
                f"q{bits}_score": score_source,
            }
            docs = [
                {
                    "doc_id": doc_id,
                    "rank": rank,
                    "gain": 1 if rank == 1 or (domain == "nfcorpus" and rank == 80) else 0,
                }
                for rank, doc_id in enumerate(doc_ids, start=1)
            ]
            payload = {
                "schema": stage2.SCORE_CACHE_SCHEMA,
                "dataset": domain,
                "bits": bits,
                "split": "train",
                "score_mode": "prepared_ip",
                "top_k": 120,
                "turboquant": {"score_mode": "prepared_ip", "bits": bits, "seed": TURBOQUANT_SEED},
                "anchor": {"package_sha256": anchor["package_sha256"], "manifest_sha256": anchor_manifest_sha},
                "topology": {"id": "aoqt_givens_v1", "sha256": sha256_json(anchor["topology"])},
                "legal_scope": stage2.LEGAL_SCOPE,
                "source_sha256": score_source,
                "qrels_sha256": qrels_sha,
                "dataset_source_provenance": provenance,
                "dataset_source_provenance_sha256": sha256_json(provenance),
                "vector_cache": {
                    "query_manifest_sha256": query_manifest_sha,
                    "query_cache_sha256": hex64(f"{domain}-query-cache"),
                    "doc_manifest_sha256": doc_manifest_sha,
                    "doc_cache_sha256": hex64(f"{domain}-doc-cache"),
                },
                "rows": [{"qid": qid, "docs": docs} for qid in qids],
            }
            score_paths.append(write_json(source / "scores" / f"{domain}.q{bits}.json", payload))

        dataset = source / "dev-dataset-source" / domain
        dataset_dirs[domain] = dataset
        write_jsonl(dataset / "corpus.jsonl", [{"_id": doc_id, "text": f"{domain} synthetic {doc_id}"} for doc_id in doc_ids])
        write_jsonl(dataset / "queries.jsonl", [{"_id": qid, "text": f"{domain} synthetic {qid}"} for qid in qids])
        qrels_tsv = dataset / "qrels" / "train.tsv"
        qrels_tsv.parent.mkdir(parents=True, exist_ok=True)
        qrels_tsv.write_text(
            "query-id\tcorpus-id\tscore\n" + "".join(f"{row['qid']}\t{row['doc_id']}\t{row['gain']}\n" for row in qrel_rows),
            encoding="utf-8",
        )
        anchor_docs[domain] = source / "proxy-vectors" / domain / "docs.jsonl"
        anchor_queries[domain] = source / "proxy-vectors" / domain / "queries.jsonl"
        write_jsonl(anchor_docs[domain], [{"id": doc_id, "embedding": unit_vec((domain_index + i) % DIM)} for i, doc_id in enumerate(doc_ids, start=1)])
        write_jsonl(anchor_queries[domain], [{"id": qid, "embedding": unit_vec(domain_index)} for qid in qids])

    write_jsonl(vectors_path, vector_rows)

    plan_path = source / "plan.json"
    stage2_args = [
        sys.executable,
        str(SCRIPT_DIR / "build_aoqt_stage2_calibration.py"),
        "--anchor-manifest",
        str(anchor_path),
        "--output-plan",
        str(plan_path),
    ]
    for path in exclusion_paths:
        stage2_args += ["--exclusion-qids", str(path)]
    for path in vector_manifest_paths:
        stage2_args += ["--vector-cache", str(path)]
    for path in score_paths:
        stage2_args += ["--score-cache", str(path)]
    run(stage2_args)

    return {
        "anchor": anchor,
        "anchor_path": anchor_path,
        "anchor_manifest_sha256": anchor_manifest_sha,
        "official_path": official_path,
        "official_registry": make_official_registry(official_qids, official_source_by_file),
        "exclusion_paths": exclusion_paths,
        "vectors_path": vectors_path,
        "score_paths": score_paths,
        "qrels_paths": qrels_paths,
        "plan_path": plan_path,
        "space": space,
        "dataset_dirs": dataset_dirs,
        "anchor_docs": anchor_docs,
        "anchor_queries": anchor_queries,
    }


def materialize_source_with_go(root: Path, fixture: dict[str, object]) -> dict[str, Path]:
    out = root / "go-source-materialized"
    out.mkdir(parents=True, exist_ok=True)
    cmd = [
        "go",
        "run",
        "./cmd/eos",
        "materialize-aoqt-sidecar",
        "--plan",
        str(fixture["plan_path"]),
        "--rows-jsonl",
        str(out / "rows.jsonl"),
        "--manifest-json",
        str(out / "manifest.json"),
        "--preflight-json",
        str(out / "preflight.json"),
        "--anchor-embedding-space-id",
        str(fixture["space"]),
        "--anchor-artifact",
        "synthetic-anchor.mll",
        "--created-at-utc",
        "2026-09-05T00:00:00Z",
        "--allow-research-only-aoqt",
    ]
    for path in fixture["exclusion_paths"]:
        cmd += ["--exclusion-qids", str(path)]
    cmd += ["--vectors", str(fixture["vectors_path"])]
    for path in fixture["score_paths"]:
        cmd += ["--score-evidence", str(path)]
    for path in fixture["qrels_paths"]:
        cmd += ["--qrels", str(path)]
    run(cmd)
    return {"manifest": out / "manifest.json", "rows": out / "rows.jsonl", "preflight": out / "preflight.json"}


def build_and_validate_split(root: Path, fixture: dict[str, object]) -> dict[str, Path | dict]:
    split_path = root / "split" / "aoqt-v7-synthetic-split.json"
    validation_path = root / "split" / "aoqt-v7-synthetic-split.validation.json"
    registry = copy.deepcopy(fixture["official_registry"])
    with mock.patch.object(splitter.heldout_gate, "PRODUCTION_TRUSTED_WORKLOAD_REGISTRY", registry):
        manifest = splitter.build_manifest(Path(fixture["plan_path"]), Path(fixture["official_path"]), 2, "aoqt-v7-synthetic-e2e", 0.0)
        write_json(split_path, manifest)
        validation = splitter.validate_manifest(split_path, Path(fixture["plan_path"]), Path(fixture["official_path"]))
        write_json(validation_path, validation)
    return {"manifest": split_path, "validation": validation_path, "registry": registry}


def materialize_fold(root: Path, fixture: dict[str, object], split: dict[str, Path | dict], source: dict[str, Path]) -> dict[str, Path | dict]:
    out = root / "fold-materialized"
    args = type(
        "Args",
        (),
        {
            "split_manifest": split["manifest"],
            "split_validation": split["validation"],
            "fold_id": "fold-0",
            "source_plan": fixture["plan_path"],
            "source_materialized_manifest": source["manifest"],
            "source_materialized_rows": source["rows"],
            "source_materialized_preflight": source["preflight"],
            "output_dir": out,
            "expected_split_manifest_sha256": sha256_file(Path(split["manifest"])),
            "expected_split_validation_sha256": sha256_file(Path(split["validation"])),
            "expected_source_plan_sha256": sha256_file(Path(fixture["plan_path"])),
            "expected_source_materialized_manifest_sha256": sha256_file(source["manifest"]),
            "expected_source_materialized_rows_sha256": sha256_file(source["rows"]),
            "expected_source_materialized_preflight_sha256": sha256_file(source["preflight"]),
        },
    )
    with mock.patch.object(fold_materializer.splitter.heldout_gate, "PRODUCTION_TRUSTED_WORKLOAD_REGISTRY", copy.deepcopy(split["registry"])):
        receipt = fold_materializer.materialize(args)
    return {
        "manifest": out / "manifest.json",
        "rows": out / "rows.jsonl",
        "preflight": out / "preflight.json",
        "row_ids": out / "row-ids.json",
        "receipt": receipt,
    }


def run_native_v7(root: Path, split: dict[str, Path | dict], fold: dict[str, Path | dict]) -> dict[str, Path | dict]:
    manifest = json.loads(Path(fold["manifest"]).read_text(encoding="utf-8"))
    dev_dir = root / "native-dev"
    metrics = root / "native-metrics.json"
    source_hashes = root / "native-source-artifact-sha256.txt"
    source_hashes.write_text("".join(f"{value}\n" for value in manifest["source_artifact_hashes"]), encoding="utf-8")
    cmd = [
        "go",
        "run",
        "./cmd/eos",
        "train-aoqt-sidecar",
        "--allow-research-only-aoqt",
        "--allow-dev-aoqt-v7",
        "--optimizer-mode",
        canary.V7_OPTIMIZER_MODE,
        "--manifest",
        str(fold["manifest"]),
        "--rows",
        str(fold["rows"]),
        "--preflight",
        str(fold["preflight"]),
        "--metrics-json",
        str(metrics),
        "--dev-output-dir",
        str(dev_dir),
        "--expected-manifest-sha256",
        str(fold["receipt"]["outputs"]["manifest_sha256"]),
        "--expected-rows-sha256",
        str(fold["receipt"]["outputs"]["rows_sha256"]),
        "--expected-preflight-sha256",
        str(fold["receipt"]["outputs"]["preflight_sha256"]),
        "--split-manifest",
        str(split["manifest"]),
        "--expected-split-manifest-sha256",
        sha256_file(Path(split["manifest"])),
        "--fold-id",
        "fold-0",
        "--expected-anchor-artifact-sha256",
        manifest["anchor_artifact_sha256"],
        "--expected-anchor-package-manifest-sha256",
        manifest["anchor_package_manifest_sha256"],
        "--anchor-embedding-space-id",
        manifest["anchor_embedding_space_id"],
        "--compatibility-digest",
        manifest["compatibility_digest"],
        "--source-artifact-sha256-file",
        str(source_hashes),
        "--source-artifact-sha256-file-sha256",
        sha256_file(source_hashes),
        "--vector-cache-sha256",
        ",".join(manifest["vector_cache_hashes"]),
        "--max-steps",
        "1",
        "--lr",
        "0.01",
    ]
    run(cmd)
    evidence = dev_dir / "aoqt-sidecar-dev-evidence.json"
    evidence_payload = json.loads(evidence.read_text(encoding="utf-8"))
    return {
        "metrics": metrics,
        "dev_evidence": evidence,
        "sidecar": Path(evidence_payload["transform_path"]),
        "dev_dir": dev_dir,
    }


def make_fake_eos(path: Path) -> None:
    script = r'''#!/usr/bin/env python3
import json
import shutil
import sys
from pathlib import Path

args = sys.argv[1:]
cmd = args[0]

def flag(name):
    return args[args.index(name) + 1]

def write_json(path, payload):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")

def dataset_qids(dataset_path):
    qids = []
    with (Path(dataset_path) / "queries.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                qids.append(row.get("_id") or row.get("id"))
    return qids

def write_per(path, qids, candidate, bits):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    anchor_docs = [f"d{i:02d}" for i in range(1, 12)]
    candidate_docs = list(reversed(anchor_docs)) if bits == 3 and candidate else anchor_docs
    docs = candidate_docs if candidate else anchor_docs
    with Path(path).open("w", encoding="utf-8") as handle:
        for qid in qids:
            row = {
                "schema": "manta.embedding_turboquant_retrieval_per_query.v1",
                "dataset": flag("--dataset"),
                "query_id": qid,
                "method": f"turboquant_ip_b{bits}",
                "bits": bits,
                "scoring_surface": "turboquant_ip_prepared",
                "top_k": [{"rank": i + 1, "doc_id": doc, "score": 1.0 / (i + 1), "relevance": 0} for i, doc in enumerate(docs)],
            }
            handle.write(json.dumps(row, sort_keys=True) + "\n")

if cmd == "transform-aoqt-vectors":
    Path(flag("--out-doc-vectors")).parent.mkdir(parents=True, exist_ok=True)
    Path(flag("--out-query-vectors")).parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(flag("--doc-vectors"), flag("--out-doc-vectors"))
    shutil.copyfile(flag("--query-vectors"), flag("--out-query-vectors"))
    write_json(flag("--sidecar-json"), {"schema": "eos.aoqt.vector_cache_transform_binding.v1", "synthetic": True})
    sys.exit(0)

candidate = "candidate" in flag("--backend")
dataset_path = args[-1]
qids = dataset_qids(dataset_path)
if cmd == "eval-retrieval-vectors":
    write_json(flag("--metrics-json"), {"schema": "manta.embedding_retrieval_metrics.v1", "dataset": flag("--dataset"), "quality": {"ndcg_at_10": 0.5, "recall_at_100": 1.0}})
    Path(flag("--per-query-jsonl")).parent.mkdir(parents=True, exist_ok=True)
    Path(flag("--per-query-jsonl")).write_text("", encoding="utf-8")
    sys.exit(0)
if cmd == "eval-retrieval-vectors-turboquant":
    bits = int(flag("--bits"))
    delta = 0.001 if candidate and bits == 3 else 0.0
    ndcg = 0.5 + delta
    write_json(flag("--metrics-json"), {"schema": "manta.embedding_turboquant_retrieval_metrics.v1", "dataset": flag("--dataset"), "rows": [{"bits": bits, "method": f"turboquant_ip_b{bits}", "rerank_overfetch": 0, "quality": {"ndcg_at_10": ndcg, "recall_at_100": 1.0}}]})
    Path(flag("--metrics-tsv")).parent.mkdir(parents=True, exist_ok=True)
    Path(flag("--metrics-tsv")).write_text("dataset\trow\n", encoding="utf-8")
    write_per(flag("--per-query-jsonl"), qids, candidate, bits)
    sys.exit(0)
sys.exit(9)
'''
    path.write_text(script, encoding="utf-8")
    path.chmod(path.stat().st_mode | stat.S_IXUSR)


def build_dev_proxy(root: Path, fixture: dict[str, object], split: dict[str, Path | dict], native: dict[str, Path | dict]) -> Path:
    fake_eos = root / "fake-eos.py"
    make_fake_eos(fake_eos)
    output = root / "dev-proxy-report.json"
    work_dir = root / "dev-proxy-work"
    cmd = [
        sys.executable,
        str(SCRIPT_DIR / "build_aoqt_v7_non_official_dev_proxy.py"),
        "--split-manifest",
        str(split["manifest"]),
        "--split-validation",
        str(split["validation"]),
        "--fold-id",
        "fold-0",
        "--official-qids",
        str(fixture["official_path"]),
        "--sidecar",
        str(native["sidecar"]),
        "--train-metrics",
        str(native["metrics"]),
        "--dev-evidence",
        str(native["dev_evidence"]),
        "--preflight",
        str(root / "fold-materialized" / "preflight.json"),
        "--output",
        str(output),
        "--work-dir",
        str(work_dir),
        "--eos-bin",
        str(fake_eos),
        "--execute",
    ]
    for domain in DOMAINS:
        cmd += ["--dataset-dir", f"{domain}={fixture['dataset_dirs'][domain]}"]
        cmd += ["--anchor-doc-vectors", f"{domain}={fixture['anchor_docs'][domain]}"]
        cmd += ["--anchor-query-vectors", f"{domain}={fixture['anchor_queries'][domain]}"]
    run(cmd)
    return output


def run_canary(root: Path, split: dict[str, Path | dict], fold: dict[str, Path | dict], native: dict[str, Path | dict], proxy: Path) -> dict:
    output = root / "canary-result.json"
    proc = subprocess.run(
        [
            sys.executable,
            str(SCRIPT_DIR / "aoqt_v7_movement_canary_gate.py"),
            "--train-metrics",
            str(native["metrics"]),
            "--sidecar",
            str(native["sidecar"]),
            "--dev-evidence",
            str(native["dev_evidence"]),
            "--dev-proxy-report",
            str(proxy),
            "--split-manifest",
            str(split["manifest"]),
            "--split-validation",
            str(split["validation"]),
            "--preflight",
            str(fold["preflight"]),
            "--output-json",
            str(output),
            "--no-fail-exit-code",
        ],
        cwd=REPO_ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"canary command failed rc={proc.returncode}\nstdout:\n{proc.stdout}\nstderr:\n{proc.stderr}")
    result = json.loads(output.read_text(encoding="utf-8"))
    if result["split_manifest"]["file_sha256"] != sha256_file(Path(split["manifest"])):
        raise AssertionError("canary split manifest file hash did not bind")
    if result["split_validation"]["file_sha256"] != sha256_file(Path(split["validation"])):
        raise AssertionError("canary split validation file hash did not bind")
    if result["dev_evidence_file_binding"]["metrics_sha256"] != sha256_file(Path(native["metrics"])):
        raise AssertionError("canary metrics hash did not bind")
    if result["dev_evidence_file_binding"]["transform_sha256"] != sha256_file(Path(native["sidecar"])):
        raise AssertionError("canary sidecar hash did not bind")
    if result["train_split_binding"]["materialized_manifest_sha256"] != fold["receipt"]["outputs"]["manifest_sha256"]:
        raise AssertionError("canary train split binding did not bind materialized manifest")
    return result


def write_summary(root: Path, split: dict[str, Path | dict], fold: dict[str, Path | dict], native: dict[str, Path | dict], proxy: Path, canary_result: dict) -> Path:
    metrics = json.loads(Path(native["metrics"]).read_text(encoding="utf-8"))
    sidecar = json.loads(Path(native["sidecar"]).read_text(encoding="utf-8"))
    proxy_payload = json.loads(proxy.read_text(encoding="utf-8"))
    summary = {
        "schema": "eos.aoqt.v7_synthetic_e2e_smoke_summary.v1",
        "label": "synthetic_non_decision",
        "official_data_used": False,
        "quality_claim": False,
        "stages": [
            "python_stage2_fixture_plan",
            "python_v7_split_build_validate",
            "go_source_materializer_fixture_setup",
            "python_v7_fold_materializer",
            "go_native_v7_dev_runner",
            "toy_dev_proxy_transform_eval",
            "python_v7_canary",
        ],
        "files": {
            "split_manifest": {"path": str(split["manifest"]), "sha256": sha256_file(Path(split["manifest"]))},
            "split_validation": {"path": str(split["validation"]), "sha256": sha256_file(Path(split["validation"]))},
            "fold_manifest": {"path": str(fold["manifest"]), "sha256": fold["receipt"]["outputs"]["manifest_sha256"]},
            "fold_rows": {"path": str(fold["rows"]), "sha256": fold["receipt"]["outputs"]["rows_sha256"]},
            "fold_preflight": {"path": str(fold["preflight"]), "sha256": fold["receipt"]["outputs"]["preflight_sha256"]},
            "native_metrics": {"path": str(native["metrics"]), "sha256": sha256_file(Path(native["metrics"]))},
            "native_sidecar": {"path": str(native["sidecar"]), "sha256": sha256_file(Path(native["sidecar"]))},
            "dev_evidence": {"path": str(native["dev_evidence"]), "sha256": sha256_file(Path(native["dev_evidence"]))},
            "dev_proxy": {"path": str(proxy), "sha256": sha256_file(proxy)},
            "canary": {"path": str(root / "canary-result.json"), "sha256": sha256_file(root / "canary-result.json")},
        },
        "bindings": {
            "fold_id": "fold-0",
            "fold_row_count": fold["receipt"]["row_count"],
            "metrics_split_binding_equals_dev_evidence": metrics["split_binding"] == json.loads(Path(native["dev_evidence"]).read_text(encoding="utf-8"))["split_binding"],
            "sidecar_pairings_sha256": sidecar["audit"]["pairings_sha256"],
            "sidecar_angles_sha256": sidecar["audit"]["angles_sha256"],
            "dev_proxy_split_manifest_sha256": proxy_payload["split_manifest"]["split_manifest_file_sha256"],
            "canary_decision_sha256": canary_result["decision_sha256"],
        },
        "native_runner": {
            "optimizer_mode": metrics["summary"]["optimizer_mode"],
            "angle_l2": metrics["summary"]["angle_l2"],
            "accepted_proposals": metrics["summary"]["optimizer_diagnostics"]["accepted_proposals"],
            "proposal_attempts": metrics["summary"]["optimizer_diagnostics"]["proposal_attempts"],
            "trust_region_proposal_attempts": metrics["summary"]["optimizer_diagnostics"].get("trust_region_proposal_attempts", 0),
        },
        "dev_proxy": {
            "q3_macro_ndcg_at_10_delta": proxy_payload["macro"]["q3_ndcg_at_10_delta"],
            "q3_top10_churn": proxy_payload["macro"]["q3_top10_churn"],
            "full_real_proxy_ran": proxy_payload["execution"]["full_real_proxy_ran"],
        },
        "canary": {
            "decision": canary_result["decision"],
            "pass": canary_result["pass"],
            "failures": canary_result["failures"],
        },
    }
    if not summary["bindings"]["metrics_split_binding_equals_dev_evidence"]:
        raise AssertionError("native metrics/dev evidence split binding mismatch")
    if summary["bindings"]["sidecar_pairings_sha256"] != canary.CANONICAL_D384_SEED191_PAIRINGS_SHA256:
        raise AssertionError("native sidecar pairings are not canonical seed191")
    if not canary_result["pass"]:
        allowed = {
            "angles_nonzero_absolute",
            "angles_nonzero_fraction",
            "angles_not_single_angle",
            "angle_l2",
            "unique_nonzero_angles",
            "accepted_proposals_absolute",
            "accepted_proposals_fraction",
        }
        unexpected = sorted(set(canary_result["failures"]) - allowed)
        if unexpected:
            raise AssertionError(f"unexpected canary failures: {unexpected}")
    summary_path = write_json(root / "summary.json", summary)
    return summary_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work-root", type=Path, help="scratch directory to create/use for this smoke")
    args = parser.parse_args(argv)

    if args.work_root is None:
        with tempfile.TemporaryDirectory(prefix="aoqt-v7-synthetic-e2e-") as td:
            return run_smoke(Path(td))
    return run_smoke(args.work_root)


def run_smoke(root: Path) -> int:
    if root.exists() and any(root.iterdir()):
        raise RuntimeError(f"work root already exists and is not empty: {root}")
    root.mkdir(parents=True, exist_ok=True)
    fixture = make_fixture_inputs(root)
    source = materialize_source_with_go(root, fixture)
    split = build_and_validate_split(root, fixture)
    fold = materialize_fold(root, fixture, split, source)
    native = run_native_v7(root, split, fold)
    proxy = build_dev_proxy(root, fixture, split, native)
    canary_result = run_canary(root, split, fold, native, proxy)
    summary = write_summary(root, split, fold, native, proxy, canary_result)
    print(json.dumps({"summary": str(summary), "canary_decision": canary_result["decision"], "canary_pass": canary_result["pass"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
