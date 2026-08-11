from __future__ import annotations

import pandas as pd
import pytest


def test_preprocess_workers_preserve_live_memory_reserve() -> None:
    from covid_rars.hst_parallel import ResourceSnapshot, choose_preprocess_workers

    workers = choose_preprocess_workers(
        snapshot=ResourceSnapshot(
            logical_cpus=24,
            cpu_affinity_count=24,
            mem_available_bytes=6 * 1024**3,
            cgroup_headroom_bytes=6 * 1024**3,
            parent_rss_bytes=1 * 1024**3,
            dev_shm_available_bytes=8 * 1024**3,
            swap_used_bytes=0,
        ),
        estimated_worker_bytes=700 * 1024**2,
        reserve_cpus=4,
        reserve_ram_bytes=4 * 1024**3,
        candidates=(1, 2, 4, 8, 12),
    )
    assert workers == 2


def test_gpu_lease_rejects_overlap(tmp_path) -> None:
    from covid_rars.hst_parallel import acquire_gpu_execution_lease

    with acquire_gpu_execution_lease(tmp_path, gpu_uuid="GPU-test", run_id="run-a"):
        with pytest.raises(BlockingIOError):
            with acquire_gpu_execution_lease(tmp_path, gpu_uuid="GPU-test", run_id="run-b"):
                pass


def test_loader_selection_uses_throughput_not_model_metrics() -> None:
    from covid_rars.hst_parallel import select_dataloader_workers

    benchmark = pd.DataFrame(
        {
            "workers": [0, 2, 4, 8],
            "batches_per_second": [2.0, 4.0, 5.0, 4.5],
            "valid": [True, True, True, True],
            "rss_delta_bytes": [0, 1, 2, 3],
        }
    )
    assert select_dataloader_workers(benchmark) == 4
    with pytest.raises(ValueError, match="model metrics"):
        select_dataloader_workers(benchmark.assign(auroc=0.9))


def test_gpu_jobs_execute_serially() -> None:
    from covid_rars.hst_parallel import run_single_gpu_job_queue

    ledger = run_single_gpu_job_queue([lambda: "a", lambda: "b"], device_count=1)
    assert ledger["concurrent_gpu_jobs"].max() == 1
    assert ledger["status"].tolist() == ["success", "success"]
