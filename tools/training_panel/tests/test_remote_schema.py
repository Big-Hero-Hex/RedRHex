from pathlib import Path


MIGRATION = (
    Path(__file__).resolve().parents[1]
    / "supabase"
    / "migrations"
    / "20260814_370_remote_parity.sql"
)


def migration_text() -> str:
    return MIGRATION.read_text(encoding="utf-8").lower()


def test_remote_parity_migration_is_additive_and_idempotent() -> None:
    sql = migration_text()
    assert "create table if not exists public.machine_capabilities" in sql
    assert "create table if not exists public.physics_presets" in sql
    assert "alter table public.jobs add column if not exists client_request_id" in sql
    assert "alter table public.runs add column if not exists metrics" in sql
    assert "alter table public.team_activity_events add column if not exists source_id" in sql
    assert "drop table" not in sql
    assert "truncate " not in sql
    assert "delete from" not in sql
    assert "on conflict (id) do nothing" in sql


def test_remote_parity_migration_enforces_identity_and_constrained_mutations() -> None:
    sql = migration_text()
    assert "new.actor_id := auth.uid()" in sql
    assert "new.actor_role := authoritative_role" in sql
    assert "drop policy if exists \"operators can update remote run metadata\"" in sql
    assert "function public.update_run_metadata" in sql
    assert "function public.cancel_queued_job" in sql
    assert "actor_id = auth.uid() or authoritative_role = 'admin'" in sql
    assert "unique index if not exists idx_jobs_machine_actor_client_request" in sql


def test_remote_parity_gpu_claim_contract_only_blocks_isaac_jobs() -> None:
    sql = migration_text()
    gpu_clause = "type not in ('start_training', 'record_video', 'export_onnx', 'export_validate_deploy')"
    assert gpu_clause in sql
    assert "order by case when type = 'stop_process' then 0 else 1 end" in sql
    assert "'export_video_drive'" in sql
    assert "'validate_deploy'" in sql
    assert "'mujoco_smoke'" in sql
