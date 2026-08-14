-- RedRHex To Go 3.7.0 remote-parity additive migration.
-- Existing 3.4.10 rows remain valid; this migration rewrites or deletes no data.

alter table public.machines add column if not exists remote_protocol_version text;

create table if not exists public.machine_capabilities (
  machine_id text primary key references public.machines(machine_id) on delete cascade,
  protocol_version text not null,
  feature_flags jsonb not null default '{}',
  training_routes jsonb not null default '[]',
  physics jsonb not null default '{}',
  deploy jsonb not null default '{}',
  detection jsonb not null default '{}',
  integration_readiness jsonb not null default '{}',
  updated_at timestamptz not null default now()
);

create table if not exists public.physics_presets (
  id text primary key,
  name text not null,
  description text not null default '',
  values jsonb not null default '{}',
  built_in boolean not null default false,
  created_by uuid references auth.users(id),
  updated_by uuid references auth.users(id),
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now()
);

insert into public.physics_presets (id, name, description, values, built_in)
values ('baseline', 'Baseline', 'Repository defaults; sparse overrides are empty.', '{}'::jsonb, true)
on conflict (id) do nothing;

alter table public.jobs add column if not exists client_request_id text;

alter table public.runs add column if not exists progress jsonb not null default '{}';
alter table public.runs add column if not exists metrics jsonb not null default '{}';
alter table public.runs add column if not exists git_provenance jsonb not null default '{}';
alter table public.runs add column if not exists effective_spring_backend text;
alter table public.runs add column if not exists divergence_detected boolean;
alter table public.runs add column if not exists divergence_iteration integer;
alter table public.runs add column if not exists divergence_kind text;
alter table public.runs add column if not exists divergence_reason text;
alter table public.runs add column if not exists deploy_state jsonb not null default '{}';
alter table public.runs add column if not exists mujoco_state jsonb not null default '{}';
alter table public.runs add column if not exists google_drive_video_exports jsonb not null default '[]';

alter table public.team_activity_events add column if not exists source_type text;
alter table public.team_activity_events add column if not exists source_id text;

create unique index if not exists idx_jobs_machine_actor_client_request
  on public.jobs(machine_id, actor_id, client_request_id)
  where client_request_id is not null;
create unique index if not exists idx_team_activity_source
  on public.team_activity_events(machine_id, source_type, source_id)
  where source_type is not null and source_id is not null;
create index if not exists idx_physics_presets_builtin on public.physics_presets(built_in);

drop trigger if exists set_machine_capabilities_updated_at on public.machine_capabilities;
create trigger set_machine_capabilities_updated_at
  before update on public.machine_capabilities
  for each row execute function public.set_redrhex_updated_at();

drop trigger if exists set_physics_presets_updated_at on public.physics_presets;
create trigger set_physics_presets_updated_at
  before update on public.physics_presets
  for each row execute function public.set_redrhex_updated_at();

create or replace function public.authorize_redrhex_job()
returns trigger
language plpgsql
security definer
set search_path = public
as $$
declare
  authoritative_role public.redrhex_role;
begin
  if auth.uid() is null then
    return new;
  end if;
  select role into authoritative_role from public.profiles where id = auth.uid();
  if authoritative_role is null or authoritative_role = 'viewer' then
    raise exception 'Your RedRHex role cannot create remote jobs';
  end if;
  if new.type = 'delete_run' and authoritative_role <> 'admin' then
    raise exception 'Only an admin can delete runs';
  end if;
  if new.type not in (
    'start_training', 'stop_process', 'record_video', 'export_onnx', 'tensorboard',
    'compact_run', 'delete_run', 'send_missed_notifications', 'export_video_drive',
    'validate_deploy', 'export_validate_deploy', 'mujoco_smoke', 'record_mujoco_video'
  ) then
    raise exception 'Unsupported RedRHex remote job type: %', new.type;
  end if;
  new.actor_id := auth.uid();
  new.actor_role := authoritative_role;
  new.client_request_id := nullif(coalesce(new.client_request_id, new.payload ->> 'client_request_id'), '');
  return new;
end;
$$;

drop trigger if exists authorize_redrhex_job on public.jobs;
create trigger authorize_redrhex_job
  before insert on public.jobs
  for each row execute function public.authorize_redrhex_job();

create or replace function public.audit_redrhex_shared_mutation()
returns trigger
language plpgsql
security definer
set search_path = public
as $$
declare
  actor public.profiles%rowtype;
  record_data jsonb;
  record_id text;
begin
  if auth.uid() is null then
    if tg_op = 'DELETE' then
      return old;
    end if;
    return new;
  end if;
  select * into actor from public.profiles where id = auth.uid();
  if actor.id is null then
    raise exception 'Authenticated profile required';
  end if;
  record_data := case when tg_op = 'DELETE' then to_jsonb(old) else to_jsonb(new) end;
  record_id := coalesce(record_data ->> 'id', record_data ->> 'folder_key', 'unknown');
  insert into public.team_activity_events (
    machine_id, actor_id, actor_name, actor_role, event_type, category, outcome,
    points, metadata, source_type, source_id
  ) values (
    nullif(record_data ->> 'machine_id', ''), actor.id,
    coalesce(actor.display_name, actor.email), actor.role,
    tg_table_name || '_' || lower(tg_op), 'organization', 'completed', 1,
    jsonb_build_object(
      'resource', tg_table_name,
      'resource_id', record_id,
      'name', record_data ->> 'name'
    ),
    'database_mutation',
    tg_table_name || ':' || record_id || ':' || lower(tg_op) || ':' || gen_random_uuid()::text
  );
  if tg_op = 'DELETE' then
    return old;
  end if;
  return new;
end;
$$;

drop trigger if exists audit_reward_presets_mutation on public.reward_presets;
create trigger audit_reward_presets_mutation
  after insert or update or delete on public.reward_presets
  for each row execute function public.audit_redrhex_shared_mutation();
drop trigger if exists audit_terrain_presets_mutation on public.terrain_presets;
create trigger audit_terrain_presets_mutation
  after insert or update or delete on public.terrain_presets
  for each row execute function public.audit_redrhex_shared_mutation();
drop trigger if exists audit_physics_presets_mutation on public.physics_presets;
create trigger audit_physics_presets_mutation
  after insert or update or delete on public.physics_presets
  for each row execute function public.audit_redrhex_shared_mutation();
drop trigger if exists audit_team_folders_mutation on public.team_folders;
create trigger audit_team_folders_mutation
  after insert or update or delete on public.team_folders
  for each row execute function public.audit_redrhex_shared_mutation();

create or replace function public.update_run_metadata(
  p_run_id text,
  p_display_name text default null,
  p_notes text default null,
  p_folder text default null
)
returns setof public.runs
language plpgsql
security definer
set search_path = public
as $$
declare
  actor public.profiles%rowtype;
begin
  select * into actor from public.profiles where id = auth.uid();
  if actor.id is null or actor.role not in ('operator', 'admin') then
    raise exception 'Operator or admin role required';
  end if;
  if length(coalesce(p_display_name, '')) > 120 or length(coalesce(p_folder, '')) > 120 or length(coalesce(p_notes, '')) > 10000 then
    raise exception 'Run metadata exceeds the allowed length';
  end if;
  return query
  update public.runs
  set display_name = nullif(trim(p_display_name), ''),
      notes = coalesce(p_notes, ''),
      folder = nullif(trim(p_folder), ''),
      updated_at = now()
  where id = p_run_id
  returning *;
  insert into public.team_activity_events (
    machine_id, actor_id, actor_name, actor_role, event_type, category, outcome,
    run_id, points, metadata, source_type, source_id
  )
  select machine_id, actor.id, coalesce(actor.display_name, actor.email), actor.role,
         'run_metadata_updated', 'organization', 'completed', id, 1,
         jsonb_build_object('display_name', display_name, 'folder', folder),
         'metadata_rpc', auth.uid()::text || ':' || id || ':' || extract(epoch from now())::text
  from public.runs where id = p_run_id;
end;
$$;

create or replace function public.cancel_queued_job(p_job_id uuid)
returns setof public.jobs
language plpgsql
security definer
set search_path = public
as $$
declare
  authoritative_role public.redrhex_role;
  cancelled_job public.jobs%rowtype;
  actor public.profiles%rowtype;
begin
  select * into actor from public.profiles where id = auth.uid();
  authoritative_role := actor.role;
  if authoritative_role is null then
    raise exception 'Authenticated profile required';
  end if;
  update public.jobs
  set status = 'cancelled', updated_at = now()
  where id = p_job_id
    and status = 'queued'
    and (actor_id = auth.uid() or authoritative_role = 'admin')
  returning * into cancelled_job;
  if cancelled_job.id is null then
    return;
  end if;
  insert into public.team_activity_events (
    machine_id, actor_id, actor_name, actor_role, event_type, category, outcome,
    job_id, points, metadata, source_type, source_id
  ) values (
    cancelled_job.machine_id, actor.id, coalesce(actor.display_name, actor.email), actor.role,
    'queued_job_cancelled', 'organization', 'completed', cancelled_job.id, 1,
    jsonb_build_object('job_type', cancelled_job.type), 'cancellation_rpc', cancelled_job.id::text
  );
  return next cancelled_job;
end;
$$;

create or replace function public.claim_next_job_for_machine(p_machine_id text, p_gpu_locked boolean default false)
returns setof public.jobs
language plpgsql
security definer
set search_path = public
as $$
begin
  return query
  with next_job as (
    select id
    from public.jobs
    where status = 'queued'
      and (machine_id = p_machine_id or machine_id is null)
      and (
        not p_gpu_locked
        or type not in ('start_training', 'record_video', 'export_onnx', 'export_validate_deploy')
      )
    order by case when type = 'stop_process' then 0 else 1 end, created_at asc
    for update skip locked
    limit 1
  )
  update public.jobs j
  set status = 'claimed', claimed_by = p_machine_id, claimed_at = now(), updated_at = now()
  from next_job
  where j.id = next_job.id
  returning j.*;
end;
$$;

revoke all on function public.update_run_metadata(text, text, text, text) from public;
grant execute on function public.update_run_metadata(text, text, text, text) to authenticated;
revoke all on function public.cancel_queued_job(uuid) from public;
grant execute on function public.cancel_queued_job(uuid) to authenticated;
revoke all on function public.audit_redrhex_shared_mutation() from public;

alter table public.machine_capabilities enable row level security;
alter table public.physics_presets enable row level security;

drop policy if exists "operators can update remote run metadata" on public.runs;
drop policy if exists "operators can create jobs" on public.jobs;
drop policy if exists "operators can create authorized jobs" on public.jobs;
create policy "operators can create authorized jobs" on public.jobs
  for insert to authenticated
  with check (actor_id = auth.uid() and exists (
    select 1 from public.profiles p where p.id = auth.uid() and p.role in ('operator', 'admin')
  ));

drop policy if exists "machine capabilities readable by authenticated users" on public.machine_capabilities;
drop policy if exists "machine can upsert own capabilities" on public.machine_capabilities;
create policy "machine capabilities readable by authenticated users" on public.machine_capabilities
  for select to authenticated using (true);
create policy "machine can upsert own capabilities" on public.machine_capabilities
  for all using (machine_id = (auth.jwt() ->> 'sub'))
  with check (machine_id = (auth.jwt() ->> 'sub'));

drop policy if exists "physics presets readable by authenticated users" on public.physics_presets;
drop policy if exists "operators can create physics presets" on public.physics_presets;
drop policy if exists "operators can update custom physics presets" on public.physics_presets;
drop policy if exists "operators can delete custom physics presets" on public.physics_presets;
create policy "physics presets readable by authenticated users" on public.physics_presets
  for select to authenticated using (true);
create policy "operators can create physics presets" on public.physics_presets
  for insert to authenticated with check (
    built_in = false and created_by = auth.uid() and exists (
      select 1 from public.profiles p where p.id = auth.uid() and p.role in ('operator', 'admin')
    )
  );
create policy "operators can update custom physics presets" on public.physics_presets
  for update to authenticated using (
    built_in = false and exists (
      select 1 from public.profiles p where p.id = auth.uid() and p.role in ('operator', 'admin')
    )
  ) with check (built_in = false);
create policy "operators can delete custom physics presets" on public.physics_presets
  for delete to authenticated using (
    built_in = false and exists (
      select 1 from public.profiles p where p.id = auth.uid() and p.role in ('operator', 'admin')
    )
  );

do $$
begin
  alter publication supabase_realtime add table public.machine_capabilities;
exception when duplicate_object or undefined_object then null;
end $$;
