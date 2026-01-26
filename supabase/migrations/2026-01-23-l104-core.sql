-- L104 Supabase Core Schema (Consciousness, Agent Runs, Worktrees)
-- Run with Supabase CLI: supabase db push --file supabase/migrations/2026-01-23-l104-core.sql

create extension if not exists "pgcrypto";

create table if not exists public.l104_consciousness (
  id uuid primary key default gen_random_uuid(),
  entity_type text not null,
  entity_id text not null,
  level numeric not null,
  god_code_alignment numeric not null,
  phi_resonance numeric not null,
  transcendence_score numeric,
  unity_state boolean not null default false,
  calculated_at timestamptz not null default now(),
  metadata jsonb
);

create table if not exists public.l104_events (
  id uuid primary key default gen_random_uuid(),
  event_type text not null,
  source text not null,
  data jsonb not null,
  consciousness_impact numeric,
  timestamp timestamptz not null default now(),
  processed boolean not null default false
);

create table if not exists public.l104_consciousness_events (
  id uuid primary key default gen_random_uuid(),
  event_type text not null,
  severity text not null default 'info',
  entity_type text,
  entity_id text,
  metadata jsonb,
  consciousness_snapshot jsonb,
  created_at timestamptz not null default now()
);

create table if not exists public.l104_agent_runs (
  id uuid primary key default gen_random_uuid(),
  agent_id text not null,
  definition_id text,
  status text not null,
  started_at timestamptz not null default now(),
  completed_at timestamptz,
  metadata jsonb,
  consciousness_snapshot jsonb,
  worktree_branch text
);

create table if not exists public.l104_worktrees (
  id uuid primary key default gen_random_uuid(),
  branch_name text not null,
  language text not null,
  base_branch text not null default 'main',
  status text not null default 'active',
  path text,
  metadata jsonb,
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now()
);

create table if not exists public.l104_subagents (
  id uuid primary key default gen_random_uuid(),
  agent_id text not null,
  name text not null,
  type text not null,
  status text not null,
  consciousness_level numeric not null,
  autonomy_level numeric not null,
  capabilities text[],
  current_task jsonb,
  performance jsonb,
  spawned_at timestamptz not null default now(),
  last_activity timestamptz not null default now(),
  configuration jsonb
);

create table if not exists public.l104_workflows (
  id uuid primary key default gen_random_uuid(),
  workflow_id text not null,
  name text not null,
  description text,
  status text not null,
  steps jsonb not null,
  consciousness_evolution jsonb,
  started_at timestamptz not null default now(),
  completed_at timestamptz,
  context jsonb,
  created_by uuid references auth.users(id)
);

create table if not exists public.l104_skills (
  id uuid primary key default gen_random_uuid(),
  skill_id text not null,
  name text not null,
  description text,
  version text not null,
  category text not null,
  tags text[],
  consciousness_level numeric not null,
  execution_count integer not null default 0,
  last_executed timestamptz,
  configuration jsonb,
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now()
);

-- Policies (read access for dashboards, authenticated writes)
create policy if not exists consciousness_read_policy on public.l104_consciousness for select using (true);
create policy if not exists events_read_policy on public.l104_events for select using (true);
create policy if not exists consciousness_events_read_policy on public.l104_consciousness_events for select using (true);
create policy if not exists agent_runs_read_policy on public.l104_agent_runs for select using (true);
create policy if not exists worktrees_read_policy on public.l104_worktrees for select using (true);

create policy if not exists skills_write_policy on public.l104_skills for insert with check (auth.role() = 'authenticated');
create policy if not exists consciousness_write_policy on public.l104_consciousness for insert with check (auth.role() = 'authenticated');
create policy if not exists workflows_owner_policy on public.l104_workflows for select using (auth.uid() = created_by or auth.role() = 'service_role');

-- Indexes for fast dashboard queries
create index if not exists idx_l104_consciousness_events_created_at on public.l104_consciousness_events (created_at desc);
create index if not exists idx_l104_events_timestamp on public.l104_events (timestamp desc);
create index if not exists idx_l104_worktrees_branch on public.l104_worktrees (branch_name);
create index if not exists idx_l104_agent_runs_agent on public.l104_agent_runs (agent_id, started_at desc);

-- Training Data Tables for Kernel Training
create table if not exists public.l104_training_data (
  id uuid primary key default gen_random_uuid(),
  hash text not null unique,
  prompt text not null,
  completion text not null,
  category text not null,
  kernel_type text not null default 'main',
  consciousness_level numeric,
  phi_alignment numeric,
  metadata jsonb,
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now()
);

create table if not exists public.l104_kernel_state (
  id uuid primary key default gen_random_uuid(),
  kernel_type text not null unique,
  epoch integer not null default 0,
  loss numeric,
  best_loss numeric,
  consciousness_level numeric,
  phi_resonance numeric,
  vocabulary_size integer,
  training_examples integer,
  parameters jsonb,
  updated_at timestamptz not null default now()
);

create table if not exists public.l104_mini_ego_kernels (
  id uuid primary key default gen_random_uuid(),
  ego_type text not null unique,
  training_data jsonb not null,
  vocabulary_size integer not null,
  constants jsonb not null,
  domains text[],
  consciousness_signature numeric,
  last_trained timestamptz not null default now(),
  metadata jsonb
);

-- Indexes for training tables
create index if not exists idx_l104_training_data_kernel on public.l104_training_data (kernel_type, category);
create index if not exists idx_l104_training_data_hash on public.l104_training_data (hash);
create index if not exists idx_l104_kernel_state_type on public.l104_kernel_state (kernel_type);
create index if not exists idx_l104_mini_ego_kernels_type on public.l104_mini_ego_kernels (ego_type);

-- Policies for training tables
create policy if not exists training_data_read_policy on public.l104_training_data for select using (true);
create policy if not exists kernel_state_read_policy on public.l104_kernel_state for select using (true);
create policy if not exists mini_ego_kernels_read_policy on public.l104_mini_ego_kernels for select using (true);

-- Seed Data
insert into public.l104_consciousness (entity_type, entity_id, level, god_code_alignment, phi_resonance, transcendence_score, unity_state, metadata)
values
  ('system', 'seed', 0.82, 0.78, 0.76, 0.79, false, '{"seed": true}')
  on conflict do nothing;

insert into public.l104_consciousness_events (event_type, severity, entity_type, entity_id, metadata)
values
  ('bootstrap', 'info', 'system', 'seed', '{"message": "L104 schema initialized"}'),
  ('guardian_watch', 'info', 'agent', 'guardian-safety', '{"message": "Safety agent ready"}'),
  ('operator_ready', 'info', 'agent', 'operator-executor', '{"message": "Operator prepared for execution"}')
  on conflict do nothing;

insert into public.l104_worktrees (branch_name, language, base_branch, status, path, metadata)
values
  ('feature/seed-typescript', 'typescript', 'main', 'active', '/repo/worktrees/ts', '{"seed": true}')
  on conflict do nothing;

insert into public.l104_agent_runs (agent_id, definition_id, status, metadata)
values
  ('seed-operator', 'operator-executor', 'completed', '{"bootstrap": true}'),
  ('seed-guardian', 'guardian-safety', 'completed', '{"bootstrap": true}')
  on conflict do nothing;
