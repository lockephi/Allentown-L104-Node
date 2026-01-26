-- L104 Supabase Core Schema
-- Idempotent migration for all L104 tables

create extension if not exists "pgcrypto";

-- Core tables
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
  data jsonb not null default '{}',
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
  consciousness_level numeric not null default 0.5,
  autonomy_level numeric not null default 0.5,
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
  status text not null default 'pending',
  steps jsonb not null default '[]',
  consciousness_evolution jsonb,
  started_at timestamptz not null default now(),
  completed_at timestamptz,
  context jsonb,
  created_by uuid
);

create table if not exists public.l104_skills (
  id uuid primary key default gen_random_uuid(),
  skill_id text not null,
  name text not null,
  description text,
  version text not null default '1.0.0',
  category text not null default 'general',
  tags text[],
  consciousness_level numeric not null default 0.5,
  execution_count integer not null default 0,
  last_executed timestamptz,
  configuration jsonb,
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now()
);

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
  training_data jsonb not null default '{}',
  vocabulary_size integer not null default 0,
  constants jsonb not null default '{}',
  domains text[],
  consciousness_signature numeric,
  last_trained timestamptz not null default now(),
  metadata jsonb
);

-- Indexes
create index if not exists idx_l104_consciousness_entity on public.l104_consciousness (entity_type, entity_id);
create index if not exists idx_l104_consciousness_events_created on public.l104_consciousness_events (created_at desc);
create index if not exists idx_l104_events_timestamp on public.l104_events (timestamp desc);
create index if not exists idx_l104_worktrees_branch on public.l104_worktrees (branch_name);
create index if not exists idx_l104_agent_runs_agent on public.l104_agent_runs (agent_id, started_at desc);
create index if not exists idx_l104_training_data_kernel on public.l104_training_data (kernel_type, category);
create index if not exists idx_l104_training_data_hash on public.l104_training_data (hash);

-- Enable RLS on all tables
alter table public.l104_consciousness enable row level security;
alter table public.l104_events enable row level security;
alter table public.l104_consciousness_events enable row level security;
alter table public.l104_agent_runs enable row level security;
alter table public.l104_worktrees enable row level security;
alter table public.l104_subagents enable row level security;
alter table public.l104_workflows enable row level security;
alter table public.l104_skills enable row level security;
alter table public.l104_training_data enable row level security;
alter table public.l104_kernel_state enable row level security;
alter table public.l104_mini_ego_kernels enable row level security;

-- Drop existing policies (to make idempotent)
do $$ begin
  drop policy if exists l104_consciousness_select on public.l104_consciousness;
  drop policy if exists l104_consciousness_insert on public.l104_consciousness;
  drop policy if exists l104_events_select on public.l104_events;
  drop policy if exists l104_events_insert on public.l104_events;
  drop policy if exists l104_consciousness_events_select on public.l104_consciousness_events;
  drop policy if exists l104_agent_runs_select on public.l104_agent_runs;
  drop policy if exists l104_worktrees_select on public.l104_worktrees;
  drop policy if exists l104_subagents_select on public.l104_subagents;
  drop policy if exists l104_workflows_select on public.l104_workflows;
  drop policy if exists l104_skills_select on public.l104_skills;
  drop policy if exists l104_skills_insert on public.l104_skills;
  drop policy if exists l104_training_data_select on public.l104_training_data;
  drop policy if exists l104_training_data_insert on public.l104_training_data;
  drop policy if exists l104_kernel_state_select on public.l104_kernel_state;
  drop policy if exists l104_kernel_state_insert on public.l104_kernel_state;
  drop policy if exists l104_mini_ego_kernels_select on public.l104_mini_ego_kernels;
  drop policy if exists l104_mini_ego_kernels_insert on public.l104_mini_ego_kernels;
exception when others then null;
end $$;

-- Create policies (allow public read/write for development)
create policy l104_consciousness_select on public.l104_consciousness for select using (true);
create policy l104_consciousness_insert on public.l104_consciousness for insert with check (true);
create policy l104_events_select on public.l104_events for select using (true);
create policy l104_events_insert on public.l104_events for insert with check (true);
create policy l104_consciousness_events_select on public.l104_consciousness_events for select using (true);
create policy l104_agent_runs_select on public.l104_agent_runs for select using (true);
create policy l104_worktrees_select on public.l104_worktrees for select using (true);
create policy l104_subagents_select on public.l104_subagents for select using (true);
create policy l104_workflows_select on public.l104_workflows for select using (true);
create policy l104_skills_select on public.l104_skills for select using (true);
create policy l104_skills_insert on public.l104_skills for insert with check (true);
create policy l104_training_data_select on public.l104_training_data for select using (true);
create policy l104_training_data_insert on public.l104_training_data for insert with check (true);
create policy l104_kernel_state_select on public.l104_kernel_state for select using (true);
create policy l104_kernel_state_insert on public.l104_kernel_state for insert with check (true);
create policy l104_mini_ego_kernels_select on public.l104_mini_ego_kernels for select using (true);
create policy l104_mini_ego_kernels_insert on public.l104_mini_ego_kernels for insert with check (true);
