create table public.conversation_sessions (
  id uuid primary key default gen_random_uuid(),
  knowledge_base_id uuid not null references public.knowledge_bases(id) on delete cascade,
  title text,
  status text not null default 'active',
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now(),
  last_turn_at timestamptz,
  turn_count integer not null default 0,
  conversation_prompt_version text not null,
  followup_rewrite_version text not null,
  metadata jsonb not null default '{}'::jsonb,
  constraint conversation_sessions_status_check check (status in ('active', 'archived', 'deleted')),
  constraint conversation_sessions_turn_count_nonnegative check (turn_count >= 0),
  constraint conversation_sessions_title_length check (title is null or length(title) <= 200),
  constraint conversation_sessions_conversation_prompt_version_nonempty check (length(btrim(conversation_prompt_version)) > 0),
  constraint conversation_sessions_followup_rewrite_version_nonempty check (length(btrim(followup_rewrite_version)) > 0)
);

create table public.conversation_turns (
  id uuid primary key default gen_random_uuid(),
  session_id uuid not null references public.conversation_sessions(id) on delete cascade,
  turn_number integer not null,
  client_request_id text,
  user_query text not null,
  standalone_query text,
  rewrite_status text not null default 'not_needed',
  answer_decision text,
  answer_text text,
  abstention_reason text,
  turn_status text not null default 'pending',
  retrieval_mode text,
  reranking_enabled boolean,
  provider_id text,
  model_id text,
  model_version text,
  prompt_fingerprint text,
  input_token_count integer,
  output_token_count integer,
  latency_ms integer,
  error_code text,
  created_at timestamptz not null default now(),
  completed_at timestamptz,
  metadata jsonb not null default '{}'::jsonb,
  constraint conversation_turns_session_turn_number_key unique (session_id, turn_number),
  constraint conversation_turns_session_client_request_key unique (session_id, client_request_id),
  constraint conversation_turns_turn_number_positive check (turn_number > 0),
  constraint conversation_turns_user_query_nonempty check (length(btrim(user_query)) > 0),
  constraint conversation_turns_client_request_id_nonempty check (client_request_id is null or length(btrim(client_request_id)) > 0),
  constraint conversation_turns_rewrite_status_check check (rewrite_status in ('not_needed', 'resolved', 'failed')),
  constraint conversation_turns_answer_decision_check check (answer_decision is null or answer_decision in ('answer', 'abstain', 'system_error')),
  constraint conversation_turns_turn_status_check check (turn_status in ('pending', 'rewriting', 'retrieving', 'generating', 'completed', 'abstained', 'failed')),
  constraint conversation_turns_completed_at_check check (
    (turn_status in ('completed', 'abstained', 'failed') and completed_at is not null)
    or (turn_status in ('pending', 'rewriting', 'retrieving', 'generating') and completed_at is null)
  ),
  constraint conversation_turns_failed_error_check check (turn_status <> 'failed' or error_code is not null),
  constraint conversation_turns_token_counts_nonnegative check (
    (input_token_count is null or input_token_count >= 0)
    and (output_token_count is null or output_token_count >= 0)
  ),
  constraint conversation_turns_latency_nonnegative check (latency_ms is null or latency_ms >= 0)
);

create table public.conversation_turn_citations (
  turn_id uuid not null references public.conversation_turns(id) on delete cascade,
  citation_id text not null,
  document_id uuid not null,
  chunk_id uuid not null,
  relative_path text not null,
  heading_path jsonb,
  start_line integer,
  end_line integer,
  content_hash text not null,
  context_rank integer not null,
  created_at timestamptz not null default now(),
  primary key (turn_id, citation_id),
  constraint conversation_turn_citations_citation_id_check check (citation_id ~ '^C[1-9][0-9]*$'),
  constraint conversation_turn_citations_relative_path_nonempty check (length(btrim(relative_path)) > 0),
  constraint conversation_turn_citations_content_hash_nonempty check (length(btrim(content_hash)) > 0),
  constraint conversation_turn_citations_context_rank_positive check (context_rank > 0),
  constraint conversation_turn_citations_start_line_positive check (start_line is null or start_line > 0),
  constraint conversation_turn_citations_end_line_valid check (end_line is null or (start_line is not null and end_line >= start_line)),
  constraint conversation_turn_citations_heading_path_array check (heading_path is null or jsonb_typeof(heading_path) = 'array')
);

create index conversation_sessions_knowledge_base_id_idx on public.conversation_sessions (knowledge_base_id);
create index conversation_sessions_status_idx on public.conversation_sessions (status);
create index conversation_turns_session_id_idx on public.conversation_turns (session_id);
create index conversation_turns_status_idx on public.conversation_turns (turn_status);
create index conversation_turn_citations_document_id_idx on public.conversation_turn_citations (document_id);
create index conversation_turn_citations_chunk_id_idx on public.conversation_turn_citations (chunk_id);

create trigger set_conversation_sessions_updated_at
before update on public.conversation_sessions
for each row
execute function public.set_updated_at();

alter table public.conversation_sessions enable row level security;
alter table public.conversation_turns enable row level security;
alter table public.conversation_turn_citations enable row level security;

revoke all on table
  public.conversation_sessions,
  public.conversation_turns,
  public.conversation_turn_citations
from public;

do $$
begin
  if exists (select 1 from pg_roles where rolname = 'anon') then
    revoke all on table
      public.conversation_sessions,
      public.conversation_turns,
      public.conversation_turn_citations
    from anon;
  end if;

  if exists (select 1 from pg_roles where rolname = 'authenticated') then
    revoke all on table
      public.conversation_sessions,
      public.conversation_turns,
      public.conversation_turn_citations
    from authenticated;
  end if;

  if exists (select 1 from pg_roles where rolname = 'service_role') then
    grant usage on schema public to service_role;
    grant select, insert, update, delete on table
      public.conversation_sessions,
      public.conversation_turns,
      public.conversation_turn_citations
    to service_role;
  end if;
end
$$;

comment on table public.conversation_sessions is 'Conversation sessions scoped to a single knowledge base.';
comment on table public.conversation_turns is 'Linear user/assistant turns with standalone query resolution metadata.';
comment on table public.conversation_turn_citations is 'Current-turn citation metadata and source content hashes.';
