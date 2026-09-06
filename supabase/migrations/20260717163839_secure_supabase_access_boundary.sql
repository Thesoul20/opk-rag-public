-- Tighten Supabase Data API exposure for the RAG core schema.
--
-- The ingestion/indexing path uses direct PostgreSQL connections. Public Data API
-- access stays opt-in through explicitly granted RPC functions.

alter table public.knowledge_bases enable row level security;
alter table public.index_configurations enable row level security;
alter table public.documents enable row level security;
alter table public.chunks enable row level security;
alter table public.index_runs enable row level security;
alter table public.index_failures enable row level security;

revoke all on table
  public.knowledge_bases,
  public.index_configurations,
  public.documents,
  public.chunks,
  public.index_runs,
  public.index_failures
from public;

do $$
begin
  if exists (select 1 from pg_roles where rolname = 'anon') then
    revoke all on table
      public.knowledge_bases,
      public.index_configurations,
      public.documents,
      public.chunks,
      public.index_runs,
      public.index_failures
    from anon;
  end if;

  if exists (select 1 from pg_roles where rolname = 'authenticated') then
    revoke all on table
      public.knowledge_bases,
      public.index_configurations,
      public.documents,
      public.chunks,
      public.index_runs,
      public.index_failures
    from authenticated;
  end if;

  if exists (select 1 from pg_roles where rolname = 'service_role') then
    grant usage on schema public to service_role;
    grant select on table public.documents, public.chunks to service_role;
  end if;
end
$$;

create or replace function public.set_updated_at()
returns trigger
language plpgsql
security invoker
set search_path = ''
as $$
begin
  new.updated_at = now();
  return new;
end;
$$;

revoke execute on function public.set_updated_at() from public;

do $$
begin
  if exists (select 1 from pg_roles where rolname = 'anon') then
    revoke execute on function public.set_updated_at() from anon;
  end if;

  if exists (select 1 from pg_roles where rolname = 'authenticated') then
    revoke execute on function public.set_updated_at() from authenticated;
  end if;
end
$$;

create index if not exists chunks_index_configuration_id_idx
  on public.chunks (index_configuration_id);

create index if not exists index_runs_index_configuration_id_idx
  on public.index_runs (index_configuration_id);

create or replace function public.match_chunks(
  p_query_embedding extensions.vector(1024),
  p_knowledge_base_id uuid,
  p_match_count integer default 10,
  p_index_configuration_id uuid default null
)
returns table (
  document_id uuid,
  chunk_id uuid,
  relative_path text,
  heading_path text[],
  content text,
  start_line integer,
  end_line integer,
  similarity double precision
)
language sql
stable
security invoker
set search_path = ''
as $$
  select
    d.id as document_id,
    c.id as chunk_id,
    d.relative_path,
    c.heading_path,
    c.content,
    c.start_line,
    c.end_line,
    (1 - (c.embedding operator(extensions.<=>) p_query_embedding))::double precision as similarity
  from public.chunks c
  join public.documents d on d.id = c.document_id
  where d.knowledge_base_id = p_knowledge_base_id
    and d.index_status = 'indexed'
    and (
      p_index_configuration_id is null
      or c.index_configuration_id = p_index_configuration_id
    )
  order by c.embedding operator(extensions.<=>) p_query_embedding
  limit least(greatest(coalesce(p_match_count, 10), 1), 100)
$$;

revoke execute on function public.match_chunks(extensions.vector, uuid, integer, uuid) from public;

do $$
begin
  if exists (select 1 from pg_roles where rolname = 'anon') then
    revoke execute on function public.match_chunks(extensions.vector, uuid, integer, uuid) from anon;
  end if;

  if exists (select 1 from pg_roles where rolname = 'authenticated') then
    revoke execute on function public.match_chunks(extensions.vector, uuid, integer, uuid) from authenticated;
  end if;

  if exists (select 1 from pg_roles where rolname = 'service_role') then
    grant execute on function public.match_chunks(extensions.vector, uuid, integer, uuid) to service_role;
  end if;
end
$$;
