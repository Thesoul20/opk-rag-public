-- Allow TASK-0005 to persist chunks before TASK-0006 generates real embeddings.
--
-- Chunking is not allowed to simulate embedding vectors. Retrieval must only see
-- chunks that have a real embedding and whose document has reached indexed.

alter table public.chunks
  alter column embedding drop not null,
  alter column embedding_model drop not null,
  alter column embedding_dimension drop not null;

alter table public.chunks
  drop constraint if exists chunks_embedding_model_nonempty,
  drop constraint if exists chunks_embedding_dimension_check;

alter table public.chunks
  add constraint chunks_embedding_model_nonempty
    check (embedding_model is null or length(btrim(embedding_model)) > 0),
  add constraint chunks_embedding_dimension_check
    check (embedding_dimension is null or embedding_dimension = 1024);

alter table public.documents
  drop constraint if exists documents_index_status_check;

alter table public.documents
  add constraint documents_index_status_check
    check (index_status in ('pending', 'processing', 'chunked', 'indexed', 'failed', 'deleted'));

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
    and c.embedding is not null
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
