create schema if not exists extensions;

create extension if not exists vector with schema extensions;

create or replace function public.set_updated_at()
returns trigger
language plpgsql
as $$
begin
  new.updated_at = now();
  return new;
end;
$$;

create table public.knowledge_bases (
  id uuid primary key default gen_random_uuid(),
  name text not null,
  root_path text not null,
  description text,
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now(),
  constraint knowledge_bases_root_path_key unique (root_path),
  constraint knowledge_bases_name_nonempty check (length(btrim(name)) > 0),
  constraint knowledge_bases_root_path_nonempty check (length(btrim(root_path)) > 0)
);

create table public.index_configurations (
  id uuid primary key default gen_random_uuid(),
  schema_version text not null,
  parser_version text not null,
  chunking_version text not null,
  embedding_provider text not null,
  embedding_model text not null,
  model_revision text,
  embedding_dimension integer not null,
  normalize boolean not null default true,
  distance_metric text not null,
  configuration_fingerprint text not null,
  created_at timestamptz not null default now(),
  constraint index_configurations_embedding_dimension_check check (embedding_dimension = 1024),
  constraint index_configurations_distance_metric_check check (distance_metric in ('cosine')),
  constraint index_configurations_fingerprint_key unique (configuration_fingerprint),
  constraint index_configurations_schema_version_nonempty check (length(btrim(schema_version)) > 0),
  constraint index_configurations_parser_version_nonempty check (length(btrim(parser_version)) > 0),
  constraint index_configurations_chunking_version_nonempty check (length(btrim(chunking_version)) > 0),
  constraint index_configurations_embedding_provider_nonempty check (length(btrim(embedding_provider)) > 0),
  constraint index_configurations_embedding_model_nonempty check (length(btrim(embedding_model)) > 0)
);

create table public.documents (
  id uuid primary key default gen_random_uuid(),
  knowledge_base_id uuid not null references public.knowledge_bases(id) on delete cascade,
  relative_path text not null,
  file_name text not null,
  title text,
  content_hash text not null,
  file_size bigint not null,
  source_modified_at timestamptz not null,
  parser_version text not null,
  chunking_version text not null,
  index_status text not null,
  last_indexed_at timestamptz,
  last_error text,
  metadata jsonb not null default '{}'::jsonb,
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now(),
  constraint documents_knowledge_base_relative_path_key unique (knowledge_base_id, relative_path),
  constraint documents_index_status_check check (index_status in ('pending', 'processing', 'indexed', 'failed', 'deleted')),
  constraint documents_relative_path_nonempty check (length(btrim(relative_path)) > 0),
  constraint documents_file_name_nonempty check (length(btrim(file_name)) > 0),
  constraint documents_content_hash_nonempty check (length(btrim(content_hash)) > 0),
  constraint documents_file_size_nonnegative check (file_size >= 0),
  constraint documents_parser_version_nonempty check (length(btrim(parser_version)) > 0),
  constraint documents_chunking_version_nonempty check (length(btrim(chunking_version)) > 0)
);

create table public.chunks (
  id uuid primary key default gen_random_uuid(),
  document_id uuid not null references public.documents(id) on delete cascade,
  index_configuration_id uuid not null references public.index_configurations(id) on delete restrict,
  chunk_index integer not null,
  content text not null,
  content_hash text not null,
  heading_path text[] not null default '{}'::text[],
  start_line integer,
  end_line integer,
  token_count integer,
  embedding extensions.vector(1024) not null,
  embedding_model text not null,
  embedding_dimension integer not null,
  metadata jsonb not null default '{}'::jsonb,
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now(),
  constraint chunks_document_id_chunk_index_key unique (document_id, chunk_index),
  constraint chunks_chunk_index_nonnegative check (chunk_index >= 0),
  constraint chunks_content_nonempty check (length(btrim(content)) > 0),
  constraint chunks_content_hash_nonempty check (length(btrim(content_hash)) > 0),
  constraint chunks_start_line_positive check (start_line is null or start_line > 0),
  constraint chunks_end_line_valid check (end_line is null or (start_line is not null and end_line >= start_line)),
  constraint chunks_token_count_nonnegative check (token_count is null or token_count >= 0),
  constraint chunks_embedding_model_nonempty check (length(btrim(embedding_model)) > 0),
  constraint chunks_embedding_dimension_check check (embedding_dimension = 1024)
);

create table public.index_runs (
  id uuid primary key default gen_random_uuid(),
  knowledge_base_id uuid not null references public.knowledge_bases(id) on delete cascade,
  index_configuration_id uuid references public.index_configurations(id) on delete restrict,
  run_type text not null,
  status text not null,
  started_at timestamptz not null default now(),
  finished_at timestamptz,
  documents_discovered integer not null default 0,
  documents_created integer not null default 0,
  documents_updated integer not null default 0,
  documents_deleted integer not null default 0,
  documents_skipped integer not null default 0,
  documents_failed integer not null default 0,
  chunks_created integer not null default 0,
  chunks_deleted integer not null default 0,
  error_message text,
  metadata jsonb not null default '{}'::jsonb,
  created_at timestamptz not null default now(),
  constraint index_runs_run_type_check check (run_type in ('full', 'incremental')),
  constraint index_runs_status_check check (status in ('running', 'completed', 'failed', 'partial')),
  constraint index_runs_finished_after_started check (finished_at is null or finished_at >= started_at),
  constraint index_runs_documents_discovered_nonnegative check (documents_discovered >= 0),
  constraint index_runs_documents_created_nonnegative check (documents_created >= 0),
  constraint index_runs_documents_updated_nonnegative check (documents_updated >= 0),
  constraint index_runs_documents_deleted_nonnegative check (documents_deleted >= 0),
  constraint index_runs_documents_skipped_nonnegative check (documents_skipped >= 0),
  constraint index_runs_documents_failed_nonnegative check (documents_failed >= 0),
  constraint index_runs_chunks_created_nonnegative check (chunks_created >= 0),
  constraint index_runs_chunks_deleted_nonnegative check (chunks_deleted >= 0)
);

create table public.index_failures (
  id uuid primary key default gen_random_uuid(),
  run_id uuid not null references public.index_runs(id) on delete cascade,
  relative_path text,
  stage text not null,
  error_type text not null,
  error_message text not null,
  retryable boolean not null default false,
  metadata jsonb not null default '{}'::jsonb,
  created_at timestamptz not null default now(),
  constraint index_failures_stage_nonempty check (length(btrim(stage)) > 0),
  constraint index_failures_error_type_nonempty check (length(btrim(error_type)) > 0),
  constraint index_failures_error_message_nonempty check (length(btrim(error_message)) > 0)
);

create index documents_knowledge_base_id_idx on public.documents (knowledge_base_id);
create index documents_index_status_idx on public.documents (index_status);
create index documents_content_hash_idx on public.documents (content_hash);
create index chunks_document_id_idx on public.chunks (document_id);
create index chunks_content_hash_idx on public.chunks (content_hash);
create index chunks_heading_path_idx on public.chunks using gin (heading_path);
create index chunks_embedding_hnsw_idx on public.chunks using hnsw (embedding extensions.vector_cosine_ops);
create index index_runs_knowledge_base_id_idx on public.index_runs (knowledge_base_id);
create index index_runs_status_idx on public.index_runs (status);
create index index_runs_started_at_idx on public.index_runs (started_at);
create index index_failures_run_id_idx on public.index_failures (run_id);
create index index_failures_relative_path_idx on public.index_failures (relative_path);
create index index_failures_stage_idx on public.index_failures (stage);

create trigger set_knowledge_bases_updated_at
before update on public.knowledge_bases
for each row
execute function public.set_updated_at();

create trigger set_documents_updated_at
before update on public.documents
for each row
execute function public.set_updated_at();

create trigger set_chunks_updated_at
before update on public.chunks
for each row
execute function public.set_updated_at();

comment on table public.knowledge_bases is 'Indexed Obsidian vaults or Markdown knowledge bases.';
comment on table public.index_configurations is 'Frozen parser/chunking/embedding configuration fingerprints for index compatibility.';
comment on table public.documents is 'Source Markdown document metadata and index status.';
comment on table public.chunks is 'Structured chunks with line anchors and pgvector embeddings.';
comment on table public.index_runs is 'Audit trail for full or incremental indexing runs.';
comment on table public.index_failures is 'Per-file indexing failures associated with an index run.';
