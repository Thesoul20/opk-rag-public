create table public.chunk_lexical_statistics (
  chunk_id uuid primary key references public.chunks(id) on delete cascade,
  knowledge_base_id uuid not null references public.knowledge_bases(id) on delete cascade,
  document_length integer not null,
  tokenizer_id text not null,
  tokenizer_version text not null,
  index_configuration_fingerprint text not null,
  indexed_at timestamptz not null default now(),
  constraint chunk_lexical_statistics_document_length_positive check (document_length > 0),
  constraint chunk_lexical_statistics_tokenizer_id_nonempty check (length(btrim(tokenizer_id)) > 0),
  constraint chunk_lexical_statistics_tokenizer_version_nonempty check (length(btrim(tokenizer_version)) > 0),
  constraint chunk_lexical_statistics_fingerprint_nonempty check (length(btrim(index_configuration_fingerprint)) > 0)
);

create table public.chunk_lexical_terms (
  chunk_id uuid not null references public.chunks(id) on delete cascade,
  knowledge_base_id uuid not null references public.knowledge_bases(id) on delete cascade,
  term text not null,
  term_frequency integer not null,
  primary key (chunk_id, term),
  constraint chunk_lexical_terms_term_nonempty check (length(btrim(term)) > 0),
  constraint chunk_lexical_terms_frequency_positive check (term_frequency > 0)
);

create table public.knowledge_base_lexical_statistics (
  knowledge_base_id uuid primary key references public.knowledge_bases(id) on delete cascade,
  indexed_chunk_count integer not null,
  total_document_length integer not null,
  average_document_length double precision not null,
  tokenizer_id text not null,
  tokenizer_version text not null,
  index_configuration_fingerprint text not null,
  updated_at timestamptz not null default now(),
  constraint knowledge_base_lexical_statistics_chunk_count_nonnegative check (indexed_chunk_count >= 0),
  constraint knowledge_base_lexical_statistics_total_length_nonnegative check (total_document_length >= 0),
  constraint knowledge_base_lexical_statistics_average_length_nonnegative check (average_document_length >= 0),
  constraint knowledge_base_lexical_statistics_average_length_finite check (
    average_document_length = average_document_length
    and average_document_length < 'Infinity'::double precision
    and average_document_length > '-Infinity'::double precision
  ),
  constraint knowledge_base_lexical_statistics_tokenizer_id_nonempty check (length(btrim(tokenizer_id)) > 0),
  constraint knowledge_base_lexical_statistics_tokenizer_version_nonempty check (length(btrim(tokenizer_version)) > 0),
  constraint knowledge_base_lexical_statistics_fingerprint_nonempty check (length(btrim(index_configuration_fingerprint)) > 0)
);

create table public.knowledge_base_lexical_terms (
  knowledge_base_id uuid not null references public.knowledge_bases(id) on delete cascade,
  term text not null,
  document_frequency integer not null,
  primary key (knowledge_base_id, term),
  constraint knowledge_base_lexical_terms_term_nonempty check (length(btrim(term)) > 0),
  constraint knowledge_base_lexical_terms_frequency_positive check (document_frequency > 0)
);

create index chunk_lexical_statistics_kb_config_idx
  on public.chunk_lexical_statistics (knowledge_base_id, tokenizer_id, tokenizer_version, index_configuration_fingerprint);

create index chunk_lexical_terms_kb_term_idx
  on public.chunk_lexical_terms (knowledge_base_id, term);

create index knowledge_base_lexical_terms_kb_term_df_idx
  on public.knowledge_base_lexical_terms (knowledge_base_id, term, document_frequency);

create index knowledge_base_lexical_statistics_readiness_idx
  on public.knowledge_base_lexical_statistics (knowledge_base_id, tokenizer_id, tokenizer_version, index_configuration_fingerprint);

create or replace function public.enforce_chunk_lexical_knowledge_base()
returns trigger
language plpgsql
security invoker
set search_path = ''
as $$
declare
  actual_knowledge_base_id uuid;
begin
  select d.knowledge_base_id
  into actual_knowledge_base_id
  from public.chunks c
  join public.documents d on d.id = c.document_id
  where c.id = new.chunk_id;

  if actual_knowledge_base_id is null or actual_knowledge_base_id is distinct from new.knowledge_base_id then
    raise exception 'lexical row knowledge_base_id does not match chunk ownership';
  end if;

  return new;
end
$$;

create trigger chunk_lexical_statistics_kb_consistency
before insert or update of chunk_id, knowledge_base_id
on public.chunk_lexical_statistics
for each row execute function public.enforce_chunk_lexical_knowledge_base();

create trigger chunk_lexical_terms_kb_consistency
before insert or update of chunk_id, knowledge_base_id
on public.chunk_lexical_terms
for each row execute function public.enforce_chunk_lexical_knowledge_base();

create or replace function public.refresh_lexical_corpus_after_chunk_stat_delete()
returns trigger
language plpgsql
security invoker
set search_path = ''
as $$
begin
  delete from public.knowledge_base_lexical_terms
  where knowledge_base_id = old.knowledge_base_id;

  insert into public.knowledge_base_lexical_terms (knowledge_base_id, term, document_frequency)
  select clt.knowledge_base_id, clt.term, count(*)::integer
  from public.chunk_lexical_terms clt
  join public.chunk_lexical_statistics cls on cls.chunk_id = clt.chunk_id
  where clt.knowledge_base_id = old.knowledge_base_id
    and cls.tokenizer_id = old.tokenizer_id
    and cls.tokenizer_version = old.tokenizer_version
    and cls.index_configuration_fingerprint = old.index_configuration_fingerprint
  group by clt.knowledge_base_id, clt.term
  having count(*) > 0;

  delete from public.knowledge_base_lexical_statistics
  where knowledge_base_id = old.knowledge_base_id;

  insert into public.knowledge_base_lexical_statistics (
    knowledge_base_id,
    indexed_chunk_count,
    total_document_length,
    average_document_length,
    tokenizer_id,
    tokenizer_version,
    index_configuration_fingerprint,
    updated_at
  )
  select old.knowledge_base_id,
         count(*)::integer,
         coalesce(sum(document_length), 0)::integer,
         coalesce(avg(document_length), 0)::double precision,
         old.tokenizer_id,
         old.tokenizer_version,
         old.index_configuration_fingerprint,
         now()
  from public.chunk_lexical_statistics
  where knowledge_base_id = old.knowledge_base_id
    and tokenizer_id = old.tokenizer_id
    and tokenizer_version = old.tokenizer_version
    and index_configuration_fingerprint = old.index_configuration_fingerprint
  having count(*) > 0;

  return old;
end
$$;

create trigger chunk_lexical_statistics_refresh_after_delete
after delete on public.chunk_lexical_statistics
for each row execute function public.refresh_lexical_corpus_after_chunk_stat_delete();

alter table public.chunk_lexical_statistics enable row level security;
alter table public.chunk_lexical_terms enable row level security;
alter table public.knowledge_base_lexical_statistics enable row level security;
alter table public.knowledge_base_lexical_terms enable row level security;

revoke all on table
  public.chunk_lexical_statistics,
  public.chunk_lexical_terms,
  public.knowledge_base_lexical_statistics,
  public.knowledge_base_lexical_terms
from public;

do $$
begin
  if exists (select 1 from pg_roles where rolname = 'anon') then
    revoke all on table
      public.chunk_lexical_statistics,
      public.chunk_lexical_terms,
      public.knowledge_base_lexical_statistics,
      public.knowledge_base_lexical_terms
    from anon;
  end if;

  if exists (select 1 from pg_roles where rolname = 'authenticated') then
    revoke all on table
      public.chunk_lexical_statistics,
      public.chunk_lexical_terms,
      public.knowledge_base_lexical_statistics,
      public.knowledge_base_lexical_terms
    from authenticated;
  end if;

  if exists (select 1 from pg_roles where rolname = 'service_role') then
    grant select, insert, update, delete on table
      public.chunk_lexical_statistics,
      public.chunk_lexical_terms,
      public.knowledge_base_lexical_statistics,
      public.knowledge_base_lexical_terms
    to service_role;
  end if;
end
$$;

create or replace function public.search_chunks_bm25(
  p_query_terms text[],
  p_knowledge_base_id uuid,
  p_match_count integer default 10,
  p_k1 double precision default 1.2,
  p_b double precision default 0.75,
  p_index_configuration_fingerprint text default null
)
returns table (
  document_id uuid,
  chunk_id uuid,
  relative_path text,
  heading_path text[],
  content text,
  start_line integer,
  end_line integer,
  bm25_score double precision,
  matched_terms text[]
)
language sql
stable
security invoker
set search_path = ''
as $$
  with query_terms as (
    select distinct btrim(term) as term
    from unnest(coalesce(p_query_terms, array[]::text[])) term
    where length(btrim(term)) > 0
  ),
  corpus as (
    select indexed_chunk_count, average_document_length
    from public.knowledge_base_lexical_statistics
    where knowledge_base_id = p_knowledge_base_id
      and (p_index_configuration_fingerprint is null or index_configuration_fingerprint = p_index_configuration_fingerprint)
  ),
  scored_terms as (
    select
      clt.chunk_id,
      sum(
        ln(1 + ((corpus.indexed_chunk_count - kblt.document_frequency + 0.5) / (kblt.document_frequency + 0.5)))
        * (
          clt.term_frequency * (p_k1 + 1)
          / (
            clt.term_frequency
            + p_k1 * (
              1 - p_b
              + p_b * (cls.document_length::double precision / nullif(corpus.average_document_length, 0))
            )
          )
        )
      )::double precision as score,
      array_agg(distinct qt.term order by qt.term) as matched_terms
    from query_terms qt
    join public.knowledge_base_lexical_terms kblt
      on kblt.knowledge_base_id = p_knowledge_base_id
     and kblt.term = qt.term
    join public.chunk_lexical_terms clt
      on clt.knowledge_base_id = p_knowledge_base_id
     and clt.term = qt.term
    join public.chunk_lexical_statistics cls
      on cls.chunk_id = clt.chunk_id
     and (p_index_configuration_fingerprint is null or cls.index_configuration_fingerprint = p_index_configuration_fingerprint)
    cross join corpus
    where corpus.indexed_chunk_count > 0
      and corpus.average_document_length > 0
    group by clt.chunk_id
  )
  select
    d.id as document_id,
    c.id as chunk_id,
    d.relative_path,
    c.heading_path,
    c.content,
    c.start_line,
    c.end_line,
    scored_terms.score as bm25_score,
    scored_terms.matched_terms
  from scored_terms
  join public.chunks c on c.id = scored_terms.chunk_id
  join public.documents d on d.id = c.document_id
  where d.knowledge_base_id = p_knowledge_base_id
    and d.index_status = 'indexed'
  order by scored_terms.score desc, d.relative_path, c.start_line nulls last, c.id
  limit least(greatest(coalesce(p_match_count, 10), 1), 100)
$$;

revoke execute on function public.search_chunks_bm25(text[], uuid, integer, double precision, double precision, text) from public;

do $$
begin
  if exists (select 1 from pg_roles where rolname = 'anon') then
    revoke execute on function public.search_chunks_bm25(text[], uuid, integer, double precision, double precision, text) from anon;
  end if;

  if exists (select 1 from pg_roles where rolname = 'authenticated') then
    revoke execute on function public.search_chunks_bm25(text[], uuid, integer, double precision, double precision, text) from authenticated;
  end if;

  if exists (select 1 from pg_roles where rolname = 'service_role') then
    grant execute on function public.search_chunks_bm25(text[], uuid, integer, double precision, double precision, text) to service_role;
  end if;
end
$$;
