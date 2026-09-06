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

drop trigger if exists chunk_lexical_statistics_refresh_after_delete on public.chunk_lexical_statistics;

create trigger chunk_lexical_statistics_refresh_after_delete
after delete on public.chunk_lexical_statistics
for each row execute function public.refresh_lexical_corpus_after_chunk_stat_delete();
