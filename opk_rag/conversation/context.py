from __future__ import annotations

from opk_rag.conversation.models import (
    CONVERSATION_PROMPT_VERSION,
    DEFAULT_CONTEXT_MAX_TURNS,
    DEFAULT_CONTEXT_TOKEN_BUDGET,
    ConversationContext,
    ConversationContextTurn,
    ConversationSession,
    ConversationTurn,
    ConversationTurnCitation,
)


def build_conversation_context(
    *,
    session: ConversationSession,
    turns: tuple[ConversationTurn, ...],
    citations_by_turn: dict[object, tuple[ConversationTurnCitation, ...]],
    token_counter,
    max_turns: int = DEFAULT_CONTEXT_MAX_TURNS,
    token_budget: int = DEFAULT_CONTEXT_TOKEN_BUDGET,
) -> ConversationContext:
    selected: list[ConversationContextTurn] = []
    total_tokens = 0
    for turn in reversed(turns[-max_turns:]):
        text = _context_text(turn)
        token_count = int(token_counter.count_tokens(text)) if text else 0
        if selected and total_tokens + token_count > token_budget:
            break
        if not selected and token_count > token_budget:
            token_count = token_budget
        total_tokens += token_count
        selected.append(
            ConversationContextTurn(
                turn_number=turn.turn_number,
                user_query=turn.user_query,
                standalone_query=turn.standalone_query,
                answer_text=turn.answer_text,
                answer_decision=turn.answer_decision,
                abstention_reason=turn.abstention_reason,
                citations=citations_by_turn.get(turn.id, ()),
                token_count=token_count,
            )
        )
    selected.reverse()
    return ConversationContext(
        session_id=session.id,
        knowledge_base_id=session.knowledge_base_id,
        prompt_version=session.conversation_prompt_version or CONVERSATION_PROMPT_VERSION,
        max_turns=max_turns,
        token_budget=token_budget,
        token_count=total_tokens,
        turns=tuple(selected),
    )


def _context_text(turn: ConversationTurn) -> str:
    parts = [turn.user_query]
    if turn.standalone_query and turn.standalone_query != turn.user_query:
        parts.append(turn.standalone_query)
    if turn.answer_text:
        parts.append(turn.answer_text)
    if turn.abstention_reason:
        parts.append(turn.abstention_reason)
    return "\n".join(part for part in parts if part)
