# Selective Agent Control

The Selective LLM Agent is a bounded control plane rather than an unrestricted planner. A Necessary-LLM gate determines whether an LLM policy call is useful. LLM output is only a proposal: structured validation and deterministic guards decide whether the action is allowed.

The governed path limits Graph Recovery to one hop and Recovery to one attempt. The LLM is not granted direct Finish authority. A safe abstention or refusal is a valid terminal outcome.
