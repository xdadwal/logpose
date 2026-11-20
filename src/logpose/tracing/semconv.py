"""Every span attribute name logpose writes. No other module spells one.

GenAI semantic conventions are still in Development; this pins one release.
"""

from __future__ import annotations

SEMCONV_VERSION = "1.45.0"
SCHEMA_URL = f"https://opentelemetry.io/schemas/{SEMCONV_VERSION}"

# Span names / operation values
OPERATION_INVOKE_AGENT = "invoke_agent"
OPERATION_CHAT = "chat"
OPERATION_EXECUTE_TOOL = "execute_tool"
SPAN_GUARDRAIL = "guardrail"
SPAN_COMPACTION = "compaction"

# GenAI conventions
GEN_AI_OPERATION_NAME = "gen_ai.operation.name"
GEN_AI_PROVIDER_NAME = "gen_ai.provider.name"
GEN_AI_REQUEST_MODEL = "gen_ai.request.model"
GEN_AI_REQUEST_MAX_TOKENS = "gen_ai.request.max_tokens"
GEN_AI_RESPONSE_FINISH_REASONS = "gen_ai.response.finish_reasons"
GEN_AI_AGENT_NAME = "gen_ai.agent.name"
GEN_AI_CONVERSATION_ID = "gen_ai.conversation.id"
GEN_AI_USAGE_INPUT_TOKENS = "gen_ai.usage.input_tokens"
GEN_AI_USAGE_OUTPUT_TOKENS = "gen_ai.usage.output_tokens"
GEN_AI_SYSTEM_INSTRUCTIONS = "gen_ai.system_instructions"
GEN_AI_TOOL_DEFINITIONS = "gen_ai.tool.definitions"
GEN_AI_INPUT_MESSAGES = "gen_ai.input.messages"
GEN_AI_OUTPUT_MESSAGES = "gen_ai.output.messages"
GEN_AI_TOOL_NAME = "gen_ai.tool.name"
GEN_AI_TOOL_CALL_ID = "gen_ai.tool.call.id"
GEN_AI_TOOL_CALL_ARGUMENTS = "gen_ai.tool.call.arguments"
GEN_AI_TOOL_CALL_RESULT = "gen_ai.tool.call.result"

# Standard non-GenAI
ERROR_TYPE = "error.type"

# logpose's own attributes
LOGPOSE_SEMCONV_VERSION = "logpose.semconv.version"
LOGPOSE_RUN_ID = "logpose.run.id"
LOGPOSE_SERVICE_NAME = "logpose.service.name"
LOGPOSE_RUN_ITERATIONS = "logpose.run.iterations"
LOGPOSE_RUN_OUTCOME = "logpose.run.outcome"
LOGPOSE_USAGE_INPUT_TOKENS = "logpose.usage.input_tokens"
LOGPOSE_USAGE_REASONING_TOKENS = "logpose.usage.reasoning_tokens"
LOGPOSE_USAGE_CACHE_READ_INPUT_TOKENS = "logpose.usage.cache_read_input_tokens"
LOGPOSE_USAGE_CACHE_CREATION_INPUT_TOKENS = "logpose.usage.cache_creation_input_tokens"
LOGPOSE_METADATA_PREFIX = "logpose.metadata."
LOGPOSE_CONFIG_PREFIX = "logpose.config."
LOGPOSE_CONTEXT_PREFIX = "logpose.context."
LOGPOSE_ATTEMPT_NUMBER = "logpose.attempt.number"
LOGPOSE_ATTEMPT_RETRYABLE = "logpose.attempt.retryable"
LOGPOSE_TOOL_QUEUE_SECONDS = "logpose.tool.queue_seconds"
LOGPOSE_TOOL_TIMED_OUT = "logpose.tool.timed_out"
LOGPOSE_TOOL_IS_ERROR = "logpose.tool.is_error"
LOGPOSE_CONTENT_TRUNCATED = "logpose.content.truncated"
LOGPOSE_GUARDRAIL_NAME = "logpose.guardrail.name"
LOGPOSE_GUARDRAIL_ALLOWED = "logpose.guardrail.allowed"
LOGPOSE_GUARDRAIL_CATEGORY = "logpose.guardrail.category"
LOGPOSE_GUARDRAIL_SCORES = "logpose.guardrail.scores"
LOGPOSE_GUARDRAIL_INPUT = "logpose.guardrail.input"
LOGPOSE_COMPACTION_POINTERS = "logpose.compaction.pointers_applied"
LOGPOSE_COMPACTION_SUMMARIZED = "logpose.compaction.summarized"
LOGPOSE_COMPACTION_ESTIMATE_BEFORE = "logpose.compaction.estimate_before"
LOGPOSE_COMPACTION_ESTIMATE_AFTER = "logpose.compaction.estimate_after"
LOGPOSE_COMPACTION_BUDGET = "logpose.compaction.input_budget"
LOGPOSE_COMPACTION_SUMMARY = "logpose.compaction.summary"
LOGPOSE_INPUT_SAME_AS_SPAN = "logpose.input.same_as_span"
LOGPOSE_CHAT_PURPOSE = "logpose.chat.purpose"
LOGPOSE_GUARDRAIL_REASON = "logpose.guardrail.reason"
LOGPOSE_RETRY_DELAY_SECONDS = "logpose.retry.delay_seconds"
LOGPOSE_GATE_IS_ERROR = "logpose.gate.is_error"
LOGPOSE_GATE_REPLACEMENT = "logpose.gate.replacement"

# Events
EVENT_GATE_BLOCKED = "logpose.gate.blocked"
EVENT_RETRY_SCHEDULED = "logpose.retry.scheduled"
EVENT_DISCARDED = "logpose.discarded"
