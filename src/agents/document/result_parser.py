"""Result parser for DocumentAgent.

Provides structured parsing of agent execution results,
extracting content from tool outputs.
"""

import json
import logging
from typing import Dict, List, Any, Optional

from langchain_core.messages import ToolMessage

from src.agents.core.token_utils import calculate_token_usage
from .schemas import (
    GeneratedContent,
    FAQ,
    Question,
    TokenUsage,
)

logger = logging.getLogger(__name__)


class AgentResultParser:
    """Parses agent execution results to extract structured content.

    Handles extraction of:
    - Summaries from summary_generator tool
    - FAQs from faq_generator tool
    - Questions from question_generator tool
    - Source paths from document_loader tool
    - Tool usage tracking
    """

    def parse(self, response_text: str, messages: List = None) -> Dict[str, Any]:
        """Parse the agent's response to extract structured content from tool outputs.

        Args:
            response_text: The final response text from the agent
            messages: List of all messages from agent execution

        Returns:
            Dict with response, content, source_path, metadata, tools_used, persisted
        """
        result = {
            "response": response_text,
            "content": None,
            "source_path": None,
            "metadata": None,
            "tools_used": [],
            "persisted": False
        }

        if not messages:
            logger.warning("No messages to parse - returning empty result")
            return result

        # Extract content from tool messages
        summary = None
        faqs = None
        questions = None
        tools_used = []

        # Debug: Log all message types to trace tool output
        logger.debug(f"Parsing {len(messages)} messages for tool outputs")
        for i, msg in enumerate(messages):
            msg_type = type(msg).__name__
            is_tool_msg = isinstance(msg, ToolMessage)
            logger.debug(f"Message [{i}]: type={msg_type}, is_ToolMessage={is_tool_msg}")

        for msg in messages:
            # Check if it's a ToolMessage (contains tool output)
            if isinstance(msg, ToolMessage):
                tool_result = self._parse_tool_message(msg)
                tools_used.append(tool_result["tool_usage"])

                if tool_result["success"]:
                    # Extract content from successful tool outputs
                    if tool_result.get("summary"):
                        summary = tool_result["summary"]
                        logger.debug(f"Extracted summary: {len(summary)} chars")

                    if tool_result.get("faqs"):
                        faqs = [
                            FAQ(question=f['question'], answer=f['answer'])
                            for f in tool_result["faqs"]
                            if isinstance(f, dict) and 'question' in f and 'answer' in f
                        ]
                        logger.debug(f"Extracted {len(faqs)} FAQs")

                    if tool_result.get("questions"):
                        questions = [
                            Question(
                                question=q['question'],
                                expected_answer=q.get('expected_answer', ''),
                                difficulty=q.get('difficulty', 'medium')
                            )
                            for q in tool_result["questions"]
                            if isinstance(q, dict) and 'question' in q
                        ]
                        logger.debug(f"Extracted {len(questions)} questions")

                    if tool_result.get("source_path"):
                        result['source_path'] = tool_result["source_path"]

        # Build GeneratedContent if we have any content
        if summary or faqs or questions:
            result['content'] = GeneratedContent(
                summary=summary,
                faqs=faqs,
                questions=questions
            )
            logger.info(
                f"Parsed content: summary={bool(summary)}, "
                f"faqs={len(faqs) if faqs else 0}, "
                f"questions={len(questions) if questions else 0}"
            )

        result['tools_used'] = tools_used

        # Debug final result
        has_summary = result['content'] and result['content'].summary if result['content'] else False
        logger.info(f"Parse complete: has_content={result['content'] is not None}, has_summary={has_summary}")

        return result

    def _parse_tool_message(self, msg: ToolMessage) -> Dict[str, Any]:
        """Parse a single ToolMessage to extract its content.

        Args:
            msg: A LangChain ToolMessage

        Returns:
            Dict with tool_usage, success flag, and extracted content fields
        """
        tool_name = getattr(msg, 'name', '') or ''
        result = {
            "tool_usage": None,
            "success": False,
            "summary": None,
            "faqs": None,
            "questions": None,
            "source_path": None,
        }

        try:
            content = msg.content if hasattr(msg, 'content') else str(msg)
            logger.debug(f"ToolMessage '{tool_name}': content_type={type(content).__name__}, content_length={len(str(content))}")
            tool_output = json.loads(content) if isinstance(content, str) else content
            logger.debug(f"ToolMessage '{tool_name}': parsed output keys={list(tool_output.keys()) if isinstance(tool_output, dict) else 'not a dict'}")

            # Create ToolUsage entry with parsed output
            tool_success = isinstance(tool_output, dict) and tool_output.get('success', False)
            result["tool_usage"] = {
                "tool_name": tool_name,
                "input_data": {},  # Not available from ToolMessage
                "output_data": tool_output if isinstance(tool_output, dict) else {"raw": str(tool_output)},
                "execution_time_ms": 0.0,  # Not tracked at this level
                "success": tool_success,
                "error_message": tool_output.get('error') if isinstance(tool_output, dict) else None
            }

            if tool_success:
                result["success"] = True

                # Extract summary from summary_generator tool
                if 'summary' in tool_output and tool_output['summary']:
                    result["summary"] = tool_output['summary']
                    logger.info(f"ToolMessage '{tool_name}': Extracted summary ({len(result['summary'])} chars)")

                # Extract FAQs from faq_generator tool
                if 'faqs' in tool_output and tool_output['faqs']:
                    result["faqs"] = tool_output['faqs']

                # Extract questions from question_generator tool
                if 'questions' in tool_output and tool_output['questions']:
                    result["questions"] = tool_output['questions']

                # Extract source path from document loader tool
                if 'source_path' in tool_output:
                    result["source_path"] = tool_output['source_path']

        except (json.JSONDecodeError, TypeError, KeyError) as e:
            logger.debug(f"Could not parse tool output as structured content: {e}")
            # Create failed ToolUsage entry
            result["tool_usage"] = {
                "tool_name": tool_name,
                "input_data": {},
                "output_data": None,
                "execution_time_ms": 0.0,
                "success": False,
                "error_message": str(e)
            }

        return result


def calculate_agent_token_usage(
    input_text: str,
    output_text: str,
    model: str
) -> TokenUsage:
    """Calculate token usage for agent execution.

    Args:
        input_text: The input/query text
        output_text: The response text
        model: The model name (e.g., 'gemini-3-flash-preview')

    Returns:
        TokenUsage with prompt_tokens, completion_tokens, total_tokens, cost
    """
    estimate = calculate_token_usage(input_text, output_text, model=model)
    return TokenUsage(
        prompt_tokens=estimate.prompt_tokens,
        completion_tokens=estimate.completion_tokens,
        total_tokens=estimate.total_tokens,
        estimated_cost_usd=estimate.estimated_cost_usd
    )
