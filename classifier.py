"""
Claude classification engine — replaces probability estimation with direction classification.
Asks "does this news confirm or deny the market question?" instead of "what's the probability?"
"""
from __future__ import annotations

import json
import re
import time
import logging
from dataclasses import dataclass

import anthropic

import config
from markets import Market

log = logging.getLogger(__name__)

# FIX: Validate API key at import time so failures are loud and immediate,
# rather than silently surfacing as neutral classifications at runtime.
if not config.ANTHROPIC_API_KEY:
      raise RuntimeError(
                "ANTHROPIC_API_KEY is not set. Please add it to your .env file. "
                "Classification cannot run without it."
      )

client = anthropic.Anthropic(api_key=config.ANTHROPIC_API_KEY)

CLASSIFICATION_PROMPT = """You are a news classifier for prediction markets.

## Market Question
{question}

## Current Market Price
YES: {yes_price:.2f} (implied probability: {yes_price:.0%})

## Breaking News
{headline}
Source: {source}

## Task
Does this news make the market question MORE likely to resolve YES, MORE likely to resolve NO, or is it NOT RELEVANT?

Also rate the MATERIALITY - how much should this move the price? 0.0 means no impact, 1.0 means this is definitive evidence.

Respond with ONLY valid JSON:
{{
  "direction": "bullish" | "bearish" | "neutral",
    "materiality": <float 0.0 to 1.0>,
      "reasoning": "<1 sentence>"
}}"""

# FIX: Regex-based JSON extraction is more robust than split-on-backticks.
# Handles cases where the model adds preamble, triple-fenced code blocks, or trailing text.
_JSON_RE = re.compile(r"\{[\s\S]*\}", re.MULTILINE)


@dataclass
class Classification:
      direction: str  # "bullish", "bearish", "neutral"
    materiality: float  # 0.0-1.0
    reasoning: str
    latency_ms: int
    model: str


def classify(headline: str, market: Market, source: str = "unknown") -> Classification:
      """Classify a news headline against a market question. Synchronous."""
      start = time.time()

    # FIX: Truncate headline and source to limit prompt injection surface.
      # External news content is untrusted and should not overflow the prompt.
      safe_headline = headline[:500]
      safe_source = source[:100]

    prompt = CLASSIFICATION_PROMPT.format(
              question=market.question,
              yes_price=market.yes_price,
              headline=safe_headline,
              source=safe_source,
    )

    last_error: Exception | None = None

    for attempt in range(1, config.EXECUTOR_MAX_RETRIES + 1):
              try:
                            response = client.messages.create(
                                              model=config.CLASSIFICATION_MODEL,
                                              max_tokens=200,
                                              temperature=0.1,
                                              messages=[{"role": "user", "content": prompt}],
                            )
                            text = response.content[0].text.strip()

                  # FIX: Use regex to find first JSON object rather than splitting on backticks.
                            match = _JSON_RE.search(text)
                            if not match:
                                              raise ValueError(f"No JSON object found in model response: {text[:200]!r}")

                            result = json.loads(match.group())
                            latency = int((time.time() - start) * 1000)

                  direction = result.get("direction", "neutral")
            if direction not in ("bullish", "bearish", "neutral"):
                              direction = "neutral"

            materiality = max(0.0, min(1.0, float(result.get("materiality", 0))))

            return Classification(
                              direction=direction,
                              materiality=materiality,
                              reasoning=result.get("reasoning", ""),
                              latency_ms=latency,
                              model=config.CLASSIFICATION_MODEL,
            )

except anthropic.RateLimitError as e:
            last_error = e
            log.warning(
                              f"[classifier] Rate limit hit on attempt {attempt}/{config.EXECUTOR_MAX_RETRIES}. "
                              f"Waiting {config.EXECUTOR_RETRY_DELAY_SECONDS * attempt}s before retry."
            )
            time.sleep(config.EXECUTOR_RETRY_DELAY_SECONDS * attempt)

except anthropic.APIStatusError as e:
            last_error = e
            if e.status_code and e.status_code < 500:
                              # 4xx errors (bad request, auth failure) won't be fixed by retrying
                              log.error(f"[classifier] Non-retryable API error {e.status_code}: {e}")
                              break
                          log.warning(f"[classifier] API error on attempt {attempt}: {e}. Retrying...")
            time.sleep(config.EXECUTOR_RETRY_DELAY_SECONDS)

except Exception as e:
            last_error = e
            log.warning(f"[classifier] Error on attempt {attempt}: {type(e).__name__}: {e}")
            if attempt < config.EXECUTOR_MAX_RETRIES:
                              time.sleep(config.EXECUTOR_RETRY_DELAY_SECONDS)

    # FIX: All retries exhausted — log clearly as an error, not just a warning,
    # so operators know classification failed rather than treating it as neutral signal.
    latency = int((time.time() - start) * 1000)
    log.error(
              f"[classifier] All {config.EXECUTOR_MAX_RETRIES} attempts failed. "
              f"Last error: {type(last_error).__name__}: {last_error}. "
              "Returning neutral/zero-materiality to avoid false signals."
    )
    return Classification(
              direction="neutral",
              materiality=0.0,
              reasoning=f"Classification failed after {config.EXECUTOR_MAX_RETRIES} attempts: {type(last_error).__name__}",
              latency_ms=latency,
              model=config.CLASSIFICATION_MODEL,
    )


async def classify_async(headline: str, market: Market, source: str = "unknown") -> Classification:
      """Async wrapper around classify()."""
    import asyncio
    loop = asyncio.get_running_loop()  # FIX: use get_running_loop() — get_event_loop() is deprecated in 3.10+
    return await loop.run_in_executor(None, classify, headline, market, source)


if __name__ == "__main__":
      test_market = Market(
                condition_id="test",
                question="Will OpenAI release GPT-5 before August 2026?",
                category="ai",
                yes_price=0.62,
                no_price=0.38,
                volume=500000,
                end_date="2026-08-01",
                active=True,
                tokens=[],
      )

    result = classify(
              headline="OpenAI reportedly testing GPT-5 internally with select partners",
              market=test_market,
              source="The Information",
    )
    print(f"Direction: {result.direction}")
    print(f"Materiality: {result.materiality}")
    print(f"Reasoning: {result.reasoning}")
    print(f"Latency: {result.latency_ms}ms")
