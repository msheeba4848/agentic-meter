"""Multi-agent LangChain pipeline with per-agent cost attribution.

Requires: pip install 'agenticmeter[langchain]' langchain-openai

Set OPENAI_API_KEY in your environment, then run:
    python examples/langchain_multi_agent.py
"""
import os
from agenticmeter import Ledger
from langchain_openai import ChatOpenAI


def main():
    researcher = ChatOpenAI(model="gpt-4o-mini")
    writer = ChatOpenAI(model="gpt-4o-mini")
    critic = ChatOpenAI(model="gpt-4o-mini")

    with Ledger(budget="$1.00", name="multi_agent_pipeline") as ledger:
        # One callback instance works for all three agents - it reads the
        # active ledger and current tag stack via ContextVar on every event.
        cb = ledger.as_langchain_callback()

        with ledger.tag("researcher"):
            research = researcher.invoke(
                "Research: what is photosynthesis? Two sentences.",
                config={"callbacks": [cb]},
            )

        with ledger.tag("writer"):
            draft = writer.invoke(
                f"Write a haiku based on: {research.content}",
                config={"callbacks": [cb]},
            )

        with ledger.tag("critic"):
            review = critic.invoke(
                f"Critique this haiku in one sentence: {draft.content}",
                config={"callbacks": [cb]},
            )

    print("Research:", research.content)
    print()
    print("Draft:", draft.content)
    print()
    print("Review:", review.content)
    print()
    print(ledger.summary())


if __name__ == "__main__":
    if not os.environ.get("OPENAI_API_KEY"):
        raise SystemExit("Set OPENAI_API_KEY in your environment first.")
    main()