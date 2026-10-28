"""Step 1.5: turn a pasted developer question into a structured triage summary.

Run:  uv run python summarize_chain.py
      uv run python summarize_chain.py "your own question here"
"""
import sys
from typing import Literal

from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama
from pydantic import BaseModel, Field

CHAT_MODEL = "qwen3.5:4b"


# 1) The schema: the "form" the model must fill in. Pydantic rejects anything else.
class TriageSummary(BaseModel):
    pain_point: str = Field(description="The user's problem in one short sentence.")
    area: Literal[
        "install", "gpu", "api", "model_support", "performance", "cli", "other"
    ] = Field(description="Which part of Ollama the problem is about.")
    kind: Literal["bug", "feature_request", "question"] = Field(
        description="bug = something broken; feature_request = asks for something new; "
        "question = asks how to do something."
    )
    key_terms: list[str] = Field(
        description="3 to 6 important terms: error messages, model names, OS, hardware.",
        min_length=1,
        max_length=8,
    )


# 2) The prompt: a reusable message with one blank, {question}.
PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You triage developer questions about Ollama, a tool for running "
            "open models locally. Read the question and fill in the summary. "
            "Use only facts from the question. Do not invent details.",
        ),
        ("human", "Question:\n{question}"),
    ]
)


# 3) The chain: prompt -> model -> validated TriageSummary object.
def build_chain(model_name: str = CHAT_MODEL):
    llm = ChatOllama(
        model=model_name,
        temperature=0,    # same input -> (nearly) same output; needed for fair comparison later
        reasoning=False,  # turn off Qwen's thinking mode so we get clean JSON
    )
    return PROMPT | llm.with_structured_output(TriageSummary, method="json_schema")


SAMPLE_QUESTION = (
    "Ollama won't use my GPU on Ubuntu 24.04. I have an RTX 3060 and the driver "
    "is installed, but `ollama ps` shows 100% CPU and llama3 is really slow. "
    "Is there a setting I'm missing?"
)

if __name__ == "__main__":
    question = sys.argv[1] if len(sys.argv) > 1 else SAMPLE_QUESTION
    chain = build_chain()
    result = chain.invoke({"question": question})
    print("Returned type:", type(result).__name__)
    print(result.model_dump_json(indent=2))