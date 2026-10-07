"""Response bodies. Descriptions and examples here appear in the API reference at /docs."""

from pydantic import BaseModel, Field


class SearchHit(BaseModel):
    work_id: str = Field(description="Id of the work the text comes from", examples=["kjv"])
    label: str = Field(description="Display name of the work", examples=["KJV"])
    type: str = Field(description="Kind of text: `verse`, `commentary`, `dictionary`, …", examples=["verse"])
    ref: str = Field(description="Book, chapter and verse", examples=["Genesis 19:26"])
    text: str = Field(examples=["But his wife looked back from behind him, and she became a pillar of salt."])
    score: float = Field(
        description="Cosine similarity for semantic search; fused rank score for keyword and hybrid. "
        "Higher is better.",
        examples=[0.0328],
    )


class SearchResponse(BaseModel):
    query: str = Field(examples=["a woman looks back at a burning city and turns into a pillar of salt"])
    mode: str = Field(examples=["hybrid"])
    hits: list[SearchHit]


class Source(BaseModel):
    n: int = Field(description="The number the answer cites as [n]", examples=[1])
    work_id: str = Field(examples=["kjv"])
    label: str = Field(examples=["KJV"])
    type: str = Field(examples=["verse"])
    ref: str = Field(examples=["Matthew 5:9"])
    text: str = Field(examples=["Blessed are the peacemakers: for they shall be called the children of God."])


class Usage(BaseModel):
    input_tokens: int = Field(examples=[1297])
    output_tokens: int = Field(examples=[229])


class AskResponse(BaseModel):
    question: str = Field(examples=["What does the Bible say about peacemakers?"])
    answer: str = Field(
        description="Written only from the sources, citing them inline as [1], [2], …",
        examples=["Peacemakers are blessed and will be called the children of God [1]."],
    )
    model: str = Field(description="The model that wrote the answer", examples=["gemini-3.1-flash-lite"])
    sources: list[Source] = Field(description="The passages retrieved for the question, numbered")
    cited: list[int] = Field(description="Source numbers the answer cites", examples=[[1]])
    invalid_citations: list[int] = Field(
        description="Numbers the answer cites that match no source. Normally empty.", examples=[[]]
    )
    usage: Usage | None = Field(default=None, description="Tokens the model used")


class Health(BaseModel):
    status: str = Field(examples=["ok"])
    packs: int = Field(description="Works loaded", examples=[19])
    documents: int = Field(description="Searchable texts across all works", examples=[294576])
    ask_enabled: bool = Field(description="Whether /ask is configured", examples=[True])
