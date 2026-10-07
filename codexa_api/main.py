"""The HTTP API. Run it with: uvicorn codexa_api.main:app"""

import secrets
from contextlib import asynccontextmanager
from dataclasses import asdict
from typing import Annotated

from fastapi import Depends, FastAPI, HTTPException, Query, Request, Security
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.security import APIKeyHeader
from langfuse import get_client
from scalar_fastapi import get_scalar_api_reference

from .assistant import Assistant, GeminiLLM, LLMError
from .config import get_settings
from .embedder import Embedder
from .packs import load_packs
from .schemas import AskResponse, Health, SearchHit, SearchResponse
from .search import Mode, Searcher

DESCRIPTION = """
Find Bible verses from a plain-English description, and ask questions that are answered only from
the passages found, with numbered citations.

- **Search** public-domain translations, the Greek and Hebrew texts, commentaries and reference works,
  by keyword, by meaning, or both.
- **Ask** returns an answer written by Gemini from the retrieved passages, with every citation checked.
  It needs an API key in the `X-API-Key` header.

Source: [github.com/brooka/codexa-api-python](https://github.com/brooka/codexa-api-python)
"""

api_key_header = APIKeyHeader(
    name="X-API-Key", auto_error=False, description="Required by /ask when the server sets a key."
)


def build_from_settings() -> tuple[Searcher, Assistant | None]:
    settings = get_settings()
    searcher = Searcher(load_packs(settings.data_dir, settings.pack_ids), Embedder(settings.model_dir))
    assistant = None
    if settings.gemini_api_key:
        llm = GeminiLLM(settings.gemini_api_key, settings.llm_model)
        assistant = Assistant(searcher, llm, settings.ask_sources)
    return searcher, assistant


def get_searcher(request: Request) -> Searcher:
    return request.app.state.searcher


def get_assistant(request: Request, api_key: Annotated[str | None, Security(api_key_header)]) -> Assistant:
    if request.app.state.assistant is None:
        raise HTTPException(status_code=503, detail="Set GEMINI_API_KEY to enable /ask.")
    expected = request.app.state.ask_key
    if expected and not secrets.compare_digest(api_key or "", expected):
        raise HTTPException(status_code=401, detail="Send the API key in the X-API-Key header.")
    return request.app.state.assistant


SearcherDep = Annotated[Searcher, Depends(get_searcher)]
AssistantDep = Annotated[Assistant, Depends(get_assistant)]


def create_app(
    searcher: Searcher | None = None, assistant: Assistant | None = None, ask_key: str | None = None
) -> FastAPI:
    """The app. Tests pass their own parts; otherwise everything loads from settings at start-up."""

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if searcher is None:
            app.state.searcher, app.state.assistant = build_from_settings()
            app.state.ask_key = get_settings().ask_key
        else:
            app.state.searcher, app.state.assistant, app.state.ask_key = searcher, assistant, ask_key
        yield
        get_client().flush()  # send any traces still buffered

    app = FastAPI(
        title="Codexa API",
        version="0.1.0",
        description=DESCRIPTION,
        docs_url="/swagger",
        redoc_url=None,
        lifespan=lifespan,
    )

    @app.get("/", include_in_schema=False)
    def root() -> RedirectResponse:
        return RedirectResponse("/docs")

    @app.get("/docs", include_in_schema=False)
    def docs() -> HTMLResponse:
        return get_scalar_api_reference(openapi_url=app.openapi_url, title="Codexa API", telemetry=False)

    @app.get("/search", summary="Search the texts", tags=["Search"])
    def search(
        q: Annotated[
            str,
            Query(
                min_length=1,
                max_length=500,
                description="Words, a phrase, or a description of the passage.",
                examples=["a woman looks back at a burning city and turns into a pillar of salt"],
            ),
        ],
        searcher: SearcherDep,
        mode: Annotated[
            Mode,
            Query(
                description="`keyword` matches words (BM25), `semantic` matches meaning (vectors), "
                "`hybrid` does both and merges the rankings."
            ),
        ] = "hybrid",
        limit: Annotated[int, Query(ge=1, le=50, description="Maximum number of results.")] = 10,
        kind: Annotated[
            str | None,
            Query(
                alias="type", description="Search one kind of text: `verse`, `commentary`, `dictionary`, …"
            ),
        ] = None,
        work_id: Annotated[str | None, Query(description="Search one work, e.g. `kjv`.")] = None,
    ) -> SearchResponse:
        """Ranks texts from every loaded work together: English translations, the Greek and Hebrew
        texts, commentaries and reference works. Each hit names its work and reference."""
        hits = searcher.search(q, mode, limit, kind, work_id)
        return SearchResponse(query=q, mode=mode, hits=[SearchHit(**asdict(h)) for h in hits])

    @app.get(
        "/ask",
        summary="Ask a question",
        tags=["Ask"],
        responses={
            401: {"description": "Missing or wrong API key"},
            502: {"description": "Upstream AI provider error"},
            503: {"description": "/ask isn't set up"},
        },
    )
    def ask(
        q: Annotated[
            str,
            Query(
                min_length=1,
                max_length=500,
                description="A question about Scripture.",
                examples=["What does the Bible say about peacemakers?"],
            ),
        ],
        assistant: AssistantDep,
    ) -> AskResponse:
        """Retrieves the eight most relevant passages with hybrid search, then asks Gemini to answer
        using only those, citing them as [1], [2], …. Citations that match no passage are listed
        in `invalid_citations`."""
        try:
            return assistant.ask(q)
        except LLMError as err:
            raise HTTPException(status_code=502, detail=str(err)) from err

    @app.get("/health", summary="Check the service", tags=["Service"])
    def health(request: Request, searcher: SearcherDep) -> Health:
        """What's loaded, and whether /ask is available."""
        return Health(
            status="ok",
            packs=len(searcher.packs),
            documents=sum(len(p.vectors) for p in searcher.packs),
            ask_enabled=request.app.state.assistant is not None,
        )

    return app


app = create_app()
