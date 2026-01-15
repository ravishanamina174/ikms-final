from pydantic import BaseModel


class QuestionRequest(BaseModel):
    """Request body for the `/qa` endpoint.

    The PRD specifies a single field named `question` that contains
    the user's natural language question about the vector databases paper.
    """

    question: str
    use_planning: bool = True


class QAResponse(BaseModel):
    """Response body for the `/qa` endpoint.

    From the API consumer's perspective we only expose the final,
    verified answer plus some metadata (e.g. context snippets).
    Internal draft answers remain inside the agent pipeline.
    """

    answer: str
    plan: str | None = None
    sub_questions: list[str] | None = None
    context: str
