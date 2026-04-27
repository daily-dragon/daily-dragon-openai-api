# daily-dragon-openai-api

FastAPI app deployed on AWS Lambda (via Mangum) for the Daily Dragon language learning app.
Uses OpenAI GPT-4o structured outputs to generate Chinese practice sentences and evaluate user translations.

## Stack
- FastAPI + Pydantic models → `openai_api/`
- Auth: AWS Cognito (`fastapi-cognito`)
- LLM: OpenAI `gpt-4o-2024-08-06` with `response_format` structured parsing
- Prompts: plain text templates in `openai_api/prompts/`
- Lambda entry point: `openai_api_handler.py`

## Run tests
```
pytest openai_api/tests/
```

## Patterns
- Add new endpoints in `openai_api_app.py`, service logic in `openai_service.py`, models in `models.py`
- New LLM prompts go in `openai_api/prompts/` as plain text files with `${variable}` placeholders