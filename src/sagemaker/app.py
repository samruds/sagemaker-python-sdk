from __future__ import absolute_import

import logging
from transformers import pipeline
from fastapi import FastAPI, Request
import uvicorn
import asyncio

logger = logging.getLogger(__name__)

app = FastAPI(
    title="Transformers In Process Server",
    version="1.0",
    description="A simple server",
)


@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.get("/generate")
async def generate_text(prompt: Request):
    logger.info("Generating Text....")

    str_prompt = await prompt.json()

    logger.info(str_prompt)

    generated_text = generator(str_prompt, max_length=30, num_return_sequences=5, truncation=True)
    return generated_text[0]["generated_text"]


generator = pipeline('text-generation', model='gpt2')


@app.post("/post")
def post(payload: dict):
    return payload


async def main():
    logger.info("Running")
    config = uvicorn.Config(
        "sagemaker.app:app",
        host="127.0.0.1",
        port=9007,
        log_level="info",
        loop="asyncio",
        reload=True,
        workers=3,
        use_colors=True,
    )
    server = uvicorn.Server(config)
    logger.info("I'm just waiting for a connection")
    await server.serve()