# Copyright : Somantra Pty Ltd.
# Licensed under the GNU General Public License v3.0 (see LICENSE / COPYRIGHT).
"""
OpenAI embedding scoring for --score-embeddings.

Requires your own OpenAI API key -- set OPENAI_API_KEY (and optionally OPENAI_EMBEDDING_MODEL,
default "text-embedding-3-small") in a local .env file or your shell environment. See
.env.example in this directory. No key is bundled with this repo, and this module makes no
calls unless --score-embeddings is passed.

Requires the optional dependencies in requirements-optional.txt (openai, numpy, python-dotenv).
"""
import logging
import os

import numpy as np
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

logger = logging.getLogger(__name__)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_EMBEDDING_MODEL = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")

_client = None
if OPENAI_API_KEY:
    _client = OpenAI(api_key=OPENAI_API_KEY)


def get_embedding(text: str, model: str = OPENAI_EMBEDDING_MODEL):
    if _client is None:
        raise RuntimeError(
            "OPENAI_API_KEY is not set. Copy .env.example to .env and add your own key, or "
            "export OPENAI_API_KEY in your shell, to use --score-embeddings."
        )
    try:
        response = _client.embeddings.create(model=model, input=text)
    except Exception as e:
        logger.error("Error getting embedding from OpenAI for text=%r: %s", text, e)
        return None
    return np.array(response.data[0].embedding)


def cosine_similarity(vec1, vec2) -> float:
    return float(np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2)))
