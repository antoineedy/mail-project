import json
import pandas as pd
from copy import deepcopy

import os
from dotenv import load_dotenv

from typing import Literal
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from langchain_core.runnables import RunnableParallel, RunnablePassthrough

from langchain_core.output_parsers import JsonOutputParser
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda

from langchain_ollama.embeddings import OllamaEmbeddings
from langchain_community.chat_models import ChatOllama

from ollama_instructor.ollama_instructor_client import OllamaInstructorClient

load_dotenv()

os.environ["HUGGINGFACEHUB_API_TOKEN"] = os.getenv("HUGGINGFACEHUB_API_TOKEN")

# local_llm = "llama3.1"
local_llm = "gemma2:27b"
local_embedder = "nomic-embed-text"

embedder = OllamaEmbeddings(model=local_embedder)
llm = ChatOllama(model=local_llm, temperature=0)


class RouteQuery(BaseModel):
    """Tu vas guider l'utilisateur vers la bonne source de données pour répondre à sa question."""

    datasource: Literal[
        "disponibilite_chambres", "ouverture_accueil", "activite", "other"
    ] = Field(
        ...,
        description="On te donne une question d'un utilisateur, et tu dois choisir la source de données la plus pertinente pour répondre à sa question. \
            disponibilite_chambres se réfère à la disponibilité des chambres. \
            ouverture_accueil se réfère aux horaires d'ouverture de l'accueil. \
            activite se réfère aux différentes activités de loisir proposées par l'hotel. \
            other se réfère à une question qui ne rentre pas dans les autres catégories.",
    )


client = OllamaInstructorClient()


def choose_route(mail: str) -> RouteQuery:
    response = client.chat_completion(
        model="llama3.1",
        pydantic_model=RouteQuery,
        messages=[{"role": "user", "content": mail}],
    )
    if response is None:
        return {"question": mail, "datasource": "error"}
    return {
        "question": mail,
        "datasource": response["message"]["content"]["datasource"],
    }


first_classifier = RunnableLambda(choose_route)
