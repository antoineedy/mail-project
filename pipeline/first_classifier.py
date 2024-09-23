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

import torch
from transformers import (
    AutoModelForSequenceClassification,
    CamembertForMaskedLM,
    AutoTokenizer,
    AutoConfig,
)

load_dotenv()

os.environ["HUGGINGFACEHUB_API_TOKEN"] = os.getenv("HUGGINGFACEHUB_API_TOKEN")

local_llm = "llama3.1"
# local_llm = "gemma2:27b"
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


tokenizer = AutoTokenizer.from_pretrained("camembert-base")


def get_preds(model, tokenizer, sentence):
    tokenized_sentence = tokenizer(sentence, return_tensors="pt")
    input_ids, attention_mask = (
        tokenized_sentence.input_ids,
        tokenized_sentence.attention_mask,
    )

    out = model(
        input_ids=tokenized_sentence.input_ids,
        attention_mask=tokenized_sentence.attention_mask,
    )

    logits = out.logits

    probas = torch.softmax(logits, -1).squeeze()

    pred = torch.argmax(probas)

    ID_TO_LABEL = {0: "activite", 1: "disponibilite_chambres", 2: "ouverture_accueil"}

    return ID_TO_LABEL[int(pred)], probas[pred].item()


def choose_route_2(mail: str):

    model = torch.load(
        "/Users/antoineedy/Documents/Formation/mail-project/model.pth",
        weights_only=False,
    )

    label_predicted, proba = get_preds(model, tokenizer, mail)

    # print("ùùù", mail)
    # print("***", proba)
    # print("%%%", label_predicted)

    if proba < 0.7:
        return {"question": mail, "datasource": "other"}
    else:
        return {"question": mail, "datasource": label_predicted}


first_classifier = RunnableLambda(choose_route_2)
