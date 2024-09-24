import pandas as pd
import json

from langchain_ollama.embeddings import OllamaEmbeddings
from langchain_community.chat_models import ChatOllama

from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from typing import Literal
from ollama_instructor.ollama_instructor_client import OllamaInstructorClient

from langchain_core.output_parsers import StrOutputParser

from langchain_core.runnables import RunnableLambda

from langchain_chroma import Chroma

from datetime import datetime

import streamlit as st

local_llm = "llama3.1"
# local_llm = "gemma2:27b"
local_embedder = "nomic-embed-text"

client = OllamaInstructorClient()

embedder = OllamaEmbeddings(model=local_embedder)
llm = ChatOllama(model=local_llm, temperature=0)


def continue_route_fct(state):
    if state["datasource"] == "ouverture_accueil":

        state = is_accueil(state)

        return state

    elif state["datasource"] == "disponibilite_chambres":

        state = is_chambre(state)

        return state

    elif state["datasource"] == "activite":

        state = is_activite(state)

        return state

    elif state["datasource"] == "other":

        state = is_from_mail(state)

        return state


continue_route = RunnableLambda(continue_route_fct)


def is_accueil(state):
    state["data"] = pd.read_csv("data/accueil.csv")
    return state


class RoomDates(BaseModel):
    """Gives the start and end dates of a room reservation that a user is asking about, in 2023, as well as the type of room."""

    start_date: str = Field(
        ...,
        description="The start date of the room reservation that the user is asking about.",
    )
    end_date: str = Field(
        ...,
        description="The end date of the room reservation that the user is asking about.",
    )
    room_type: Literal["double", "family", "not_specified"] = Field(
        ..., description="The type of room that the user is asking about."
    )


def is_chambre(state: str) -> RoomDates:
    mail = state["question"]
    response = client.chat_completion(
        model=local_llm,
        pydantic_model=RoomDates,
        messages=[{"role": "user", "content": mail}],
    )
    start_date = response["message"]["content"]["start_date"]
    end_date = response["message"]["content"]["end_date"]
    room_type = response["message"]["content"]["room_type"]

    start_date = datetime.strptime(start_date, "%Y-%m-%d")
    end_date = datetime.strptime(end_date, "%Y-%m-%d")

    data = pd.read_csv("data/chambres.csv")

    week_beg = None
    week_end = None

    for i in range(len(data)):
        debut_week = datetime.strptime(data["Date_Debut"][i], "%Y-%m-%d")
        fin_week = datetime.strptime(data["Date_Fin"][i], "%Y-%m-%d")
        if start_date >= debut_week and start_date <= fin_week:
            week_beg = i
        if end_date >= debut_week and end_date <= fin_week:
            week_end = i

    if week_beg is None or week_end is None:
        raise Exception("No data found for this period")

    data = data.iloc[week_beg : week_end + 1]

    # new line and if dispo and dispo then dispo else not dispo

    final = pd.DataFrame(columns=data.columns)
    final.loc[0, "Date_Debut"] = data["Date_Debut"][week_beg]
    final.loc[0, "Date_Fin"] = data["Date_Fin"][week_end]
    final.loc[0, "Semaine"] = "/"
    for column in data.columns[3:]:
        # print(data.value_counts(column)["dispo"])
        temp = data.value_counts(column).reset_index()
        if temp[temp[column] == "dispo"]["count"].values == len(data):
            final.loc[0, column] = "dispo"
        else:
            final.loc[0, column] = "indispo"

    state["start_date"] = start_date
    state["end_date"] = end_date
    state["week_beg"] = week_beg + 1
    state["week_end"] = week_end + 1
    state["room_type"] = room_type
    state["data"] = data
    state["final"] = final

    # PROMPT = (
    #     f"""Tu es un assistant va répondre à une question de l'utilisateur à propos de la disponibilité des chambres. Tu vas être très simple et factuel. Tu vas t'exprimer clairement.

    #         Question : {mail}

    #         Tu vas prendre en compte la période suivante : {out["start_date"]} - {out["end_date"]} pour une chambre de type {out["room_type"]}.
    #         Tu vas répondre en prenant cette exemple:
    #         Pour la période demandée, j'ai trouvé ces données :
    #         Chambre XX (Double) : (disponible ou non disponible)
    #         Chambre XX (Famille) : ...
    #         etc...

    #         Puis ensuite tu conclus :

    #         Avec votre demande, la chambre XX semble être la plus adaptée.

    #         Tu vas répondre à l'aide de ces données (0 = non disponible, 1 = disponible) :
    #         Données :
    #         """
    #     + out["final"].to_string()
    # )

    # state["prompt"] = PROMPT

    return state


def is_activite(state):
    state["data"] = pd.read_csv("data/actis.csv")
    return state


def is_from_mail(state):
    from langchain_core.documents import Document

    with open("data/mails.json", "r") as f:
        email_database = json.load(f)

    full_text = []
    for i, page in enumerate(email_database):
        question = page["question"]
        answer = page["answer"]
        document = Document(
            page_content=f" QUESTION:\n {question}\n\n ANSWER:\n {answer}",
        )
        full_text.append(document)

    embedder = OllamaEmbeddings(model=local_embedder)

    vectorstore = Chroma()
    vectorstore.delete_collection()
    vectorstore = Chroma.from_documents(documents=full_text, embedding=embedder)
    retriever = vectorstore.as_retriever(k=3)

    docs = retriever.invoke(state["question"])
    l_docs = []
    for doc in docs:
        l_docs.append(doc.page_content)

    state["docs"] = l_docs

    return state
