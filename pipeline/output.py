from langchain_community.chat_models import ChatOllama
from langchain_core.output_parsers import StrOutputParser

from langchain_core.runnables import RunnableLambda

local_llm = "llama3.1"
llm = ChatOllama(model=local_llm, temperature=0)


def as_output_fct(state):
    BASE_PROMPT = """Tu vas me rédiger un mail assez long et cordial à un client de l'hotel en signant MozambFun. Je veux que tu me donnes uniquement le mail à envoyer en réponse."""
    if state["datasource"] == "activite":
        PROMPT = f"""
        Le client demande à réaliser une activité. Voici son message: {state["question"]}.
        Tu vas lui répondre en utilisant la base de données suivante : {state["data"].to_string()}.
        """
    elif state["datasource"] == "disponibilite_chambres":
        PROMPT = f"""
        Le client demande la disponibilité des chambres de l'hotel. Voici son message : {state["question"]}."
        Tu vas lui répondre en utilisant la base de données suivante : {state["final"].to_string()}.
        """
    elif state["datasource"] == "ouverture_accueil":
        PROMPT = f"""
        Le client demande les horaires d'ouverture de l'accueil. Voici son message : {state["question"]}.
        Tu vas lui répondre en utilisant la base de données suivante : {state["data"].to_string()}."""
    elif state["datasource"] == "other":
        PROMPT = f"""
        Le client demande des informations supplémentaires générales. Voici son message : {state["question"]}.
        Tu vas lui répondre en utilisant exemples de question/réponses suivants : {" ".join(state["docs"])}.
        """

    PROMPT = BASE_PROMPT + PROMPT

    output_chain = llm | StrOutputParser()

    return output_chain.invoke(PROMPT)


as_output = RunnableLambda(as_output_fct)
