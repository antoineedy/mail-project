from langchain_community.chat_models import ChatOllama
from langchain_core.output_parsers import StrOutputParser

from langchain_core.runnables import RunnableLambda

local_llm = "llama3.1"
llm = ChatOllama(model=local_llm, temperature=0)


def as_output_fct(state):
    question = state["question"]
    reponse = state["response"]
    print("-----------------")
    print(f"Question : {question}")
    print(f"Réponse : {reponse}")
    print("-----------------")
    PROMPT = f"""Je cherche à réécire un mail que j'ai rédigé, en faisant bien attention à utiliser les règles de politesse et de développer tes réponses. Tu vas signer "L'équipe de la résidence MozambFun".
    Tu vas UNIQUEMENT réécrire le mail, sans ajouter de contenu.

    Question : {question}

    Réponse : {reponse} 

    Je ne peux pas refuser de répondre. Voici une proposition de réponse :"""

    output_chain = llm | StrOutputParser()

    return output_chain.invoke(PROMPT)


as_output = RunnableLambda(as_output_fct)
