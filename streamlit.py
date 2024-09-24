"""
# My first app
Here's our first attempt at using data to create a table:
"""

import streamlit as st
import pandas as pd

from pipeline.first_classifier import first_classifier
from pipeline.routes import continue_route
from pipeline.output import as_output

chain = first_classifier | continue_route | as_output

st.set_page_config(layout="wide")

example = """Bonjour,

Je veux avoir des informations sur la disponibilité des chambres familliales à partir du 1er juillet et pour une semaine.

Cordialement,

Steven Stevenson"""

st.text_area("Mail reçu", value=example, key="received_mail", height=200)


def write_center(message):
    return st.markdown(
        f'<div style="text-align: center;">{message}</div>', unsafe_allow_html=True
    )


def my_pipeline(mail):

    col1, col2 = st.columns([1, 1])

    # logs = st.expander("Expand for logs")

    with col1:
        pipeline = st.container(border=True)
    with col2:
        answer = st.container(border=True)

    with pipeline:
        st.write("**Pipeline**")
    with answer:
        st.write("**Réponse**")
        wait = st.container(border=False)

    with wait:
        with st.spinner("Traitement de la question..."):
            first = first_classifier.invoke(mail)
            with pipeline:
                st.write("Détection d'intention : ", "**" + first["datasource"] + "**")

    with wait:
        with st.spinner("Accès aux bases de données..."):
            second = continue_route.invoke(first)
            with pipeline:
                if first["datasource"] == "disponibilite_chambres":
                    st.divider()
                    st.write("Date de début détectée : ", second["start_date"])
                    st.write("Date de fin détectée : ", second["end_date"])
                    st.write("Semaine de départ :", second["week_beg"])
                    st.write("Semaine de fin :", second["week_end"])
                    st.write("Type de chambre :", second["room_type"])
                    st.write("Données : ", second["data"])
                    st.write("Données finales : ", second["final"])
                if first["datasource"] == "activite":
                    st.divider()
                    st.write("Données : ", second["data"])
                if first["datasource"] == "ouverture_accueil":
                    st.divider()
                    st.write("Données : ", second["data"])
                if first["datasource"] == "other":
                    st.divider()
                    st.write("**Source 1 :**", second["docs"][0])
                    st.write("**Source 2 :**", second["docs"][1])
                    st.write("**Source 3 :**", second["docs"][2])

    with wait:
        with st.spinner("Rédaction de la réponse..."):
            third = as_output.invoke(second)
            # with logs:
            #    st.write("Third step : ", third)

    with answer:
        st.write(third)


if st.button("Soumettre", key="submit_button"):
    mail = st.session_state.received_mail

    my_pipeline(mail)
