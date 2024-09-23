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

st.text_area("Mail reçu", value="", key="received_mail")


def write_center(message):
    return st.markdown(
        f'<div style="text-align: center;">{message}</div>', unsafe_allow_html=True
    )


def my_pipeline(mail):

    col1, col2 = st.columns([1, 1])

    # logs = st.expander("Expand for logs")

    with col1:
        pipeline = st.container(border=True)
        wait = st.container(border=False)
    with col2:
        answer = st.container(border=True)

    with pipeline:
        st.write("**Pipeline**")
    with answer:
        st.write("**Réponse**")

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
