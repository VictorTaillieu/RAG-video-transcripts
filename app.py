import streamlit as st

from rag import query_rag

LLM_BACKEND_TO_MODEL = {
    "ollama": "mistral:7b",
    "openai": "mistral-small-3.2-24b-instruct"
}


def main():
    st.set_page_config(page_title="RAG")
    st.title("RAG pour transcriptions vidéo")

    st.sidebar.header("Paramètres")

    llm_backend = st.sidebar.selectbox(
        "Backend LLM",
        ["ollama", "openai"],
        help="ollama = local | openai = api"
    )
    model = LLM_BACKEND_TO_MODEL[llm_backend]

    st.sidebar.info(f"Modèle : `{model}`")

    st.subheader("Requête")
    query = st.text_area(
        "Requête",
        placeholder="Posez votre question",
        label_visibility="collapsed"
    )

    run = st.button("Lancer la recherche", type="primary")

    if run:
        if not query.strip():
            st.warning("Veuillez entrer une question.")
        else:
            with st.spinner("Recherche en cours...", show_time=True):
                response, results = query_rag(query, llm_backend)

                st.subheader("Réponse")
                st.write(response)

                st.divider()

                st.subheader("Sources")
                for chunk, score in results:
                    title = chunk.metadata.get("title")
                    author = chunk.metadata.get("creator")
                    with st.expander(f"{title} - {author} (score : {score:.3f})"):
                        st.write(chunk.page_content.removeprefix("passage: "))


if __name__ == "__main__":
    main()
