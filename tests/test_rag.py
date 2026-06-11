from langchain_ollama import OllamaLLM

from rag import query_rag

EVAL_PROMPT = """Tu es un évaluateur strict et factuel.

Règles :
- Compare la RÉPONSE ATTENDUE et la RÉPONSE OBTENUE.
- Considère la réponse correcte si elle contient les mêmes informations factuelles essentielles,
  même si la formulation est différente.
- La réponse est incorrecte si elle :
  - contredit la réponse attendue
  - invente des informations absentes
  - omet une information essentielle
- Si la réponse attendue est "Je ne sais pas", la réponse est correcte uniquement si elle dit aussi "Je ne sais pas".

RÉPONSE ATTENDUE :
{expected_response}

RÉPONSE OBTENUE :
{actual_response}

Réponds uniquement par : true ou false.
"""


def query_and_validate(question: str, expected_response: str) -> bool:
    response = query_rag(question, "openai")[0]
    prompt = EVAL_PROMPT.format(
        expected_response=expected_response, actual_response=response
    )

    model = OllamaLLM(model="mistral", temperature=0, num_predict=32)
    evaluation_results = model.invoke(prompt)
    final_result = evaluation_results.strip().lower()

    print(prompt)
    if "true" in final_result:
        print("\033[92m" + f"Response: {final_result}" + "\033[0m")
        return True
    elif "false" in final_result:
        print("\033[91m" + f"Response: {final_result}" + "\033[0m")
        return False
    else:
        raise ValueError("Cannot determine if true or false")


def test_homophilie() -> None:
    assert query_and_validate(
        question="Qu’est-ce que l’homophilie et comment se manifeste-t-elle dans les réseaux sociaux ?",
        expected_response="L’homophilie est la tendance des individus à se lier avec des personnes qui leur ressemblent, notamment en termes d’âge ou de genre. Dans les réseaux sociaux, cela se traduit par des groupes où les amis ont en moyenne des caractéristiques similaires.",
    )


def test_jugement_majoritaire() -> None:
    assert query_and_validate(
        question="Quel est le principe du jugement majoritaire ?",
        expected_response="Le jugement majoritaire consiste à attribuer une mention à chaque candidat plutôt qu’un vote binaire. Le candidat est ensuite évalué selon sa mention majoritaire, c’est-à-dire la plus haute mention qu’une majorité de votants est prête à lui accorder.",
    )


def test_budget_suisse() -> None:
    assert query_and_validate(
        question="Pourquoi la Suisse affiche-t-elle des excédents budgétaires sans augmenter fortement les impôts ?",
        expected_response="Parce qu’elle repose sur un système économique public performant et une discipline budgétaire stricte, les excédents étant principalement utilisés pour rembourser la dette et constituer des réserves.",
    )


def test_metaphore_cuisine() -> None:
    assert query_and_validate(
        question="Quelle métaphore culinaire est utilisée pour décrire le rôle des outils d’IA dans le développement informatique ?",
        expected_response="Les outils d’IA sont comparés aux commis et chefs de partie dans une brigade de cuisine : ils peuvent exécuter efficacement des tâches techniques, mais nécessitent un chef expérimenté pour encadrer les décisions complexes et la vision d’ensemble.",
    )


def test_bonheur() -> None:
    assert query_and_validate(
        question="Quels sont les trois piliers du bien-être et comment interagissent-ils ?",
        expected_response="Les trois piliers sont l’alignement avec ses valeurs, la progression personnelle (notamment dans une passion choisie) et les relations sociales. Ils interagissent de manière synergique : progresser dans une passion favorise les rencontres avec des personnes similaires, ce qui renforce les relations authentiques et alimente un cercle vertueux de motivation et de bien-être.",
    )
