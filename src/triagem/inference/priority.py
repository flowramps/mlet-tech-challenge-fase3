"""Prioridade de atendimento derivada da condição prevista.

Isto é uma regra de negócio determinística, **não** uma predição do modelo. O modelo
classifica a condição clínica; a fila de atendimento é uma política operacional aplicada
por cima dela, e a API sinaliza essa distinção no campo ``priority_source``.

A ordenação abaixo é uma convenção deste projeto, adotada para fins de demonstração do
fluxo de triagem. Ela não substitui protocolo clínico: um sistema real seria calibrado com
a equipe médica e levaria em conta sinais vitais e histórico, não só a categoria da
condição.
"""

from __future__ import annotations

PRIORITY_BY_CONDITION: dict[str, str] = {
    "cardiovascular diseases": "alta",
    "nervous system diseases": "alta",
    "neoplasms": "media",
    "digestive system diseases": "media",
    "general pathological conditions": "baixa",
}

DEFAULT_PRIORITY = "media"


def priority_for(condition: str) -> str:
    """Devolve a prioridade da condição, ou o padrão se ela for desconhecida."""
    return PRIORITY_BY_CONDITION.get(condition, DEFAULT_PRIORITY)
