import pytest

from triagem.data.prepare import CONDITION_NAMES
from triagem.inference.priority import (
    DEFAULT_PRIORITY,
    PRIORITY_BY_CONDITION,
    priority_for,
)


def test_mapa_cobre_exatamente_as_condicoes_do_modelo():
    """Guarda contra o mapa e o conjunto de rótulos saírem de sincronia."""
    assert set(PRIORITY_BY_CONDITION) == set(CONDITION_NAMES.values())


@pytest.mark.parametrize(
    ("condicao", "esperado"),
    [
        ("cardiovascular diseases", "alta"),
        ("nervous system diseases", "alta"),
        ("neoplasms", "media"),
        ("digestive system diseases", "media"),
        ("general pathological conditions", "baixa"),
    ],
)
def test_prioridade_de_cada_condicao(condicao, esperado):
    assert priority_for(condicao) == esperado


def test_condicao_desconhecida_cai_no_padrao():
    assert priority_for("condição inexistente") == DEFAULT_PRIORITY


def test_prioridades_usam_apenas_os_tres_niveis():
    assert set(PRIORITY_BY_CONDITION.values()) <= {"alta", "media", "baixa"}
