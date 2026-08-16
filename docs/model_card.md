# Model Card — Classificador de Triagem de Laudos

## 1. Detalhes do modelo

| | |
|---|---|
| Tarefa | Classificação de texto em 5 condições clínicas |
| Arquitetura | TF-IDF (5.000 features) + Regressão Logística, scikit-learn |
| Versão | 1.0.0 |
| Artefato | `models/model.joblib`, 0,24 MB |
| Treinado por | `make train` ou DAG `triagem_training` (mesmo código) |

A Regressão Logística foi selecionada contra um Random Forest de 200 árvores sobre a mesma
vetorização, vencendo em f1-macro (0,587 vs 0,529 na validação), latência (0,68 ms vs
23,5 ms de p50) e tamanho de artefato (0,24 MB vs 19,2 MB). A análise completa da seleção
está no README.

## 2. Uso pretendido

Ordenar a fila de leitura de laudos, colocando os casos de provável maior risco no topo
antes de um humano abrir cada documento. O modelo prevê a **condição clínica**; a
**prioridade** é uma regra de negócio determinística aplicada por cima, sinalizada pela API
no campo `priority_source`.

**Fora de escopo:** diagnóstico, decisão clínica autônoma ou qualquer uso sem um humano na
decisão final. O sistema reordena uma fila de leitura — não substitui a leitura.

## 3. Dados de treino

[Medical Abstracts TC Corpus](https://github.com/sebischair/Medical-Abstracts-TC-Corpus):
14.438 resumos de literatura médica **em inglês**, rotulados em 5 categorias de condição
clínica. Split estratificado: 9.240 treino, 2.310 validação, 2.888 teste.

As classes são desbalanceadas em 3,2x — por isso a métrica de decisão é o f1-macro e os
classificadores usam `class_weight` balanceado. O corpus rotula condição, não urgência:
nenhum rótulo de prioridade foi sintetizado.

## 4. Desempenho (conjunto de teste, 2.888 laudos)

**Acurácia 0,579 · f1-macro 0,581** — contra 0,20 de um classificador aleatório.

O desempenho varia por classe: de f1 0,709 (`neoplasms`) a 0,403 (`general pathological
conditions`). A tabela completa por classe está no README, seção "Resultados".

Latência de inferência: p50 0,678 ms, p99 0,934 ms (sem HTTP); 2,35 ms / 3,29 ms de ponta
a ponta no container.

## 5. Limitações e cenários de falha conhecidos

- **Idioma.** O corpus é em inglês. Um laudo em português é fora de distribuição: o modelo
  responde, mas a predição não é confiável. Uso em produção exigiria corpus no idioma real
  dos laudos.
- **Classe guarda-chuva.** `general pathological conditions` tem recall 0,32: é a maior
  classe do conjunto e se sobrepõe semanticamente às outras quatro. É o teto do problema
  (classes não mutuamente exclusivas), não um defeito de ajuste.
- **Registro de texto.** O corpus é literatura científica, não laudo hospitalar. A
  transferência para o registro textual de laudos reais não foi medida.
- **Confiança mal calibrada é possível.** A probabilidade da Regressão Logística não passou
  por calibração; deve ser lida como escore ordinal, não como probabilidade verdadeira.
- **Reprodutibilidade entre máquinas.** Com seeds fixadas, o f1-macro varia na terceira
  casa decimal entre CPUs (solver `lbfgs`); a margem do gate de qualidade absorve isso.

## 6. Vieses e considerações éticas

O custo do erro é assimétrico: um caso urgente classificado como baixa prioridade é pior
do que o inverso. A política condição → prioridade deste projeto é uma **convenção de
demonstração** — um sistema real seria calibrado com a equipe médica e consideraria sinais
vitais e histórico, não só a categoria prevista. O dado processado é clínico: o serviço
não persiste o texto dos laudos, e o container roda sem privilégios.

## 7. Manutenção e retreino

Retreino semanal pela DAG `triagem_training` (ou `make train`), com gate de qualidade:
um candidato só é promovido com f1-macro ≥ 0,53 na validação. Reprovado, o run falha e o
modelo publicado **não é tocado** — um retreino ruim não derruba produção.

## 8. Monitoramento em produção

A API expõe métricas Prometheus consumidas pelo dashboard Grafana provisionado:

- **Saúde do serviço** — tráfego por rota/status, latência HTTP (p50/p95/p99), taxa de erro.
- **Saúde do modelo** — predições por condição e prioridade, distribuição da confiança e
  latência de inferência por backend.

Sinais de alerta que antecedem incidente: uma classe engolindo a distribuição de predições
ou queda sustentada da confiança — ambos visíveis no dashboard antes de qualquer erro HTTP.
