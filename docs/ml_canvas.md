# ML Canvas — Triagem de Laudos Médicos

Documento de arquitetura de decisão do projeto, no formato de 10 blocos do
[Machine Learning Canvas](https://www.louisdorard.com/machine-learning-canvas) (Louis Dorard),
mesmo template usado nos dois desafios anteriores do grupo. Cada bloco marca seu estado:
✅ implementado · 💡 hipótese/decisão consciente de escopo · 🔜 extensão futura, fora do
enunciado desta fase.

## 0. Objetivo / Proposta de valor (bloco central)

Um hospital recebe laudos médicos (resumos clínicos em texto) num volume que excede a
capacidade de leitura imediata. A proposta de valor é **reordenar a fila de leitura** para que
os casos de maior risco cheguem primeiro à triagem humana — não diagnosticar, não decidir
tratamento, não substituir o profissional. O sistema classifica a condição clínica por texto e
converte essa classificação numa prioridade operacional (alta/média/baixa) via regra de
negócio. ✅

## Blocos de aprendizado (como o modelo é construído)

### 1. Fontes de dados ✅

[Medical Abstracts TC Corpus](https://github.com/sebischair/Medical-Abstracts-TC-Corpus):
14.438 resumos de literatura médica em inglês, rotulados em 5 categorias de condição clínica
(`neoplasms`, `digestive system diseases`, `nervous system diseases`,
`cardiovascular diseases`, `general pathological conditions`). Escolhido por já vir separado
em splits de treino/teste e por atender ao critério do enunciado (texto + rótulo, ≥ 2.000
amostras).

### 2. Coleta de dados ✅

Download idempotente via HTTP (`triagem.data.download`): arquivos já presentes em disco são
reaproveitados, o que mantém `make train` e o CI reprodutíveis sem depender da rede a cada
execução. Não há coleta contínua — o corpus é estático, e o retreino semanal da DAG existe
para exercitar o caminho de retreino, não para incorporar dado novo.

### 3. Features / Engenharia de atributos ✅

TF-IDF sobre o texto do laudo: 5.000 features, uni+bigramas, `min_df=2`, `sublinear_tf`,
normalização Unicode. Nenhuma feature estruturada (idade, sinais vitais) — o corpus só
oferece texto, e é isso que o classificador de triagem do enunciado pede.

### 4. Construção dos modelos ✅

Dois candidatos disputam a vaga sobre a mesma vetorização — Random Forest (200 árvores) e
Regressão Logística —, ambos com `class_weight` balanceado por causa do desbalanceamento de
3,2x entre a classe mais e a menos frequente. Campeão: Regressão Logística, vencedora em
f1-macro, latência e tamanho de artefato (ver [README](../README.md#seleção-do-modelo)).
Limitação conhecida, não resolvida: a EDA (`notebooks/01_eda.ipynb`) mede ~26% dos textos
únicos do corpus repetidos sob rótulos diferentes, impondo um teto teórico de ~78% de acurácia
independente do modelo escolhido.

## Blocos de predição (como o modelo é usado)

### 5. Tarefa de predição ✅

Classificação multiclasse (5 condições clínicas) a partir do texto do laudo. Saída: a
condição prevista e a confiança do modelo nela.

### 6. Decisões / Uso das predições ✅ · 💡

A condição prevista alimenta `priority_for()` (`src/triagem/inference/priority.py`), uma
regra determinística — não uma predição — que mapeia condição → prioridade
(`cardiovascular`/`nervous system` → alta, `neoplasms`/`digestive` → média, guarda-chuva →
baixa). A API expõe as duas informações separadamente (`priority_source`), para que a
distinção entre o que o modelo prevê e o que o negócio decide fique visível ao consumidor.
A calibração real dessa política com a equipe médica — considerando sinais vitais e histórico,
não só a categoria — é hipotética: este projeto usa uma convenção de demonstração.

### 7. Realização das predições ✅

Servido em **tempo real**, síncrono, via API FastAPI — não em lote. Decisão de arquitetura
documentada na Etapa 1: um hospital precisa da prioridade no momento em que o laudo chega, não
num relatório do dia seguinte. Inferência sub-milissegundo (p50 0,68 ms), API containerizada
para deploy em serviço gerenciado (Cloud Run/App Runner/Container Apps).

### 8. Avaliação offline ✅

f1-macro como métrica de decisão (não acurácia, dado o desbalanceamento), medido numa
partição de teste nunca usada na seleção do campeão. O gate de promoção do pipeline de treino
vai além da métrica técnica: exige também recall mínimo na prioridade "alta" — a **métrica de
negócio** da triagem, porque rebaixar um caso realmente urgente pesa mais do que confundir
duas condições que já dariam na mesma prioridade — e que o candidato supere o modelo hoje em
produção nos dois eixos, não só um piso fixo. Todo run fica registrado em
`metrics/training_history.jsonl`, promovido ou não.

### 9. Avaliação ao vivo e monitoramento ✅ · 🔜

Prometheus + Grafana instrumentam tanto a saúde do serviço (tráfego, latência, taxa de erro)
quanto a saúde do modelo (predições por condição/prioridade, distribuição de confiança,
latência de inferência por backend) — sinais que antecipam drift antes de virar incidente. O
que cada painel significa, limiar de alerta e playbook de resposta está no
[plano de monitoramento](monitoring_plan.md).
**Não implementado:** loop de feedback formal (rótulo real do laudo depois de lido por um
humano) e detecção estatística de drift (KS/PSI) — extensões naturais, presentes no repo
irmão `grupo4`, mas fora do que o enunciado desta fase pede. Ambas as lacunas estão detalhadas
na seção "Limitações conhecidas" do plano de monitoramento.

## Resumo do estado atual

| Bloco | Estado |
|---|---|
| Objetivo, Tarefa de predição, Features | ✅ Definidos e implementados |
| Fontes de dados, Coleta, Construção dos modelos | ✅ Pipeline reprodutível (`make train` / DAG Airflow), 2 candidatos comparados |
| Avaliação offline | ✅ f1-macro + recall de prioridade alta (métrica de negócio) + histórico de execuções |
| Decisões (condição → prioridade) | ✅ Regra determinística e auditável · 💡 calibração com equipe médica é hipotética |
| Realização das predições | ✅ API síncrona em tempo real, containerizada |
| Avaliação ao vivo / monitoramento | ✅ Prometheus + Grafana · 🔜 loop de feedback e detecção de drift ainda não existem |