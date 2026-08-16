# Triagem Automática de Laudos Médicos

Classificação de texto clínico servida por API REST, com pipeline de treino orquestrado,
observabilidade e otimização de latência.

O sistema recebe o texto livre de um laudo, prevê a categoria da condição clínica e devolve
a prioridade de atendimento correspondente — em menos de 3 ms de ponta a ponta.

---

## O problema

Um laudo que fica na fila é um diagnóstico que atrasa. Em um pronto-socorro, a ordem em que
os exames são lidos determina quem é atendido primeiro, e essa ordem hoje depende de alguém
ler cada texto. Automatizar a categorização não substitui o julgamento clínico: ela ataca o
gargalo de triagem, colocando os casos de maior risco no topo da fila antes que um humano
precise abrir cada documento.

Isso impõe duas restrições ao sistema. A resposta precisa ser **síncrona** — chegar enquanto
a decisão ainda está sendo tomada, não em um relatório noturno. E precisa ser **rápida**, com
latência estável mesmo sob carga, porque uma cauda longa de p99 significa exatamente o caso
que ficou para trás.

## Dados

O corpus é público e real: [Medical Abstracts TC Corpus](https://github.com/sebischair/Medical-Abstracts-TC-Corpus),
com 14.438 resumos de literatura médica rotulados por categoria de condição clínica.

| Partição | Amostras |
|---|---|
| Treino | 11.550 (9.240 treino + 2.310 validação) |
| Teste | 2.888 |

| Rótulo | Condição | Amostras (treino) |
|---|---|---|
| 1 | neoplasms | 2.530 |
| 2 | digestive system diseases | 1.195 |
| 3 | nervous system diseases | 1.540 |
| 4 | cardiovascular diseases | 2.441 |
| 5 | general pathological conditions | 3.844 |

Duas características dos dados moldaram decisões técnicas:

**As classes são desbalanceadas em 3,2x** (de 1.195 a 3.844 amostras). Por isso a métrica de
decisão é o **f1-macro**, não a acurácia: acurácia recompensa acertar a classe majoritária e
esconde falhas nas minoritárias. Pelo mesmo motivo, os classificadores usam `class_weight`
balanceado e o split de validação é estratificado.

**O corpus rotula condição clínica, não urgência.** O modelo prevê a condição; a prioridade
de atendimento é uma regra de negócio determinística aplicada por cima, e a API sinaliza essa
distinção explicitamente no campo `priority_source`. Nenhum rótulo foi sintetizado.

## Arquitetura de deploy em nuvem

### Batch ou tempo real?

A triagem responde a um laudo por vez, no instante em que ele é emitido, e o resultado
altera a fila de atendimento. Um processamento em lote noturno entregaria a classificação
depois de a decisão clínica já ter sido tomada — o que torna o resultado correto e inútil.

A carga também não é uniforme: hospitais têm picos por turno e vales de madrugada. Uma
instância dimensionada para o pico fica ociosa a maior parte do dia.

Isso define o perfil: **inferência síncrona, em container, com escala elástica e capacidade
de escalar a zero.**

### Comparativo

| Provedor | Serviço | A favor | Contra |
|---|---|---|---|
| **GCP** | Cloud Run | Escala a zero; cobrança por requisição; recebe o container sem alteração; TLS e URL pública gerenciados | Cold start após ociosidade |
| **AWS** | App Runner | Integração nativa com o ecossistema; abstrai a infraestrutura | Não escala a zero — custo fixo mesmo ocioso |
| **AWS** | Lambda + container | Escala a zero de fato; granularidade de cobrança por ms | Limite de tamanho de imagem e cold start mais sensível a dependências pesadas |
| **Azure** | Container Apps | Escala a zero; autoscaling por métrica customizada via KEDA | Menos familiaridade operacional do time |

### Decisão: Cloud Run

**Custo.** Com carga intermitente, pagar por requisição em vez de por instância parada é a
diferença entre custo proporcional ao uso e custo fixo. É o argumento decisivo aqui — App
Runner cobraria 24h por dia para atender picos de algumas horas.

**Operação.** A mesma imagem validada localmente sobe sem modificação. Não há um artefato
"de nuvem" diferente do artefato testado, o que elimina uma classe inteira de defeitos de
ambiente.

**Segurança.** TLS e URL pública são gerenciados, sem configurar balanceador ou certificado
— menos superfície para errar em um serviço que trafega dado clínico.

**O custo dessa escolha** é o cold start: após ociosidade, a primeira requisição paga a
subida do container. Medimos esse efeito localmente — a primeira inferência de um processo
frio levou 43 ms contra 0,86 ms das seguintes. A mitigação é `min-instances=1` na janela de
pico, aceitando o custo de uma instância morna nesse período; fora dela, escala a zero.

### Estimativa de custo

Premissa: 50.000 laudos/mês, 3 ms de CPU por requisição, 512 MB de memória, `min-instances=1`
por 12h/dia.

O volume de requisições em si cai dentro da camada gratuita do Cloud Run com folga — 50 mil
requisições é uma fração dos 2 milhões mensais gratuitos, e o tempo de CPU consumido
(150 segundos/mês) é desprezível. O custo real vem da instância morna mantida no horário de
pico, na ordem de poucos dólares mensais. **O dimensionamento é dominado pela política de
disponibilidade, não pelo volume de inferência** — o que reforça a escolha de um serviço que
permite desligar essa política fora do pico.

### Preparação para deploy

O `Dockerfile` produz uma imagem que roda como usuário sem privilégios, expõe a porta 8000 e
declara `HEALTHCHECK` — os três requisitos que Cloud Run, App Runner e Container Apps esperam
de um container gerenciado. A imagem construída e validada localmente é a mesma que subiria.

## Resultados

### Seleção do modelo

Dois candidatos disputaram a vaga, sobre a **mesma** vetorização TF-IDF — só o classificador
mudou, o que isola a variável em comparação:

| Candidato | f1-macro (validação) | Latência p50 | Artefato |
|---|---|---|---|
| Random Forest (200 árvores) | 0,529 | 23,5 ms | 19,2 MB |
| **Regressão Logística** | **0,587** | **0,68 ms** | **0,24 MB** |

A Regressão Logística venceu nos três eixos. O resultado não é acidental: com TF-IDF, cada
split de uma árvore sorteia ~70 de 5.000 features, e quase todas valem zero em um documento
qualquer — a floresta gasta capacidade em ruído. O modelo linear opera sobre todo o vetor
esparso de uma vez.

A seleção usa a partição de **validação**. O conjunto de teste é gasto uma única vez, para
reportar o número final.

### Desempenho do campeão (conjunto de teste, 2.888 laudos)

**Acurácia 0,579 · f1-macro 0,581** — contra 0,20 de um classificador aleatório em 5 classes.

| Condição | Precisão | Recall | f1 | Amostras |
|---|---|---|---|---|
| neoplasms | 0,676 | 0,746 | 0,709 | 633 |
| cardiovascular diseases | 0,657 | 0,718 | 0,686 | 610 |
| nervous system diseases | 0,479 | 0,665 | 0,557 | 385 |
| digestive system diseases | 0,473 | 0,662 | 0,552 | 299 |
| general pathological conditions | 0,541 | 0,320 | 0,403 | 961 |

A classe `general pathological conditions` tem recall de 0,32 e é a maior do conjunto. Ela é
uma categoria guarda-chuva que se sobrepõe semanticamente às outras quatro — um resumo sobre
inflamação cardíaca pertence legitimamente a duas delas. Esse é o teto do problema, não um
defeito de ajuste: as classes não são mutuamente exclusivas.

**Sobre reprodutibilidade:** as sementes estão fixadas, mas os valores variam na terceira
casa decimal entre máquinas diferentes — o mesmo commit rendeu f1-macro 0,5812 localmente e
0,5802 no runner do CI. A causa é o solver `lbfgs`, sensível à ordem de acumulação em ponto
flutuante, que depende da biblioteca de álgebra linear e do conjunto de instruções da CPU. A
variação não altera a escolha do campeão nem a decisão do gate, e é justamente por isso que a
margem do piso de qualidade é de 0,05 e não de 0,005.

### Latência (baseline)

Duas medições distintas, que não devem ser confundidas:

| Medição | p50 | p95 | p99 | Throughput |
|---|---|---|---|---|
| Inferência do modelo (sem HTTP) | 0,678 ms | 0,838 ms | 0,934 ms | 1.451 req/s |
| HTTP ponta a ponta, no container | 2,35 ms | 3,06 ms | 3,29 ms | — |

A diferença de ~1,7 ms é o custo de rede, serialização e do servidor — não do modelo. Separar
as duas é o que permite atribuir corretamente qualquer ganho de otimização futura.

Ambas medidas com 20 requisições de aquecimento descartadas. Sem descartá-las, a primeira
inferência de um processo frio (43 ms) distorceria a média em uma ordem de grandeza.

## Automação

### Esteira de verificação

| Workflow | Gatilho | O que verifica |
|---|---|---|
| `ci.yml` — `lint` | push e PR | Regras de lint e formatação (ruff) |
| `ci.yml` — `test` | push e PR | Suíte completa com cobertura |
| `ci.yml` — `build` | push e PR | Treina o modelo, constrói a imagem e confirma que o container responde em `/health` |
| `security.yml` | push, PR e semanalmente | Vulnerabilidades CRITICAL e HIGH na imagem (Trivy) |
| `cd.yml` | manual | Publica a imagem no registry |

O job `build` treina o modelo antes de construir a imagem, porque a imagem embute o
artefato. O efeito colateral é útil: cada push reexecuta o pipeline de treino inteiro, então
uma quebra nele aparece no CI e não na hora do deploy.

O scan de segurança roda também por agendamento semanal. Uma imagem que não mudou fica
insegura sozinha — CVEs novos são publicados contra dependências que já estão lá.

O `cd.yml` só roda por acionamento manual. A decisão desta fase foi preparar o deploy em
nuvem sem provisionar recursos, então o gatilho automático está desligado de propósito.

### Pipeline de treino

A DAG `triagem_training` encadeia cinco tarefas:

```
ingestao -> preparo -> treino -> selecao -> publicacao
```

| Tarefa | Responsabilidade |
|---|---|
| `ingestao` | Baixa o corpus público, reaproveitando o que já está em disco |
| `preparo` | Valida o schema e separa a validação, estratificada |
| `treino` | Treina os dois candidatos e pontua cada um na validação |
| `selecao` | Escolhe o campeão e mede o desempenho no conjunto de teste |
| `publicacao` | Promove o campeão, sujeito ao gate de qualidade |

A DAG é um invólucro fino: cada tarefa delega para uma função de
`src/triagem/pipeline/steps.py`, testada isoladamente na suíte. O que a DAG declara é a
topologia — ordem, agendamento, retentativa — não a lógica. Consequência prática: `make
train` e o Airflow executam **o mesmo código**, então o treino local e o orquestrado não
podem divergir.

As etapas trocam apenas tipos serializáveis (`str`, `float`, `dict`), porque no Airflow esse
valor trafega por XCom, onde um `Path` não sobreviveria. Os datasets vão para disco em vez de
trafegarem entre tarefas: não caberiam em XCom, e materializá-los permite inspecionar o que
cada etapa produziu quando algo falha.

### Gate de qualidade

A tarefa `publicacao` só promove o modelo se o f1-macro atingir `min_f1_macro = 0,53`.

Esse número não é arbitrário: é o f1-macro de validação do campeão (0,587) menos uma margem
de 0,05, que absorve a variação natural entre retreinos sem transformar o gate em enfeite
que sempre passa.

Reprovando, a tarefa falha e **o artefato publicado não é tocado** — o modelo que já está
atendendo continua no ar. Um retreino ruim não derruba produção. O comportamento é coberto
por teste (`test_publish_preserva_o_modelo_anterior_ao_reprovar`) e foi verificado na prática
forçando um piso impossível: a DAG falha na publicação e o `model.joblib` permanece
inalterado.

## Observabilidade

### Métricas expostas

A API expõe três métricas em `/metrics`, no formato Prometheus:

| Métrica | Tipo | O que mede |
|---|---|---|
| `triagem_http_requests_total` | contador | Requisições por método, rota e status da resposta |
| `triagem_http_request_duration_seconds` | histograma | Duração HTTP, do recebimento ao envio da resposta |
| `triagem_inference_duration_seconds` | histograma | Só a inferência do modelo, sem HTTP, por backend |

Throughput e taxa de erro não são métricas próprias: derivam do contador na consulta
(`rate`), o que elimina a chance de séries redundantes divergirem entre si.

Três decisões de instrumentação merecem registro:

- **O rótulo de rota é a rota declarada** (`/predict`), nunca o path bruto. Um scanner
  varrendo URLs aleatórias criaria uma série temporal nova por URL até esgotar a memória
  do processo — cardinalidade controlada é o que faz a métrica sobreviver exposta à rede.
- **Os buckets dos histogramas começam em milissegundos.** Os padrão da biblioteca começam
  em 5 ms; a distribuição inteira desta API (p99 ~3 ms) cairia no primeiro bucket e os
  percentis sairiam sem resolução.
- **O scrape do `/metrics` não conta como tráfego.** O Prometheus raspa a cada 5 s: se o
  scrape contasse, o painel de total de requisições subiria sozinho com a API parada.

A métrica de inferência leva para produção a mesma separação do benchmark local — modelo
versus HTTP — rotulada por backend. É ela que permitirá comparar o motor atual com um
otimizado medindo o sistema real, não um script paralelo.

### Stack local

O `docker-compose.yml` sobe a API, o Prometheus (raspando a cada 5 s) e o Grafana já
provisionado por arquivo: fonte de dados e dashboard entram por volume, sem nenhum clique
de configuração manual. O JSON do dashboard é versionado em
`docker/grafana/dashboards/triagem-api.json` — o que o avaliador vê é o que está no git.

### Dashboard

| Painel | Pergunta que responde |
|---|---|
| Total de requisições | Quanto tráfego o serviço atendeu? |
| Taxa de erro (não-2xx) | Que fração das respostas falhou nos últimos 5 minutos? |
| Requisições por segundo | Qual o throughput, rota a rota, agora? |
| Latência HTTP do `/predict` (p50/p95/p99) | O tempo de resposta está estável? A cauda cresceu? |
| Inferência do modelo por backend (p95) | Quanto do tempo é modelo — e como os backends se comparam? |

## Como executar

**Pré-requisitos:** Python 3.12, [Poetry](https://python-poetry.org/) 2.x e Docker.

Na primeira execução, nesta ordem:

```bash
make install     # dependências e hooks de pre-commit
make data        # baixa o corpus público (~17 MB, idempotente)
make train       # treina os dois candidatos e promove o campeão
```

Com o modelo treinado, os demais alvos ficam disponíveis:

```bash
make evaluate    # avalia no conjunto de teste -> metrics/metrics.json
make bench       # mede a latência de inferência
make help        # lista todos os alvos
```

### Testes e lint

```bash
make test        # suíte completa com cobertura (pytest)
make lint        # estilo e formatação (ruff)
```

São os mesmos comandos que a esteira de CI executa a cada push — passar localmente é passar
na esteira. O `make install` também instala os hooks de pre-commit, que verificam o código
antes de cada commit. A suíte não depende de `make train`: os testes usam classificadores
falsos e dados sintéticos, então rodam em segundos e sem rede.

### API local

```bash
make api         # http://localhost:8000/docs
```

### API em container

```bash
make docker-build   # exige `make train` antes: a imagem embute o modelo
make docker-run
```

### Pipeline de treino no Airflow

```bash
make airflow-up      # http://localhost:8080 — usuário admin, senha admin
make airflow-test    # executa a DAG de ponta a ponta (~40s)
make airflow-down
```

O Airflow sobe em **um único container** (`LocalExecutor` com SQLite): uma DAG linear de
cinco tarefas não justifica um container de banco só para a demonstração.

Ele tem compose próprio, separado da stack de observabilidade, e a razão é prática: quem
quiser ver o dashboard não precisa subir o Airflow, e quem quiser rodar a DAG não precisa
subir Prometheus e Grafana. Cada `docker compose` sobe só o que a tarefa exige.

O container roda com o UID do host, o que faz os artefatos gravados pela DAG (`models/`,
`metrics/`, `data/interim/`) pertencerem a quem executou — sem isso, escreveria como um
usuário interno e ficaria sem permissão nos diretórios montados.

`make airflow-test` usa `exec`, não `run`: o banco de metadados vive dentro do container que
o `standalone` migrou no startup, então é preciso subir o Airflow antes.

### Stack de observabilidade

```bash
make monitoring-up     # API + Prometheus + Grafana (exige `make train` antes)
make traffic           # ~200 requisições para movimentar os painéis
make monitoring-down
```

| Serviço | Endereço |
|---|---|
| API | http://localhost:8000/docs |
| Prometheus | http://localhost:9090 |
| Grafana | http://localhost:3000 — usuário `admin`, senha `admin` |

O dashboard **Triagem — API de Laudos** já aparece provisionado no Grafana. O `make traffic`
alterna laudos válidos e, a cada décima requisição, envia um texto curto demais — rejeitado
com 422 — para o painel de taxa de erro ter o que exibir.

### Exemplo de requisição

```bash
curl -X POST localhost:8000/predict \
  -H 'Content-Type: application/json' \
  -d '{"text":"Coronary artery bypass grafting in patients with severe left ventricular dysfunction and ejection fraction below 30 percent undergoing myocardial revascularization."}'
```

```json
{
  "condition": "cardiovascular diseases",
  "confidence": 0.866,
  "priority": "alta",
  "priority_source": "regra_de_negocio",
  "model_version": "1.0.0",
  "backend": "sklearn",
  "inference_ms": 1.04
}
```

`priority_source` existe para deixar explícito que a prioridade **não** é uma predição do
modelo: é uma regra determinística documentada em `src/triagem/inference/priority.py`, uma
convenção deste projeto para demonstrar o fluxo de triagem. Um sistema real calibraria essa
política com a equipe médica e consideraria sinais vitais e histórico, não apenas a categoria.

### Endpoints

| Método | Rota | Descrição |
|---|---|---|
| `GET` | `/health` | Estado do serviço e identificação do modelo carregado |
| `POST` | `/predict` | Classifica um laudo e devolve a prioridade |
| `GET` | `/metrics` | Métricas no formato Prometheus |
| `GET` | `/docs` | Documentação interativa (Swagger) |

## Estrutura do projeto

```
src/triagem/
├── config.py                   configuração por variáveis de ambiente
├── data/
│   ├── download.py             obtenção idempotente do corpus
│   └── prepare.py              validação de schema e split estratificado
├── model/
│   ├── train.py                candidatos, treino e seleção do campeão
│   └── evaluate.py             métricas e matriz de confusão
├── inference/
│   ├── base.py                 Protocol Classifier + Prediction
│   ├── sklearn_backend.py      motor de inferência scikit-learn
│   ├── factory.py              escolha do backend por configuração
│   └── priority.py             regra de negócio condição -> prioridade
├── api/
│   ├── main.py                 aplicação FastAPI
│   ├── metrics.py              instrumentação Prometheus
│   └── schemas.py              contratos de entrada e saída
├── pipeline/
│   ├── steps.py                etapas do treino, encadeáveis e serializáveis
│   └── training.py             execução local do pipeline completo
└── bench/
    └── latency.py              medição de latência com percentis

dags/
└── triagem_training_dag.py     topologia da DAG de treino no Airflow

docker/
├── airflow/Dockerfile          imagem do Airflow com as dependências de ML do projeto
├── prometheus/prometheus.yml   alvo e intervalo de coleta de métricas
└── grafana/                    fonte de dados e dashboard provisionados por arquivo
```

A fronteira que sustenta o projeto é o Protocol `Classifier`: a API depende dele, não de
scikit-learn. Trocar o motor de inferência é trocar uma variável de ambiente, o que torna a
comparação entre backends uma medição do sistema real em vez de um script paralelo.

## Roadmap

- [x] **Etapa 1** — Modelo, API FastAPI, container e baseline de latência
- [x] **Etapa 2** — CI/CD no GitHub Actions e DAG de treino no Airflow
- [x] **Etapa 3** — Métricas Prometheus e dashboard Grafana
- [ ] **Etapa 4** — Exportação ONNX, quantização e comparativo de latência
