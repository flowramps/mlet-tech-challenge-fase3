# Plano de Monitoramento — Triagem de Laudos

Complementa o dashboard Grafana (`docker/grafana/dashboards/triagem-api.json`) com o que os 7
painéis não dizem sozinhos: o que cada métrica significa, que valor conta como alerta, e o que
fazer quando ele dispara. Um painel sem esse contexto é só um gráfico bonito — quem estiver de
plantão precisa saber, sem abrir o código, se o número que está vendo é normal ou é incidente.

## 1. Objetivo

Detectar degradação de **serviço** (latência, taxa de erro) e de **modelo** (confiança,
distribuição de predições) antes que vire reclamação da fila de triagem — não só ter os
gráficos disponíveis. Ver também "Avaliação ao vivo e monitoramento" no
[ML Canvas](ml_canvas.md).

## 2. Métricas monitoradas

### 2.1 Saúde do serviço

| Métrica Prometheus | Painel Grafana | Alerta sugerido |
|---|---|---|
| `triagem_http_requests_total` | Total de requisições · Taxa de erro (não-2xx) | qualquer `5xx` sustentado (bug real); `4xx` acima de 2x a taxa histórica (upstream mandando laudo malformado) |
| `triagem_http_request_duration_seconds` | Latência HTTP do `/predict` (percentis) | p99 > 50 ms sustentado por 5 min — a baseline medida é p50 2,35 ms / p99 3,29 ms (README, "Latência"); 50 ms já é uma ordem de grandeza acima, margem generosa contra ruído |

`4xx` não é sempre incidente: `make traffic` injeta laudos curtos de propósito (1 a cada 10
requisições) para o painel de erro ter o que mostrar — isso é o **baseline esperado** em
demonstração, não um alerta. O que importa é o desvio da taxa histórica, não o valor absoluto.

### 2.2 Saúde do modelo

| Métrica Prometheus | Painel Grafana | Alerta sugerido |
|---|---|---|
| `triagem_prediction_confidence` | Confiança das predições (p50/p90) | p50 sustentado abaixo de 0,40 (aviso) ou 0,25 (crítico) |
| `triagem_predictions_total{condition}` | Predições por condição | uma condição concentrando > 50% das predições, sustentado |
| `triagem_inference_duration_seconds{backend}` | Inferência do modelo por backend (p95) | p95 > 5 ms sustentado — baseline é p50 0,68 ms / p99 0,93 ms (README) |

Os limiares de confiança e concentração não são chutados — vêm do comportamento real do
campeão no conjunto de teste (medido para este documento):

- **Confiança:** p50 = 0,544 · p10 = 0,350 · p90 = 0,832. O piso de 0,20 é o chute uniforme
  entre 5 classes (mesmo raciocínio do `CONFIDENCE_BUCKETS` em `src/triagem/api/metrics.py`) —
  um p50 caindo para perto disso significa que o modelo está tão perdido quanto sorteio.
- **Concentração:** nenhuma condição passa de 24,1% das predições hoje (`neoplasms`, a maior).
  Mesmo a classe mais frequente do corpus real (`general pathological conditions`, 33,3% dos
  rótulos verdadeiros) fica abaixo de 50% — um valor sustentado acima disso é desvio real, não
  ruído da distribuição natural das classes.

### 2.3 Métrica de negócio — o que NÃO é monitorado ao vivo

`priority_recall_alta` (fração de casos realmente urgentes que o modelo não rebaixa — ver
"Gate de qualidade" no [README](../README.md#gate-de-qualidade)) só é calculada **em tempo de
retreino**, contra o conjunto de teste estático. Não existe uma métrica Prometheus contínua
para ela, e por um motivo estrutural, não só falta de instrumentação: medir o recall de
verdade exige o rótulo real do laudo, que só existiria depois de um humano ler e confirmar a
condição — e este projeto não tem esse loop de feedback (rótulo de produção não retorna pro
sistema).

O que existe como **proxy indireto**, sem substituir a métrica de verdade: se
`triagem_predictions_total` mostrar as condições `cardiovascular`/`nervous system` (prioridade
alta) caindo de proporção ao mesmo tempo em que a confiança cai, é sinal de que o recall de
prioridade alta pode estar degradando entre um retreino e outro — mas ninguém dispara alerta
automaticamente nisso hoje.

## 3. Ferramentas implementadas

| Ferramenta | Localização | Função |
|---|---|---|
| `prometheus-client` | `src/triagem/api/metrics.py` | instrumenta as 6 métricas expostas em `/metrics` |
| Prometheus | `docker/prometheus/prometheus.yml` | raspa `/metrics` a cada 5 s |
| Grafana | `docker/grafana/{dashboards,provisioning}/` | 7 painéis provisionados como código, sem clique manual |
| `metrics/training_history.jsonl` | `promote()`/`_append_history()` em `src/triagem/pipeline/steps.py` | histórico append-only de cada execução de treino — promovida ou não, com o motivo da recusa |

## 4. Playbook de resposta

### 4.1 Taxa de erro `5xx` sustentada

Bug real no serviço, não laudo malformado (isso seria `4xx`). Checar `docker compose logs api`
(ou logs do container em produção) pela stack trace; se for exceção não tratada no
`predict()`, é regressão de código, não de modelo — reverter o deploy é mais rápido que
debugar em produção.

### 4.2 Latência p99 acima de 50 ms sustentada

Descartar primeiro se é o modelo ou o servidor: `triagem_inference_duration_seconds` isola a
inferência pura. Se ela continua ~1 ms mas o HTTP subiu, o problema é rede/serialização/CPU do
container, não o classificador — checar recursos do container (`docker stats`) antes de mexer
no modelo.

### 4.3 Confiança mediana caindo (p50 < 0,40)

Sinal de drift: o texto que está chegando parece cada vez menos com o corpus de treino. Não é
motivo para reverter o deploy sozinho — é motivo para puxar uma amostra recente de
`/predict` (via log) e ler manualmente: o laudo real se parece com abstract científico em
inglês (o que o modelo aprendeu), ou é outro registro textual? Se for mudança de fonte de
dado, o retreino semanal não resolve sozinho — precisa de corpus novo.

### 4.4 Uma condição concentrando > 50% das predições

Mesma investigação da 4.3, ângulo diferente: puxar uma amostra dos laudos classificados nessa
condição e checar se fazem sentido clinicamente ou se o modelo colapsou numa única saída
(sintoma comum de embaralhar rótulos ou vetorizar com o vocabulário errado num retreino).

### 4.5 Gate de qualidade reprovando retreinos repetidamente

Ver `metrics/training_history.jsonl` — cada linha tem `rejection_reasons`. Se for sempre
"não supera o modelo em produção", o incumbente pode já estar num teto real (não é bug); se
for o piso de `priority_recall_alta`, é o sinal mais sério: o retreino está aprendendo um
modelo tecnicamente melhor (f1-macro) mas pior em segurança — não promover é o comportamento
correto, mas vale investigar a causa (mudança na proporção de classes do corpus, por exemplo).

### 4.6 Target do Prometheus em `down` / `/health` não responde

Liveness básico. Se o container está de pé mas `/health` não responde, é mais provável travar
no carregamento do modelo (`load_classifier` no lifespan) do que no request handler — checar o
log de startup do container.

## 5. Ciclo de retreino

Semanal, pela DAG `triagem_training` (`schedule="@weekly"`) — o corpus é estático, então a
cadência existe para exercitar o caminho de retreino, não para incorporar dado novo. Gate de
4 critérios antes de qualquer promoção; detalhes em "Gate de qualidade" no README.

## 6. Limitações conhecidas

- **Sem loop de feedback:** o rótulo real do laudo (confirmado por um humano depois da
  leitura) não volta pro sistema — por isso a seção 2.3 é a métrica de negócio mais importante
  do projeto e a que menos visibilidade tem em produção.
- **Sem detecção estatística formal de drift** (PSI/KS, como o repo irmão `grupo4` tem): os
  limiares deste documento são heurísticas sobre as métricas já expostas, não um teste
  estatístico automatizado com alerta próprio.
- **Sem canal de escalação real** (Slack/PagerDuty/on-call): este é um ambiente de
  demonstração local via Docker Compose, não um deploy de produção monitorado 24/7.