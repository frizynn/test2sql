# Nivii

Consultas en lenguaje natural sobre datos de ventas de un restaurante (24,212 filas). Dos modelos locales corren en CPU, offline, en Docker: uno traduce la pregunta a SQL, el otro explica los resultados en español.

## Instalacion

```bash
git clone https://github.com/frizynn/text2sql.git
cd text2sql

mkdir -p models
curl -L -o models/qwen2.5-coder-1.5b-instruct-q4_k_m.gguf \
  https://huggingface.co/frizynn/qwen2.5-coder-1.5b-instruct-q4_k_m-gguf/resolve/main/qwen2.5-coder-1.5b-instruct-q4_k_m.gguf
curl -L -o models/Qwen3.5-2B-Q4_K_M.gguf \
  https://huggingface.co/unsloth/Qwen3.5-2B-GGUF/resolve/main/Qwen3.5-2B-Q4_K_M.gguf
```

## Uso con Docker

```bash
cp .env.example .env
docker compose up --build
# Abrir http://localhost:8000
```

## Uso con CLI (sin Docker)

```bash
pip install rich
nivii
```

La CLI chequea el entorno, instala llama-server si falta, descarga modelos, levanta los servidores y te deja elegir entre Web UI o TUI.

```bash
nivii ask "¿Cuales son los 5 productos mas vendidos?"
```

## Arquitectura

| Servicio | Modelo | Rol |
|----------|--------|-----|
| `text2sql` | qwen2.5-coder-1.5b | Pregunta → SQL |
| `nlg` | Qwen3.5-2B | Resultados → respuesta en español |
| `api` | FastAPI + SQLite | Orquesta pipeline, sirve la UI |

Pipeline MCTS: genera SQL, verifica semanticamente, y si falla refina con busqueda de arbol (critique → refine → evaluate, hasta 5 rollouts).

## Estructura

```
api/             Backend FastAPI + pipeline MCTS
cli/             CLI con setup wizard
static/          Web UI
prompts/         Prompts y few-shot examples
grammars/        Gramatica GBNF para SQL
docker/          Dockerfiles
data.csv         Dataset de ventas (24,212 filas)
```
