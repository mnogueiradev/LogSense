# CyberIA — Análise de Logs de Firewall com IA

Projeto de TCC reorganizado para manter um pipeline único, eliminar conflito de modelos e garantir consistência entre treino, validação e API.

## Objetivo

Detectar anomalias e classificar possíveis ataques a partir de dados de rede/logs de firewall, utilizando:

- Isolation Forest
- Autoencoder
- Ensemble entre os dois modelos
- API FastAPI para consumo externo

## Estrutura final

```text
cyberia/
├── cleanup_legacy.py
├── README.md
├── requirements.txt
├── data/
│   ├── raw/
│   ├── processed/
│   └── models/
└── src/
    ├── __init__.py
    ├── api/
    │   ├── __init__.py
    │   └── main.py
    ├── core/
    │   ├── __init__.py
    │   ├── artifacts.py
    │   └── paths.py
    ├── features/
    │   ├── __init__.py
    │   ├── constants.py
    │   └── dataset_loader.py
    ├── models/
    │   ├── __init__.py
    │   ├── autoencoder.py
    │   ├── classifier.py
    │   ├── detection.py
    │   └── ensemble.py
    └── pipelines/
        ├── __init__.py
        ├── train.py
        └── validate.py
```

## O que foi corrigido

- Um único pipeline oficial de treino: `src/pipelines/train.py`
- Um único pipeline oficial de validação: `src/pipelines/validate.py`
- Uma única API oficial: `src/api/main.py`
- Um único local para artefatos: `data/models/`
- Mesmas 5 features em treino, validação e inferência
- Re-treinamento sobrescreve artefatos antigos antes de salvar novos
- Upload CSV via FastAPI funcionando com `python-multipart`

## Features oficiais

A ordem abaixo é fixa e obrigatória:

1. `connections`
2. `bytes`
3. `packets`
4. `packet_size`
5. `ports`

Se o arquivo de entrada for um CSV do CICIDS2017 com nomes brutos, o loader converte automaticamente:

- `Flow Duration` → `connections`
- `Total Length of Fwd Packets` → `bytes`
- `Total Fwd Packets` → `packets`
- `Average Packet Size` → `packet_size`
- `Destination Port` → `ports`

## Instalação

```bash
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Limpeza dos arquivos legados

Modo de teste:

```bash
python cleanup_legacy.py --dry-run
```

Executar de verdade:

```bash
python cleanup_legacy.py
```

## Treino

Com features prontas:

```bash
python -m src.pipelines.train --input data/processed/features.csv
```

Com CICIDS bruto:

```bash
python -m src.pipelines.train --input data/raw/cicids.csv
```

Artefatos gerados:

- `data/models/scaler.pkl`
- `data/models/isolation.pkl`
- `data/models/autoencoder/model.keras`
- `data/models/autoencoder/threshold.pkl`
- `data/models/manifest.json`

## Validação

```bash
python -m src.pipelines.validate --input data/processed/features.csv
```

Saídas:

- `data/processed/validation_predictions.csv`
- `data/models/validation_report.json`

## API

```bash
uvicorn src.api.main:app --reload
```

### Endpoints

- `GET /health`
- `GET /metadata`
- `POST /predict/json`
- `POST /predict/csv`
- `POST /reload-models`

## Exemplo JSON

```json
{
  "records": [
    {
      "connections": 1024,
      "bytes": 4096,
      "packets": 32,
      "packet_size": 128,
      "ports": 443
    }
  ]
}
```

## Regras do projeto

- Nunca mudar a quantidade ou a ordem das features sem re-treinar.
- Nunca salvar modelos fora de `data/models/`.
- Nunca usar scripts paralelos de treino/inferência fora dos pipelines oficiais.
- Sempre validar com os mesmos artefatos gerados no treino.

## Classificação de ataque

O arquivo `src/models/classifier.py` usa heurísticas explicáveis para:

- `port_scan_suspeito`
- `brute_force_suspeito`
- `ddos_suspeito`

Isso é adequado para apresentação do TCC e pode ser refinado depois com logs reais do pfSense.
