# Classificatie of ECGs

**Stan Schepers | Juni 2019**

## Install

```sh
virtualenv --python=python3 env
source env/bin/activate
pip install -r requirements.py
python main.py # voor model te trainen en performance.
python feature_engineering.py # voor features te generen uit de raw data.
python parameter_tuning.py # voor de parameters te tunen zodat de performance vergroot.
```

## Bestanden

Een oplijsting van alle bestanden voor dit project.

| Bestandsnaam             | Beschrijving                                                 |
| ------------------------ | ------------------------------------------------------------ |
| `paper.pdf`              | De paper voor het project.                                   |
| `main.py`                | Dit bestand bevat de code om het model traint en de roc curve gemaakt. |
| `feature_engineering.py` | Dit bestand bevat de code die een csv-bestand aanmaakt met de berekende features van de raw data. |
| `parameter_tuning.py`    | Dit bestand bevat de code om de parameters te optimaliserne. |
| `ecg.pdf`                | Het verslag over ECG's.                                      |
| `Projectvoorstel.pdf`    | Het projectvoorstel.                                         |
| `metrics.pdf`            | Het verslag over de metrieken om de performance van het model te beoordelen. |
| `output.py`              | Dit bestand bevat code om het  genereren van een JSON bestand die de performance van een model bespreekt. (in progress) |
| `requirements.txt`       | Lijst van alle Python Libaries gebruikt in dit project       |
| `webapp` + `api.py`      | Een webapplicatie de de JSON bestanden uit `output.py` gaat weergeven en hierrond grafieken genereert. (in progress) |
| `output.py`              | Klasse gebruikt om de metrieken en de attributen te schrijven naar een JSON bestand. |
| `ecg.csv`                | Een csv-bestand met alle features van de data.               |

