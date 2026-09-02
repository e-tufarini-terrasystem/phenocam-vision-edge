# Phenocam Vision dataset v3

`dataset-v3/` è l'artifact YOLO leakage-aware pronto per training e valutazione.
Contiene tutte le 2.240 immagini dell'artifact v3 originario, senza esclusioni.

## Split

| Split | Immagini | Positive | Negative | Annotazioni | Ruolo |
| --- | ---: | ---: | ---: | ---: | --- |
| Train | 1.600 | 995 | 605 | 3.750 | pesi e augmentation |
| Validation | 200 | 125 | 75 | 464 | tuning, early stopping, scelta modello |
| TEST-ID | 200 | 124 | 76 | 463 | valutazione finale pubblica |
| TEST-OOD | 240 | 231 | 9 | 5.886 | valutazione finale operativa |
| **Totale** | **2.240** | **1.475** | **765** | **10.563** | |

Il pubblico originario (1.279 Open Images e 721 PhenoCam) è diviso 80/10/10.
Le 240 immagini delle due camere operative private sono interamente TEST-OOD.
Sul totale, Train è 71,43%, Validation 8,93%, TEST-ID 8,93% e TEST-OOD 10,71%.

| Classe | Train | Validation | TEST-ID | TEST-OOD |
| --- | ---: | ---: | ---: | ---: |
| `person` | 2.442 | 306 | 303 | 180 |
| `bicycle` | 160 | 20 | 20 | 0 |
| `car` | 624 | 74 | 74 | 5.558 |
| `motorcycle` | 163 | 20 | 21 | 37 |
| `bus` | 144 | 18 | 18 | 1 |
| `truck` | 217 | 26 | 27 | 110 |

L'assenza di `bicycle` e la rarità di `bus` in TEST-OOD sono proprietà misurate
del dominio operativo. Le camere non vengono spostate per correggerle, perché la
prevenzione del leakage ha priorità sulla stratificazione.

## Regola di assegnazione

- Open Images: il `group_id` di deduplicazione revisionato è indivisibile.
- PhenoCam: l'intera camera/sito è indivisibile; vengono inoltre uniti i
  componenti con distanza pHash Hamming ≤ 6. Il pHash raggruppa ma non elimina.
- Dati interni: l'intera camera/sito resta in TEST-OOD.
- Seed: `42`.
- L'ottimizzazione intera preserva i conteggi immagini esatti e minimizza lo
  scostamento di classi, stagioni e luminosità dopo i vincoli di gruppo.

Le 721 immagini PhenoCam coprono 157 camere/siti e il periodo 2001–2023. La
divisione finale non condivide camere né coppie pHash≤6 tra split. I due siti
interni sono successivi (2025–2026), privati e mai usati da Train/Validation.

## Struttura

```text
dataset-v3/
├── images/{train,val,test/{id,ood}}/
├── labels/{train,val,test/{id,ood}}/
├── manifests/{train,val,test,test-id,test-ood}.csv
├── metadata/
├── reports/{dataset-analysis.md,verification.json}
├── dataset.yaml
├── dataset-test-id.yaml
└── dataset-test-ood.yaml
```

`dataset.yaml` dichiara il test combinato; i due YAML aggiuntivi permettono di
misurare ID e OOD separatamente. Le negative omettono intenzionalmente il file
label, come previsto dal formato detection di Ultralytics.

Ogni manifest conserva path, split, sorgente, sito, camera, timestamp,
`partition_group_id` e i conteggi ricostruibili dalle label. Il manifest
canonico completo è `metadata/source-images.csv`.

## Riproducibilità

Dalla directory `dataset/`:

```sh
.venv/bin/python -m builder.partition build dataset-v3 dataset-v3-rebuilt
.venv/bin/python -m builder.partition verify dataset-v3
```

Il build usa hard link sullo stesso filesystem e non duplica i byte delle grandi
immagini. Rifiuta una destinazione già esistente e verifica conteggi, path,
label, classi, checksum, duplicati, gruppi, camere e pHash prima della promozione.

## Viewer

Aprire `dataset/viewer.html`, scegliere `dataset-v3/` e usare **Dataset split**.
Sono disponibili Train, Validation, Test combinato, Test ID e Test OOD. Il
cambio split aggiorna immagini, annotazioni, navigazione, filtri e metadati
visibili, senza conservare immagini del subset precedente.

## Protocollo

Train può ricevere in futuro nuovi dati e augmentation. Validation può guidare
gli esperimenti. TEST-ID e TEST-OOD sono congelati: non devono guidare tuning,
model selection, mining o selezione di nuove immagini. Le immagini `internal`
sono private e non devono essere redistribuite.

Metodologia, misure di ridondanza, confronto delle alternative e fonti sono in
`reports/dataset-analysis.md` e nel report versionato
`docs/status/dataset-v3-split-2026-09-02.md`.
