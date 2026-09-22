# Report di analisi e split del dataset 0.1.3

> Historical artifact labels use normalized model versions, not filesystem paths.
> Exact commands, identifiers and paths remain in the original document:
> `git cat-file blob 3748e3d0df97998ddde907918288f4f9bc3d9bd5` from the
> [source snapshot](https://github.com/e-tufarini-terrasystem/phenocam-vision-edge/tree/cc63857c12c5553c2e3451854863edf7c0705e2a).

> Historical record for this iteration; use the [current guide](../development.md#current-workflow).
> Commands refer to the original workflow; ignored artifacts require local evidence.

Data della misura: 2026-09-02. Stato: split materializzato e verificato.

## Confini dell'evidenza

- **Misurato:** ricavato da immagini, label e manifest dell'artifact originario.
- **Documentato:** indicazione attribuita a una fonte esterna.
- **Raccomandato:** decisione metodologica specifica per questo dataset.
- **Non verificabile:** informazione non presente nei metadati; non viene inferita.

## Dataset originale — misurato

| Voce | Valore |
| --- | ---: |
| Immagini | 2.240 |
| Immagini positive | 1.475 |
| Negative intenzionali senza label | 765 |
| Annotazioni | 10.562 |
| Immagini multi-oggetto | 1.032 |
| Box medi per immagine | 4,716 |
| Box medi per positiva | 7,161 |

Sorgenti: 1.279 Open Images V7, 721 PhenoCam Network v3 e 240 immagini
operative private. La struttura originaria aveva `train` (2.000),
`operational_dev` (120) e `operational_mining` (120), ma non una validation o
un test finale YOLO.

| Classe | Immagini | % immagini | Annotazioni | % annotazioni |
| --- | ---: | ---: | ---: | ---: |
| `person` | 808 | 36,07% | 3.231 | 30,59% |
| `bicycle` | 125 | 5,58% | 200 | 1,89% |
| `car` | 665 | 29,69% | 6.330 | 59,93% |
| `motorcycle` | 144 | 6,43% | 241 | 2,28% |
| `bus` | 126 | 5,63% | 180 | 1,70% |
| `truck` | 317 | 14,15% | 380 | 3,60% |

Tutti i 2.240 path immagine e i 1.475 path label sono presenti e listati. Le
label non sono vuote, hanno cinque valori, classi configurate e coordinate
normalizzate valide. Non risultano SHA-256 duplicati per file sorgente, pixel
decodificati o file compilato. Le 765 assenze di label corrispondono esattamente
alle righe `polarity=negative` e non sono errori.

## PhenoCam — misurato

- 721 immagini da 157 camere/siti; massimo 9 immagini per camera;
- periodo 2001-09-01–2023-12-30;
- inverno 197, primavera 186, estate 161, autunno 177;
- 564 intervalli consecutivi entro camera: 0 entro 5 minuti, 3 entro 30, 12
  entro 60 e 165 entro un giorno;
- luminanza media thumbnail: 70 sotto 40/255, 42 tra 40–79, 606 tra 80–179 e
  3 almeno 180;
- 70 coppie pHash identiche, 360 a Hamming ≤4 e 479 a Hamming ≤6;
  rispettivamente 55, 291 e 352 sono cross-camera.

Il pHash segnala forte ridondanza percettiva, soprattutto per fotogrammi scuri
o uniformi. Non prova da solo che due immagini siano copie: viene usato per
unire gruppi, mai per cancellare. Gli sfondi ricorrenti sono controllati
trattenendo tutta la camera nello stesso split. Meteo, neve, foschia e altre
condizioni ambientali non hanno etichette versionate e sono **non verificabili**.

Le immagini interne non sono riclassificate come PhenoCam Network: il manifest
le dichiara `internal`. Sono 120 per `raspberrypi2.local` (2025) e 120 per
`sitets02` (2026), con campionamento spesso orario o ogni tre ore e background
fisso fortemente ricorrente.

## Ricerca documentale

- Ultralytics ammette directory o liste per train/val/test, label YOLO
  normalizzate e negative senza `.txt` ([dataset detection](https://docs.ultralytics.com/datasets/detect/)).
- Ultralytics usa `val` durante lo sviluppo e permette la valutazione esplicita
  del test dichiarato nel YAML ([model testing](https://docs.ultralytics.com/guides/model-testing)).
- Split prima di augmentation evita che varianti correlate attraversino il
  confine ([Ultralytics preprocessing](https://docs.ultralytics.com/guides/preprocessing-annotated-data)).
- `StratifiedGroupKFold` tenta la stratificazione senza dividere gruppi; la
  preservazione dei gruppi è il vincolo principale ([scikit-learn](https://scikit-learn.org/stable/modules/cross_validation.html#stratifiedgroupkfold)).
- Con dipendenze temporali/spaziali/gerarchiche va usata block cross-validation
  ([Roberts et al., Ecography 2017](https://www.wsl.ch/lud/biodiversity_events/papers/Roberts_et_al-2017-Ecography.pdf)).
- Lo split deve riflettere le dipendenze dei dati per evitare leakage
  ([Kapoor e Narayanan, Patterns 2023](https://doi.org/10.1016/j.patter.2023.100804)).
- WILDS separa prestazioni ID e OOD e osserva cali sotto distribution shift
  ([Koh et al., PMLR 2021](https://proceedings.mlr.press/v139/koh21a.html)).
- Nelle camere fisse lo sfondo cambia poco; una location mai vista misura la
  generalizzazione reale ([Beery et al., ECCV 2018](https://www.ecva.net/papers/eccv_2018/papers_ECCV/html/Beery_Recognition_in_Terra_ECCV_2018_paper.php)).

Le fonti non impongono le percentuali sotto: esse sono una raccomandazione
derivata dalla composizione misurata.

## Strategie valutate

| Strategia | Vantaggi | Svantaggi/leakage | Esito 0.1.3 |
| --- | --- | --- | --- |
| A — 70/15/15 random stratificato | più evaluation | stessa camera in tutti i subset | respinta |
| B — 80/10/10 random | più training | le percentuali non risolvono leakage | respinta senza gruppi |
| C — sequenza/camera-day | separa frame vicini | stesso background tra split | insufficiente sola |
| D — camera | separa background | classi meno bilanciabili | adottata PhenoCam |
| E — sito | misura siti nuovi | sito=camera nei metadati correnti | adottata/equivalente |
| F — tempo | simula il futuro | Open Images non ha date; sito e anno confusi | non globale |
| G — ibrida | leakage basso, classi controllate | più vincoli | adottata |

## Strategia finale applicata

Il pubblico originario è diviso 80/10/10: Train 1.600, Validation 200 e
TEST-ID 200. Tutte le 240 immagini interne formano TEST-OOD. Sul dataset totale
le quote sono 71,43%, 8,93%, 8,93% e 10,71%; il test combinato è 19,64%.

Unità indivisibili:

1. Open Images: `group_id` di deduplicazione revisionato.
2. PhenoCam: camera/sito intero più componente pHash Hamming ≤6.
3. Interno: camera/sito intero, assegnato a TEST-OOD.

L'ottimizzazione intera con seed 42 mantiene conteggi esatti e minimizza lo
scostamento di classi, stagioni e luminanza **dopo** i vincoli di gruppo. Le
classi rare non giustificano mai la rottura di un gruppo.

| Split | Immagini | Annotazioni | Positive | Negative | Siti | Camere |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Train | 1.600 | 3.750 | 995 | 605 | 117 | 117 |
| Validation | 200 | 464 | 125 | 75 | 23 | 23 |
| TEST-ID | 200 | 463 | 124 | 76 | 17 | 17 |
| TEST-OOD | 240 | 5.885 | 231 | 9 | 2 | 2 |

| Classe | Train | Validation | TEST-ID | TEST-OOD |
| --- | ---: | ---: | ---: | ---: |
| `person` | 2.442 | 306 | 303 | 180 |
| `bicycle` | 160 | 20 | 20 | 0 |
| `car` | 624 | 74 | 74 | 5.558 |
| `motorcycle` | 163 | 20 | 21 | 37 |
| `bus` | 144 | 18 | 18 | 0 |
| `truck` | 217 | 26 | 27 | 110 |

TEST-OOD è volutamente car-heavy e non contiene `bicycle`; descrive i due siti
operativi disponibili, non tutta PhenoCam Network. TEST-ID mantiene le sorgenti
pubbliche del training ma usa camere PhenoCam disgiunte.

## Ruoli e congelamento

- Train: pesi, augmentation e future aggiunte.
- Validation: tuning, early stopping, confronto esperimenti e model selection.
- TEST-ID / TEST-OOD: sola valutazione finale; non guidano tuning, mining,
  selezione immagini, soglie o checkpoint.

## Verifica e modifiche

Il verificatore conferma 2.240 immagini e 10.562 annotazioni, 1.417 gruppi
finali, 157 camere PhenoCam, zero hash duplicati e zero coppie pHash≤6
cross-split. Nessuna immagine o gruppo attraversa split. Tutte le classi sono
presenti in Train, Validation e TEST-ID. Nessun elemento è escluso.

Il progetto ora contiene `builder/partition/`, manifest per split, tre YAML
(test combinato, ID e OOD), statistiche, checksum e report. Il viewer espone
Train, Validation, Test, Test ID e Test OOD, aggiorna conteggi e metadati.

Comandi dalla directory `dataset/`:

Per i comandi storici, consultare il documento originale indicato sopra.

Limite di verifica UI: in questa sessione non era disponibile un browser
controllabile. La sintassi JavaScript e i controlli statici sono testati; il
click-through manuale resta una verifica locale esplicita.
