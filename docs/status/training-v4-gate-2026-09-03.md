# Ciclo multi-sito PhenoCam — gate di revisione

> Historical record for this iteration; use the [current guide](../development.md#current-workflow).
> Commands refer to the original workflow; ignored artifacts require local evidence.

Data di avvio: 2026-09-03.

Stato: investigazione e preparazione del gate. Nessun dato di TEST-ID,
`sitets02` o `raspberrypi2.local` può guidare screening, selezione, soglia o
checkpoint. Le metriche TEST-OOD storiche restano solo storia e non sono
evidenza decisionale per questo ciclo.

## Invarianti

- Solo siti assegnati integralmente a `train` possono entrare nella coda.
- Identità, hash, camera-day e near-duplicate non attraversano gli split.
- Le 240 immagini interne restano fuori da training e tuning.
- Le predizioni full-image o tiled sono proposte, mai ground truth.
- Il training non parte senza un export CVAT completato e attribuito.
- La selezione finale usa soltanto validation; i test vengono aperti una volta
  dopo il congelamento.

## Struttura locale dichiarata prima dell'implementazione

```text
dataset/config/
  training-v4.json                         (~55 linee dati)

dataset/builder/mining/
  __main__.py                              (~19 linee aggiunte; totale <200)
  cvat.py                                  (~2 linee aggiunte; responsabilità invariata)
  training_pool.py                         (~115 linee produttive, test esclusi)
  multisite_selection.py                   (~175 linee produttive, test esclusi)
  preview.py                               (~75 linee produttive, test esclusi)
  teacher.py                               (~12 linee modificate; responsabilità invariata)
  teacher_bundle.py                        (~85 linee produttive, test esclusi)

scripts/runtime_evaluation/
  __init__.py                              (~0 linee produttive)
  prediction.py                            (~145 linee produttive, test esclusi)
  metrics.py                               (~190 linee produttive, test esclusi)
  evaluate.py                              (~190 linee produttive, test esclusi)

phenocam/inference/
  detections.py                            (~4 linee modificate; responsabilità invariata)

dataset/tests/test_mining.py               (solo test; escluso dalle stime)
tests/test_runtime_evaluation.py            (solo test; escluso dalle stime)
```

Responsabilità:

- `training_pool.py`: materializza e attesta il solo pool PhenoCam appartenente
  a siti train, applicando esclusioni di identità, hash e camera-day prima
  dell'inferenza.
- `multisite_selection.py`: costruisce una coda teacher bilanciata per sito e
  camera-day, con vincolo SSCD e priorità agli oggetti piccoli senza lasciare
  dominare scene estreme.
- `preview.py`: produce una pagina locale, senza rete, che sovrappone le
  proposte alle immagini della coda.
- `teacher_bundle.py`: promuove l'output tiled in un bundle autosufficiente e
  verificato che l'import post-revisione può usare come receipt iniziale.
- `prediction.py`: esegue sul checkpoint le stesse sedici viste e la stessa
  geometria inversa/fusione del runtime, conservando l'origine delle detection.
- `metrics.py`: assegna detection e target e calcola contabilità per classe,
  dimensione e provenienza della vista.
- `evaluate.py`: possiede I/O, misure di tempo/memoria e aggregazioni per
  sorgente/sito, mantenendo separata la validation Ultralytics standard.

La complessità aggiunta è necessaria perché il valutatore storico misura solo
la vista completa a 640 e perché il selettore storico può leggere candidati di
siti riservati prima del controllo post-hoc. I nuovi confini rendono entrambe
le proprietà verificabili prima dell'operazione costosa.

## Audit del pool train-only

Il manifest canonico assegna 117 siti a Train, 23 a Validation e 17 a TEST-ID.
Il filtro è stato eseguito prima del teacher e non ha letto immagini, label o
embedding sigillati. Da 5.675 candidati iniziali sono rimasti 2.152 frame,
distribuiti su 102 siti e 1.502 camera-day. Sono stati esclusi 1.439 frame di
siti held-out, 765 identità già usate, 1.282 camera-day già usati, un hash già
usato e 36 siti non presenti nella partizione canonica.

Il teacher YOLO26x a 1.280, confidence floor 0,10, ha completato 2.152/2.152
frame senza errori. Hash indice:
`b546191b81d76144f7a86c1a16b89b27b39b0ce226d922fe251fec437f107737`.

## Distribuzione del training corrente

Il training effettivo ricostruito dalla partizione canonica più le 18 aggiunte
revisionate contiene 1.618 immagini: 1.023 Open Images e 595 PhenoCam da 117
siti. Le positive sono 1.013 e le negative 605.

| Sorgente | Immagini | Positive | Piccoli | Medi | Grandi |
| --- | ---: | ---: | ---: | ---: | ---: |
| Open Images | 1.023 | 983 | 1.561 | 845 | 1.288 |
| PhenoCam | 595 | 30 | 150 | 8 | 2 |
| **Totale** | **1.618** | **1.013** | **1.711** | **853** | **1.290** |

| Classe | Istanze |
| --- | ---: |
| `person` | 2.443 |
| `bicycle` | 160 |
| `car` | 714 |
| `motorcycle` | 163 |
| `bus` | 145 |
| `truck` | 229 |

Le soglie dimensionali sono applicate dopo il letterbox equivalente a 640:
piccolo `<32² px`, medio `<96² px`, grande altrimenti. Nei soli frame PhenoCam
le 160 annotazioni sono `person` 8, `car` 134, `bus` 1 e `truck` 17; 150 sono
piccole. La diversità nominale di 595 immagini non sostituisce il dato più
importante: 117 siti ma soltanto 30 frame positivi.

## Coda multi-sito pronta per revisione

La selezione finale contiene 20 immagini da 13 siti e 20 camera-day, con
massimo tre frame per sito e uno per camera-day. SSCD `<0,95` è stato applicato
contro 1.665 embedding appartenenti esclusivamente al training o a precedenti
decisioni di review; gli embedding Validation/TEST-ID letti sono zero.

| Sito | Immagini |
| --- | ---: |
| `arsgreatbasinltar177` | 2 |
| `bartlettir` | 1 |
| `bigtraillake` | 2 |
| `bitterootvalley` | 1 |
| `borgocioffinorth` | 1 |
| `butte` | 2 |
| `dollysods` | 1 |
| `forbes` | 1 |
| `laurentides` | 1 |
| `nphtin` | 2 |
| `riverton` | 1 |
| `sevmveblack28ambinc` | 2 |
| `siwetland` | 3 |

Le proposte finali full-image + 15 crop sono 37: `person` 28, `car` 3,
`motorcycle` 1 e `truck` 5; per dimensione, 27 piccole, 9 medie e 1 grande.
Dieci box sono state aggiunte dalle crop. Sono soltanto predizioni e questi
conteggi non devono essere descritti come distribuzione del ground truth.

Artifact principale:

- immagini e receipt: `dataset/workspace/training-v4/review/gate/`;
- anteprima locale: `dataset/workspace/training-v4/review/gate/preview.html`;
- preannotazioni COCO:
  `dataset/workspace/training-v4/review/gate/annotations.coco.zip`;
- SHA-256 annotazioni:
  `f24291eb4fa5304bd6c4d2e08bfce9576ec08bf6a11a56932e280a6c16b7c3c9`;
- SHA-256 receipt bundle:
  `86f992c0f1165079d5470c06d5d7b309fcb74a290c80f317d89fbfdc74d43704`.

## Baseline validation aderente al runtime

Il valutatore storico Ultralytics usa `model.val(..., imgsz=640)` e quindi una
sola vista. Il nuovo artifact mantiene quella misura separata e misura anche
full-image, crop-only e pipeline fusa con geometria e soppressione del runtime.
La metrica operativa esclude `bicycle`, coerentemente con il filtro privacy;
la mAP standard storica include le sei classi annotate. Le soglie sotto sono i
massimi F1 su una griglia validation diagnostica `0,15..0,40` a passo `0,05`;
non congelano ancora la selezione del ciclo v4.

| Checkpoint | mAP50-95 standard | Soglia pipeline | Full F1 | Crop F1 | Pipeline F1 | Pipeline P/R |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| YOLO26n originale | 0,5318 | 0,40 | 0,5081 | 0,2875 | 0,4243 | 0,3804 / 0,4797 |
| v3-extended corrente | 0,5404 | 0,35 | 0,4724 | 0,2504 | 0,3633 | 0,3234 / 0,4144 |

La mAP a vista singola migliora nel modello corrente, ma la pipeline reale
regredisce di 0,0610 F1 rispetto al checkpoint originale. Questa è evidenza di
validation compatibile con forgetting e impedisce di considerare sufficiente
la sola mAP Ultralytics. Sul modello corrente, le crop recuperano 63 target
persi dalla full view, dei quali 41 piccoli, ma introducono 385 FP complessivi
al punto operativo selezionato. Full e 15 crop richiedono in media 23,1 ms e
317,1 ms di inferenza MPS per immagine; picco RSS del processo 760 MB. Sono
misure su Mac M4, non sul Raspberry Pi 3.

## Istruzioni per la revisione umana

1. Aprire `dataset/workspace/training-v4/review/gate/preview.html` e verificare
   l'allineamento iniziale.
2. Il task CVAT locale `14`, `V4 public PhenoCam multi-site positives - review
   gate`, è stato creato con le 20 immagini sotto `gate/images/default/`, le
   label di `gate/labels.json` e `gate/annotations.coco.zip` come COCO 1.0. Il
   read-back conferma 20 immagini, 37 proposte e le categorie attese; non creare
   un secondo task.
3. Controllare ogni frame: rimuovere falsi positivi, stringere le box, aggiungere
   ogni persona o veicolo reale visibile, marcare `ambiguous` quando necessario
   e seguire `dataset/docs/annotation-guide-it.md`. I siti
   `arsgreatbasinltar177` e `nphtin` richiedono attenzione speciale perché
   revisioni precedenti hanno trovato confusori statici in quelle camere.
4. Completare il task ed esportare COCO 1.0 esattamente come
   `dataset/workspace/training-v4/review/human-export.coco.zip`.

Comando esatto per riprendere dopo l'export (richiede soltanto l'ID numerico
mostrato da CVAT):

```sh
printf 'CVAT task id: ' && read -r TASK_ID && \
dataset/.venv/bin/python -m dataset.builder.mining \
  --config dataset/config/training-v4.json import-reviewed \
  --selection dataset/workspace/training-v4/selection/multisite.csv \
  --bundle-dir dataset/workspace/training-v4/review/gate \
  --export dataset/workspace/training-v4/review/human-export.coco.zip \
  --output-dir dataset/workspace/training-v4/reviewed/public-multisite \
  --task-id "$TASK_ID" --annotator Emanuele --reviewer Emanuele \
  --task-completed
```

## Gate e lavoro non eseguito

La revisione umana manca. Di conseguenza non sono stati creati un nuovo
dataset, candidati fine-tuned, interpolazioni o run multi-seed; nessun
checkpoint è stato selezionato o esportato e TEST-ID, `sitets02` e
`raspberrypi2.local` non sono stati aperti. Le tabelle candidate, gli stress
test e la misura su Raspberry Pi 3 saranno prodotti solo dopo il superamento
del gate e il congelamento validation-only.
