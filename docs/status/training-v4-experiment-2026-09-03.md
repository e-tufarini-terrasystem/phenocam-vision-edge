# YOLO26n training v4 — registro sperimentale

> Historical record for this iteration; use the [current guide](../development.md#current-workflow).
> Commands refer to the original workflow; ignored artifacts require local evidence.

Stato iniziale: revisione training completata; preparazione della validation
PhenoCam positiva prima del training. TEST-ID e TEST-OOD restano sigillati.

## Invarianti

- Le aggiunte training appartengono soltanto a siti canonici `train`.
- Le aggiunte validation appartengono soltanto a siti canonici `val` e non
  entrano mai nel training.
- I candidati teacher sono proposte: soltanto un export CVAT completato produce
  ground truth.
- TEST-ID, `sitets02` e `raspberrypi2.local` non vengono letti fino al freeze
  validation-only di checkpoint, soglia e configurazione.
- Nessun dataset, modello o risultato esistente viene sovrascritto.

## Struttura dichiarata

La struttura prevista usa il minimo dominio nuovo necessario; tutte le stime
escludono test e fixture.

```text
dataset/builder/mining/
  training_pool.py              (~125 linee; filtro train/val site-disjoint)
  validation_selection.py       (~120 linee; coda positiva soltanto validation)
  __main__.py                   (~14 linee modificate; CLI invariata nel ruolo)

scripts/training_v4/
  __init__.py                   (~1 linea)
  dataset.py                    (~195 linee; materializzazione e audit v4)
  train.py                      (~145 linee; un esperimento isolato)
  standard.py                   (~145 linee; sola valutazione Ultralytics)
  interpolate.py                (~155 linee; griglia lineare dichiarata)
  selection.py                  (~190 linee; ranking e freeze validation-only)
  finalize.py                   (~160 linee; export ONNX e verifica contratto)
  experiments.json             (~45 linee; ricerca piccola prescritta)
```

L'evaluatore runtime full-image piu 15 crop resta in
`scripts/runtime_evaluation/`: non viene duplicato. Ogni file di orchestrazione
resta sotto 200 linee produttive.

## Gate training completato

Il task CVAT locale 14 e stato chiuso come `completed`. Export COCO SHA-256
`afbecb7d0f988f49646723e5bffab83038b5fc19baed277c15c3a4be0dc03644`.
L'import verificato contiene 20 immagini: 3 positive, 17 negative, 0 ambigue e
4 box reali (`car`: 2, `truck`: 2). Delle 37 proposte teacher, 35 sono state
rimosse e 2 nuove box sono state aggiunte.

## Validation prima del training

La validation canonica ha 200 immagini e 464 box. I 72 frame PhenoCam
contengono soltanto 2 box `car`, in 2 frame di un unico sito positivo; i 23 siti
PhenoCam validation sono comunque completamente disgiunti dai 117 siti train.
Questa copertura e troppo debole per selezionare con affidabilita un modello per
telecamere fisse, quindi viene preparata una piccola coda umana esclusivamente
su siti `val`, senza aprire i test.

Lo screening YOLO26x a 1280 e confidenza minima 0,10 ha completato 501/501
frame senza failure. Soltanto 5 frame da `marena` e `nwohiocrop` superano la
soglia, con 8 proposte (`car`: 7, `truck`: 1); a 0,15 resterebbe un solo sito.
La coda conserva quindi 0,10 come puro recall gate e richiede revisione umana.

Il filtro pHash e camera-day conserva 3 frame da 2 siti e scarta 2 near
duplicate. La teacher inference sui 15 crop aggiunge 2 proposte alle 5 full
image: il gate finale ha 7 proposte, tutte piccole a 640 (`car`: 4, `truck`: 1,
`person`: 1, `bus`: 1). Il bundle e in
`dataset/workspace/training-v4/validation/review/gate/`, SHA-256 receipt
`4040c6fce5603493c73eed0bf748f77a27f3c7f3be1b64a12bbd5c512816256`.
Il task CVAT locale 15, `V4 public PhenoCam validation positives - review
gate`, e stato creato con le 3 immagini e le 7 proposte.

La revisione del task 15 e stata chiusa in `Acceptance/Completed`. L'export
COCO ha SHA-256
`54282ab6937264d40610de1c871ce772a1a826e14822439344d6bbaf829fd635`:
tutte le 3 immagini sono negative confermate e tutte le 7 proposte sono state
rimosse. Nessuna pseudo-label entra nella validation. Rimane quindi il limite
grave di 2 soli target PhenoCam positivi da un sito, che verra dichiarato nei
risultati e impedisce conclusioni forti sul dominio.

## Dataset e criterio di selezione congelati prima dei risultati

`dataset-v4` contiene 1.621 immagini train con 3.858 box e 203 immagini
validation con 464 box. Il train aggiunge soltanto le 3 immagini positive
revisionate; la validation aggiunge le 3 negative confermate. L'audit ha
verificato 2.968 checksum, zero sovrapposizioni di siti e nessun file TEST nel
dataset. Fingerprint SHA-256:
`ac6d6cc4bd7850cd59bccd3bb28426bdd1f9d59befb6a7e3baed16f7b500e976`.

Per ogni candidato la soglia e scelta soltanto sulla validation, massimizzando
la F1 della pipeline full-image piu 15 crop sulla griglia 0,15--0,45; a parita
si preferisce la soglia maggiore. Un candidato e ammissibile soltanto se la
mAP50-95 standard su OpenImages non scende oltre 0,01 rispetto all'originale e
se il recall PhenoCam non e inferiore all'originale. Tra gli ammissibili vince
la F1 pipeline globale; seguono mAP50-95 globale e, infine, la modifica piu
piccola.

Il miglior fine-tuning genera la famiglia di interpolazione gia dichiarata
`alpha = {0,10; 0,25; 0,50; 0,75}`. La linea finalista viene ripetuta con seed
17 e 73. Il nuovo modello sostituisce l'originale soltanto se la F1 media sui
tre seed supera l'originale di almeno 0,005 e ogni replica conserva entrambi i
vincoli di ammissibilita; altrimenti resta l'originale.

## Distribuzione finale del dataset

| Split / sorgente | Immagini | Positive | Siti | Siti positivi | Box per classe | Dimensioni a 640 |
| --- | ---: | ---: | ---: | ---: | --- | --- |
| train / Open Images | 1.023 | 983 | — | — | person 2.435; car 580; truck 212; motorcycle 163; bicycle 160; bus 144 | small 1.384; medium 909; large 1.401 |
| train / PhenoCam | 598 | 33 | 117 | 10 | car 136; truck 19; person 8; bus 1 | small 150; medium 10; large 4 |
| val / Open Images | 128 | 123 | — | — | person 306; car 72; truck 26; bicycle 20; motorcycle 20; bus 18 | small 160; medium 128; large 174 |
| val / PhenoCam | 75 | 2 | 23 | 1 | car 2 | small 2 |

Le dimensioni seguono COCO dopo il letterbox a 640: small `< 32²`, medium
`< 96²`, large altrimenti. La copertura PhenoCam positiva in validation resta
troppo debole per una conclusione universale.

## Ricerca validation-only

| Candidato | mAP50-95 | mAP Open Images | Soglia | F1 full | F1 crop | F1 pipeline | Recall small | Ammissibile |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| originale | 0,5322 | 0,5335 | 0,45 | 0,4849 | 0,2966 | 0,4320 | 0,3936 | sì |
| head-only 5e-5 | 0,5427 | 0,5438 | 0,35 | 0,4833 | 0,2464 | 0,3683 | 0,3245 | sì |
| head-only 1e-4 | 0,5430 | 0,5443 | 0,35 | 0,4886 | 0,2473 | 0,3547 | 0,3245 | sì |
| partial-freeze 1e-4 | 0,5286 | 0,5292 | 0,35 | 0,4924 | 0,2473 | 0,3870 | 0,3032 | no: recall PhenoCam 0,5 |
| interpolation 10% | 0,5329 | 0,5344 | 0,45 | 0,4795 | 0,2919 | 0,4339 | 0,3883 | sì |
| interpolation 25% | 0,5389 | 0,5402 | 0,45 | 0,4655 | 0,2898 | 0,4226 | 0,3723 | sì |
| interpolation 50% | 0,5484 | 0,5488 | 0,45 | 0,4664 | 0,2807 | 0,4004 | 0,3138 | no: recall PhenoCam 0,5 |
| interpolation 75% | 0,5495 | 0,5493 | 0,40 | 0,4730 | 0,2634 | 0,3820 | 0,3191 | sì |

La mAP cresce mentre la pipeline peggiora: la mAP single-view non è un proxy
sufficiente per il prodotto multi-crop.

Il finalista interpolation 10% produce F1 pipeline `0,4339`, `0,4339` e
`0,4346` per seed 42, 17 e 73 (media `0,4341`, range `0,0007`). La mAP50-95
globale è rispettivamente `0,5329`, `0,5336`, `0,5323`; tutti conservano recall
PhenoCam 1,0 e il vincolo Open Images. Il guadagno F1 medio rispetto
all'originale è però soltanto `+0,0021`, inferiore al minimo `+0,005` congelato
prima dei risultati. Il finalista non passa il gate e viene mantenuto il
checkpoint originale.

## Selezione congelata ed export

- checkpoint: `models/yolo26n.pt`, copiato come `models/yolo26n-v4.pt`;
- SHA-256 PT: `9b09cc8bf347f0fc8a5f7657480587f25db09b34bf33b0652110fb03a8ad4fef`;
- soglia: `0,45`;
- ONNX opset 20: `models/yolo26n-v4.onnx`;
- SHA-256 ONNX: `d2f2aad7f15c578db23b9bbf74923c16f1e7ba42591746c177ecd96cae936db1`;
- contratto verificato: input `1×3×640×640`, output `1×300×6`, 80 classi COCO.

La receipt `output/training-v4/frozen-selection.json` è stata scritta prima di
aprire TEST-ID e TEST-OOD. La selezione non è stata modificata dopo il test.

## Risultato domain-representative

Non viene calcolata una media artificiale fra domini con annotazioni molto
diverse. Il risultato operativo disponibile è `sitets02`; TEST-ID è riportato
separatamente come controllo di generalizzazione.

| Controllo / vista | TP | FP | FN | Precision | Recall | F1 | Accuracy detection |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| TEST-ID full | 204 | 68 | 239 | 0,7500 | 0,4605 | 0,5706 | 0,3992 |
| TEST-ID crop-only | 169 | 513 | 274 | 0,2478 | 0,3815 | 0,3004 | 0,1768 |
| TEST-ID pipeline | 239 | 311 | 204 | 0,4345 | 0,5395 | 0,4814 | 0,3170 |
| sitets02 full | 392 | 12 | 479 | 0,9703 | 0,4501 | 0,6149 | 0,4439 |
| sitets02 crop-only | 625 | 28 | 246 | 0,9571 | 0,7176 | 0,8202 | 0,6952 |
| sitets02 pipeline | 638 | 28 | 233 | 0,9580 | 0,7325 | 0,8302 | 0,7097 |

La mAP50-95 standard TEST-ID è `0,5823` (mAP50 `0,7306`). Le 72 immagini
PhenoCam pubbliche TEST-ID coprono 17 siti ma hanno una sola box positiva: la
pipeline la trova, con 6 FP complessivi su quattro siti. È un controllo utile,
non una stima stabile del dominio. Su TEST-ID la pipeline recupera 62 target
persi dalla full view, 46 piccoli, ma perde 27 target full dopo la fusione.

Su `sitets02`, dei 430 target piccoli la pipeline ne trova 245 (recall 0,5698)
contro 76 della full view (recall 0,1767). Le classi deboli sono `truck`
(recall 0,2162) e, in misura minore, `motorcycle` (0,6429); `car` raggiunge
recall 0,7849.

## Extreme-stress: raspberrypi2.local

Questo è un parcheggio 4.608×2.592 ad altissima densità, con 5.014 target e
4.233 target small a 640. Non fa parte del risultato operativo principale.

| Vista | TP | FP | FN | Precision | Recall | F1 | Accuracy detection |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| full | 385 | 5 | 4.629 | 0,9872 | 0,0768 | 0,1425 | 0,0767 |
| crop-only | 1.671 | 91 | 3.343 | 0,9484 | 0,3333 | 0,4932 | 0,3273 |
| pipeline | 1.706 | 88 | 3.308 | 0,9509 | 0,3402 | 0,5012 | 0,3344 |

La pipeline aggiunge 1.321 veri positivi netti rispetto alla vista intera. Sui
target small passa da 175/4.233 (recall `0,0413`) a 1.357/4.233 (recall
`0,3206`). Rimane quindi un limite severo, ma i crop producono un recupero
sostanziale senza esplosione di FP.

Il run TEST-OOD unico, composto da 120 frame `sitets02` e 120 frame stress,
misura su M4 MPS `24,75 ms` per la full inference e `211,27 ms` per i 15 crop,
con picco RSS `729 MB`; preprocessing, fusione e I/O portano il wall medio a
`326,3 ms`. Il profiler non separa la latenza per sito, quindi questi numeri
non vengono presentati come una misura esclusiva di raspberrypi2.local.

## Effetto dello stress test sull'aggregato OOD

| Vista | F1 senza Raspberry | F1 OOD aggregata | Delta | Recall senza Raspberry | Recall aggregata | Delta |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| full | 0,6149 | 0,2327 | -0,3822 | 0,4501 | 0,1320 | -0,3180 |
| crop-only | 0,8202 | 0,5533 | -0,2670 | 0,7176 | 0,3901 | -0,3274 |
| pipeline | 0,8302 | 0,5618 | -0,2684 | 0,7325 | 0,3983 | -0,3342 |

Per la pipeline Raspberry riduce inoltre l'accuracy detection da `0,7097` a
`0,3906` (`-0,3191`), mentre la precision cambia soltanto di `-0,0051`.

## Prestazioni e limiti residui

L'ONNX selezionato impiega sul Mac M4 CPU `23,86 ms` medi per una singola
inferenza zero-tensor (mediana `23,91`, p95 `28,84`) dopo tre warmup. Il target
Raspberry Pi 3 non era raggiungibile né risolvibile durante il ciclo: latenza,
RSS e limite di 15 secondi del comando completo restano quindi esplicitamente
non qualificati sul target e non vengono stimati da misure Mac.

Gli artifact di ogni valutazione contengono metriche per sorgente, sito, classe
e dimensione, tassonomia errori, deduplicazione, `errors.csv` e 12 esempi
visuali. Il prossimo intervento consigliato è annotare una validation PhenoCam
positiva realmente multi-sito; con due sole box da un sito, nuovi fine-tuning
non possono essere selezionati con evidenza sufficiente. In seconda priorità va
riesaminata la fusione, perché su TEST-ID perde 27 target della full view.

## Verifica finale

- suite applicativa: 244 test superati, 7 esclusi;
- suite dataset: 50 test superati;
- sintassi POSIX degli script `batch.sh` e `installer.sh` verificata;
- moduli Python modificati compilati senza errori;
- 2.968 checksum del dataset verificati e fingerprint ricalcolato;
- receipt di freeze ed export, hash PT/ONNX e artifact dei test finali verificati;
- `git diff --check` completato senza errori.
