# Training v5 — esito finale

> Historical record for this iteration; use the [current guide](../development.md#current-workflow).
> Commands refer to the original workflow; ignored artifacts require local evidence.

Data: 2026-09-07.

Stato: **v5 promossa con un trade-off dichiarato**. Il modello di deployment è
YOLO26n; YOLO26x è stato usato soltanto come teacher per le proposte da
revisionare e non viene distribuito sul Raspberry Pi.

## Risultato

Il finalista è la media uniforme di tre head YOLO26n addestrate con seed 42, 17
e 73, interpolata al 15% con il checkpoint nano originale. La soup non aumenta
parametri, memoria o costo d'inferenza rispetto a YOLO26n:

- 2.408.932 parametri fused e 5,4 GFLOPs;
- soglia runtime congelata: `0.47`;
- checkpoint: `models/yolo26n-v5.pt`;
- ONNX statico, batch 1, 640×640, opset 20: `models/yolo26n-v5.onnx`;
- output verificato: `[1, 300, 6]`, 80 classi COCO.

La decisione privilegia il dominio richiesto: l'ONNX v5 migliora F1 di `+0,04491`
sul PKLot holdout e di `+0,01169` su TEST-OOD, con una regressione di `-0,00520`
su TEST-ID. Non è quindi un miglioramento uniforme su ogni slice.

## Dataset e assenza di leakage

`dataset-v5` contiene 2.004 immagini e 6.402 box:

| Split | Immagini | Box | Uso |
| --- | ---: | ---: | --- |
| train | 1.792 | 4.580 | training |
| validation | 206 | 509 | ricetta, checkpoint e soglia |
| PKLot holdout | 6 | 1.313 | aperto solo dopo il freeze |

Fingerprint SHA-256:
`406a4c3d12b1e69e5c5a807fa9dcd154ba352256a6b710bb49960a018d5acd13`.
Il task CVAT 17 revisionato fornisce 27 scene ufficiali PKLot bilanciate per
vista, meteo e occupazione; 18 vanno in train, 3 in validation e 6 nel holdout.
Il task 13 resta escluso perché le sue 1.729 box YOLO26x sono ancora proposte
automatiche, non ground truth umana.

Sono stati controllati checksum, identità decodificate, pHash, camera-day e
assenza di sovrapposizioni con i test canonici. I crop di training usano la
stessa griglia runtime 5×3 e non attraversano gli split.

## Ricerca e ricetta

La prima griglia ha seguito la
[ricetta ufficiale YOLO26 per piccoli dataset](https://docs.ultralytics.com/guides/yolo26-training-recipe)
e la documentazione sul
[fine-tuning e layer freezing](https://docs.ultralytics.com/guides/finetuning-guide):
AdamW, 50 epoche, patience 20, `lr0=0.001`, mosaic 0,5, senza mixup o
copy-paste, con ablation `freeze=10`. Entrambe hanno adattato troppo il modello.

La prova conservativa ha quindi congelato i layer 0–22 e addestrato soltanto
Detect per 30 epoche, patience 10, AdamW, `lr0=0.0001`, `lrf=0.1`, warmup 2,
mosaic 0,5 e `close_mosaic=5`. Questa scelta è coerente con il rischio noto di
[catastrophic forgetting](https://arxiv.org/abs/1708.06977).

Per mantenere le capacità generali sono state usate due tecniche senza costo
runtime aggiuntivo:

- interpolazione col modello originale, motivata da
  [WiSE-FT](https://arxiv.org/abs/2109.01903);
- media dei pesi di run indipendenti, motivata da
  [Model Soups, ICML 2022](https://proceedings.mlr.press/v162/wortsman22a.html).

L'allineamento dei crop tra training e runtime segue l'evidenza di
[SAHI](https://arxiv.org/abs/2202.06934). PKLot è CC BY 4.0 e viene attribuito
come Almeida et al., *PKLot – A robust dataset for parking lot classification*;
la [pagina ufficiale](https://web.inf.ufpr.br/luizoliveira/research-interests/pklot/)
descrive sorgente e licenza.

## Addestramenti

| Run | Seed | Best epoch | mAP50–95 train-val | Durata |
| --- | ---: | ---: | ---: | ---: |
| AdamW full | 42 | 43 | 0,45557 | 9.562 s |
| AdamW freeze 10 | 42 | 44 | 0,49882 | 7.581 s |
| AdamW head | 42 | 19 | 0,54488 | 3.480 s |
| AdamW head | 17 | 22 | 0,54354 | 3.566 s |
| AdamW head | 73 | 19 | 0,54400 | 3.344 s |

Gli hash dei tre checkpoint head sono rispettivamente:

- `2deaae6b48b4784acf684fac8fe74b91cde92772bd2d7b2898dc458ae3c51320`;
- `aaafc2af4047a424d16166cad6b9166c4d98d265b589805188ccd7b1679797ad`;
- `b186c771f4aeaed5afba4dc0ffbfbc1e6fa90425a64c2159a4ebded2489bb561`.

PyTorch MPS ha segnalato che `index_put_with_accumulate_mps` non dispone di
un'implementazione deterministica. Seed, configurazione e dataset sono
registrati, ma non si dichiara riproducibilità bit-per-bit su MPS.

## Selezione su validation

La metrica primaria è l'F1 della pipeline reale: immagine completa, 15 crop e
deduplicazione globale. Il gate richiedeva almeno `+0,005` F1 sul baseline,
nessuna perdita Open Images oltre 0,01, recall PhenoCam non inferiore e recall
PKLot non inferiore.

| Modello | mAP complessiva | mAP Open Images | F1 pipeline | Esito |
| --- | ---: | ---: | ---: | --- |
| YOLO26n originale | 0,52545 | 0,53277 | 0,43307 | baseline |
| AdamW full | 0,45917 | 0,44107 | 0,39535 | scartato |
| AdamW freeze 10 | 0,48868 | 0,48312 | 0,37764 | scartato |
| AdamW head non interpolato | 0,54010 | 0,53598 | 0,40228 | scartato |
| Head seed 42, alpha 10% | 0,52877 | 0,53517 | 0,43640 | instabile |
| Soup tre seed, alpha 15% | 0,53165 | 0,53890 | 0,43825 | selezionato |

Il baseline fine usa soglia 0,46; il finalista usa 0,47. Le perturbazioni
leave-one-seed-out a alpha 15% producono F1 `0,43922`, `0,44024` e `0,43800`;
media `0,43915`, guadagno `+0,00608`. La selezione congelata ha SHA-256 modello
`93cca0065d9d06da5c78edadb388967a88b32fc2f0557a09fa82e1fcf8a638a3`.

## Test finali del deployment ONNX

Il confronto usa l'ONNX baseline alla soglia 0,46 e l'ONNX v5 alla soglia
congelata 0,47. Modello e soglie non sono stati modificati dopo l'apertura.

| Split | F1 baseline | F1 v5 | Delta | Precision v5 | Recall v5 |
| --- | ---: | ---: | ---: | ---: | ---: |
| PKLot holdout | 0,29547 | 0,34037 | +0,04491 | 0,81977 | 0,21478 |
| TEST-ID | 0,48627 | 0,48106 | −0,00520 | 0,44007 | 0,53047 |
| TEST-OOD | 0,55688 | 0,56857 | +0,01169 | 0,95322 | 0,40510 |
| OOD senza camera-stress | 0,82799 | 0,83971 | +0,01172 | 0,96567 | 0,74282 |
| sola camera-stress Raspberry | 0,49572 | 0,50752 | +0,01180 | 0,94866 | 0,34643 |

Le metriche standard PyTorch confermano mAP50–95 TEST-ID `0,58229 → 0,58256`
e TEST-OOD `0,13260 → 0,13421`. Sul PKLot holdout v5 ottiene mAP50–95
`0,09184`; le auto lontane restano il limite principale.

PT e ONNX coincidono sui conteggi PKLot e OOD. Sulla validation, l'export ONNX
perde una sola associazione dopo la deduplicazione: F1 `0,43825 → 0,43669`.
Il delta ONNX contro ONNX sulla validation è quindi `+0,00362`, sotto il gate
PT originale; la promozione è sostenuta dai guadagni finali PKLot/OOD e accetta
esplicitamente il piccolo trade-off TEST-ID.

TEST-OOD è un benchmark di regressione storico già usato in precedenti cicli,
non un test indefinitamente incontaminato. Il PKLot holdout ha soltanto sei
scene della stessa vista PUCPR: molte box, ma bassa indipendenza statistica.

## Artefatti e runtime Raspberry

| Artefatto | Byte | SHA-256 |
| --- | ---: | --- |
| `models/yolo26n-v5.pt` | 5.542.277 | `93cca0065d9d06da5c78edadb388967a88b32fc2f0557a09fa82e1fcf8a638a3` |
| `models/yolo26n-v5.onnx` | 9.941.955 | `252f302257759cae6d40579fb76b74d66f87bdce2997d44f89dba85d03420379` |

Il benchmark di contratto sul Mac Apple M4 misura 20 forward CPU dopo tre
warmup: media 21,78 ms, mediana 21,47 ms e p95 24,41 ms per singola vista. Non
è un benchmark Raspberry Pi e non include le 16 viste, caricamento o output.
Prima del rilascio hardware vanno misurati sul Raspberry reale latenza
end-to-end, RSS, temperatura e throttling con lo stesso ONNX e immagini
rappresentative.

## Dati successivi utili

Altre immagini PKLot quasi adiacenti avrebbero forte ridondanza e non
risolverebbero il trade-off TEST-ID. La prossima tranche utile resta:

- 120 frame operativi train, almeno 12 camera-day e 60 per camera;
- 60 frame validation da camera-day disgiunti;
- 60 frame futuri come nuovo test sigillato.

Vanno stratificati per luce, meteo, occupazione, distanza e inquadratura. Il
task 13 può essere ammesso solo dopo revisione umana completa; non va usato come
pseudo-ground-truth per aumentare artificialmente il numero di immagini.

## Comandi riproducibili

```sh
.venv-export/bin/python -m scripts.training_v5.train adamw-head 42
.venv-export/bin/python -m scripts.training_v5.train adamw-head 17
.venv-export/bin/python -m scripts.training_v5.train adamw-head 73
.venv-export/bin/python -m scripts.training_v5.interpolate \
  --parent output/training-v5/runs/adamw-head/weights/best.pt \
  --parent output/training-v5/runs/adamw-head-seed17/weights/best.pt \
  --parent output/training-v5/runs/adamw-head-seed73/weights/best.pt \
  --name adamw-head-soup-refined
.venv-export/bin/python -m scripts.training_v5.selection freeze
PYTHONPATH=. .venv-export/bin/python -m scripts.training_v5.finalize
```

Le ricevute complete di training, selection, test ed export sono sotto
`output/training-v5/`; gli artifact di lavoro sono ignorati da Git, mentre PT,
ONNX e ricevuta finale sono versionati in `models/`.
