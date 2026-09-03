# Dataset e training v5 — decisione misurata

Stato: dataset materializzato e verificato. Nessun modello v5 è stato ancora
addestrato; il dataset da solo non costituisce evidenza di miglioramento.

## Risultato

`dataset/dataset-v5/` è un nuovo artifact e non modifica `dataset-v4`. Contiene:

| Split | Immagini | Box | Ruolo |
| --- | ---: | ---: | --- |
| train | 1.792 | 4.580 | addestramento |
| validation | 206 | 509 | scelta di checkpoint, soglia e ricetta |
| PKLot holdout | 6 | 1.313 | controllo finale cross-view, mai training |
| totale | 2.004 | 6.402 | — |

Fingerprint SHA-256:
`406a4c3d12b1e69e5c5a807fa9dcd154ba352256a6b710bb49960a018d5acd13`.
Sono stati verificati 3.230 checksum, 2.004 identità uniche, 2.004 immagini
compilate uniche, zero camera-day condivisi tra split e zero corrispondenze di
identità o SHA-256 con i test canonici.

## Perché la v4 non ha migliorato il prodotto

Le prove v4 hanno migliorato la mAP single-view, ma hanno peggiorato la F1 della
pipeline full-image più 15 crop. Il finalista interpolato ha prodotto soltanto
`+0,0021` F1, sotto il gate `+0,005`. Le ricette erano brevi e conservative
(20 epoche, learning rate `5e-5`/`1e-4`, backbone in parte o interamente
congelato), mentre il training vedeva immagini intere e il runtime usa anche 15
crop sovrapposti. La validation PhenoCam conteneva appena due box positive:
troppo poco per scegliere con stabilità un modello operativo.

La correzione v5 affronta soltanto ciò che si può correggere senza leakage:

- mantiene integralmente train e validation v4;
- integra il task CVAT 17 completato, non le proposte automatiche;
- aggiunge tre crop per immagine positiva PhenoCam train e per immagine PKLot
  train, usando esattamente la griglia runtime 5×3 con overlap 20%;
- ruota la colonna scelta e conserva una vista per ogni riga, così tutte le 15
  posizioni sono bilanciate senza moltiplicare il dataset per sedici;
- taglia le box al bordo e scarta i frammenti con meno del 50% visibile quando
  il centro dell'oggetto è fuori dal crop.

L'allineamento tra tiling di training e inferenza è coerente con i risultati
pubblicati su small-object detection: SAHI riporta benefici sia dal sliced
inference sia dal sliced fine-tuning, e il lavoro CVPRW sul tiling applica la
stessa trasformazione nelle due fasi
([SAHI](https://arxiv.org/abs/2202.06934),
[The Power of Tiling](https://openaccess.thecvf.com/content_CVPRW_2019/html/UAVision/Unel_The_Power_of_Tiling_for_Small_Object_Detection_CVPRW_2019_paper.html)).

## Integrazione PKLot

Il task 17 revisionato contiene 27 scene ufficiali PKLot e 1.891 box umane. La
selezione evita una sequenza di immagini quasi uguali:

| Allocazione | Viste | Meteo | Date distinte |
| --- | --- | --- | ---: |
| train, 18 | `parking1a`: 9; `parking1b`: 9 | cloudy/rainy/sunny: 6 ciascuno | 18 |
| validation, 3 | `parking2`: 3 | 1 per meteo | 3 |
| holdout, 6 | `parking2`: 6 | 2 per meteo | 6 |

Il train riceve 533 box full-image e 144 box nei 54 crop. Il numero di box
`car` nel train passa da 716 a 1.368; la sua quota passa dal 18,6% al 29,9%,
mentre `person` resta la classe maggiore al 54,3%. È un riequilibrio mirato al
caso parcheggio, non una duplicazione indiscriminata. Le altre classi conservano
tutti gli esempi v4 e ricevono soltanto le occorrenze realmente presenti nelle
scene revisionate.

Le sei immagini holdout hanno 1.313 box, ma rappresentano una sola vista: sono
un buon controllo di trasferimento fra parcheggi, non un test operativo
indipendente. Il lavoro originale PKLot mostra infatti che la generalizzazione
fra parcheggi è nettamente più difficile della valutazione sulla stessa vista e
raccomanda bilanciamento e analisi delle condizioni atmosferiche
([articolo PKLot](https://www.inf.ufpr.br/lesoliveira/download/ESWA2015.pdf)).

Il task 13 resta aperto e viene escluso: le sue 1.729 box YOLO26x sono proposte,
non ground truth. Aggiungere altre immagini PKLot adesso avrebbe rendimento
decrescente: il collo di bottiglia misurato non è più il numero di automobili
esterne, ma la scarsità di frame operativi positivi e indipendenti.

## Errore di protocollo scoperto e contenuto

I 120 frame `operational-mining-informative` e i 120
`operational-dev-representative`, già revisionati, coincidono esattamente con
l'intero TEST-OOD v3/v4. Il builder v3 aveva assegnato tutti i record `internal`
al test, nonostante i nomi originari dei due cohort. Non possono entrare in v5
e continuare a sostenere un confronto storico valido.

Esiste un vero cohort `sealed_test` di 241 frame, separato per camera-day, ma le
sorgenti puntano a un volume esterno non montato e non esiste un export umano
nel repository. Il TEST-OOD storico può quindi essere usato ancora come
regression benchmark, ma non è più un test incontaminato per una successione
illimitata di esperimenti.

## Quanti dati servono davvero

Non esiste un numero universale di immagini che garantisca un modello migliore.
La successiva tranche utile deve aumentare la diversità indipendente, non solo
le box dello stesso parcheggio. Obiettivo minimo pratico:

| Ruolo futuro | Frame | Vincolo minimo |
| --- | ---: | --- |
| training operativo | 120 | 60 per camera, almeno 12 camera-day, luce e occupazione stratificate |
| validation operativa | 60 | camera-day disgiunti dal train, almeno 30 per camera |
| test finale nuovo | 60 | giorni futuri, sigillato fino al freeze |

La densità misurata nei 240 frame revisionati è circa 24,5 box per immagine:
questa tranche produrrebbe indicativamente 2.900 box train e circa 1.470 box per
validation e test. Il conteggio di box da solo non basta perché gli oggetti
nella stessa scena sono correlati; camera-day, meteo, luce, occupazione e
inquadratura restano le vere unità di diversità. Se il volume originario torna
disponibile, i 241 frame `sealed_test` possono coprire il nuovo test dopo
annotazione umana, senza riutilizzare i 240 frame storici.

## Protocollo per il prossimo training

La prima prova deve usare la baseline ufficiale YOLO26 per piccoli dataset,
senza la combinazione v4 di learning rate quasi nullo e training breve:
AdamW, `lr0=0.001`, 50 epoche, `patience=20`, `mosaic=0.5`, `mixup=0` e
`copy_paste=0`. Una sola ablation aggiunge `freeze=10`; il backbone libero resta
la prova principale. È la ricetta raccomandata dalla documentazione ufficiale
([YOLO26 training recipe](https://docs.ultralytics.com/guides/yolo26-training-recipe),
[training tips](https://docs.ultralytics.com/guides/model-training-tips)).

Checkpoint e soglia vanno scelti una volta sulla validation v5 con la pipeline
reale full più crop. Poi si apre una sola volta il PKLot holdout e, per sola
regressione storica, TEST-ID/TEST-OOD. Un miglioramento può essere dichiarato
solo se supera rumore e repliche, conserva le classi non-parcheggio e migliora
la pipeline, non soltanto la mAP single-view. Per una conclusione realmente
operativa resta necessario il nuovo test da giorni futuri.

## Struttura implementata

Le stime escludono test e fixture e rispettano il limite di 200 linee
produttive per l'orchestrazione:

```text
dataset/config/training-v5.json       (~43 linee; contratto e receipt)
scripts/training_v5/
  __init__.py                         (~1 linea)
  tiles.py                            (~41 linee; geometria e clipping)
  sources.py                          (~126 linee; import delle due sorgenti)
  dataset.py                          (~190 linee; materializzazione e audit)
tests/test_training_v5.py             (test esclusi dal limite)
dataset/dataset-v5/                   (artifact generato)
```

L'invariante centrale è che un parent e i suoi crop restino nello stesso split;
la guardia su identità e SHA-256 del test viene eseguita prima di creare
l'artifact. La complessità aggiunta è limitata alla trasformazione necessaria
per rendere coerenti training e runtime.
