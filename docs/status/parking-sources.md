# Valutazione sorgenti parcheggio — gate CVAT

> Historical artifact labels use normalized model versions, not filesystem paths.
> Exact commands, identifiers and paths remain in the original document:
> `git cat-file blob c3657ea1168e38bcccf70176c58030fd01166d66` from the
> [source snapshot](https://github.com/e-tufarini-terrasystem/phenocam-vision-edge/tree/cc63857c12c5553c2e3451854863edf7c0705e2a).

> Historical record for this iteration; use the [current guide](../development.md#current-workflow).
> Commands refer to the original workflow; ignored artifacts require local evidence.

Data: 2026-09-03.

Stato: due gate PKLot auto-annotati con YOLO26x sono disponibili in CVAT;
nessuna immagine è stata ammessa al training set.

## Esito

È possibile aggiungere immagini di parcheggi con una licenza compatibile con
uso commerciale. La sorgente più pertinente trovata è **PKLot**, pubblicata
dall'Universidade Federal do Paraná con licenza CC BY 4.0. Il dataset contiene
12.417 immagini 1280×720 da tre viste fisse di due parcheggi, in condizioni di
sole, nuvolosità e pioggia.

La licenza CC BY 4.0 permette condivisione e adattamento anche commerciale, ma
richiede attribuzione e indicazione delle modifiche. Questa valutazione tecnica
non sostituisce una revisione legale su privacy, marchi o altri diritti che la
licenza copyright non può concedere.

Fonti:

- pagina ufficiale PKLot e dichiarazione di licenza:
  <https://web.inf.ufpr.br/luizoliveira/research-interests/pklot/>;
- licenza CC BY 4.0: <https://creativecommons.org/licenses/by/4.0/>;
- articolo da citare: Almeida et al., *PKLot – A robust dataset for parking lot
  classification*, 2015.

## Campioni in CVAT

### PKLot — sorgente raccomandata

- Task: `PKLot pilot - 50 scene review - CC BY 4.0`
- ID: `13`
- URL locale: <http://localhost:8080/tasks/13>
- Contenuto: 50 frame, un job e 1.729 suggerimenti YOLO26x da correggere:
  1.645 `car`, 47 `person`, 20 `truck` e 17 `bus`.
- Scopo: giudicare inquadratura, densità, meteo, risoluzione utile e distanza
  dal dominio operativo. Le predizioni non sono ground truth.

Il campione da 50 immagini è un derivato pubblico 640×640 trasportato da
`ajankelo/pklot_50`, revisione
`1fc8d17a6617ec0ea4d098ff55b497b6a40187ec`. Serve solo per il gate visivo.
Se il gate passa, le immagini candidate al training devono provenire
dall'archivio ufficiale PKLot, non da questo derivato.

Ricevuta e manifest locali ignorati da Git:

- `dataset/workspace/sources/pklot-pilot/receipt.json`;
- `dataset/workspace/sources/pklot-pilot/manifest.csv`.

L'auto-annotazione usa il checkpoint YOLO26x con SHA-256
`9fdd44a31c504547ffb81d2c6d9e6dac3493c8eaa8b0398d3f43bae6c7003e92`,
soglia 0,20, inferenza intera e tiled a 1280 px e deduplicazione IoU 0,50.
Il read-back del task contiene esattamente le stesse 50 immagini e 1.729 box
dell'archivio importato dopo l'arrotondamento CVAT a due decimali.

### PKLot ufficiale — campione bilanciato 0.1.4

- Task: del ciclo 0.1.4 (PKLot balanced official - 27 CC BY 4.0 - review gate)
- ID: `17`
- URL locale: <http://localhost:8080/tasks/17>
- Contenuto: 27 frame originali 1280×720 e 1.835 suggerimenti YOLO26x:
  1.751 `car`, 42 `person`, 33 `truck` e 9 `bus`.
- Campionamento: 3 viste × 3 meteo × 3 fasce di occupazione, con giorni e
  fasce orarie differenti.

I frame provengono dall'archivio ufficiale, non dal derivato del task 13. La
selezione non ha corrispondenze esatte, decodificate o pHash a distanza ≤4 con
0.1.4; non contiene coppie pHash interne a distanza ≤4 e non riusa timestamp del
task 13. Il read-back CVAT ha preservato tutte le 27 immagini e tutte le 1.835
geometrie. Manifest, ricevuta e audit sono in
`dataset/workspace/sources/pklot-balanced-27/` e restano ignorati da Git.

Il task provvisorio 16 è stato sostituito perché conteneva un timestamp già
presente nel task 13. Il suo backup locale è conservato; il solo task da
revisionare è il 17.

### Open Images V7 — confronto, non raccomandato

- Task: `Parking lot pilot - Open Images V7 - 17 CC BY 2.0`
- ID: `12`
- URL locale: <http://localhost:8080/tasks/12>
- Contenuto: 17 immagini e 39 box Open Images da verificare.

Le immagini hanno label umana positiva `Parking lot`, licenza per immagine CC
BY 2.0, metadati completi e nessuna sovrapposizione con i manifest correnti.
Il controllo visivo mostra però soprattutto ritratti ravvicinati di singole
auto: la provenienza è lecita e auditabile, ma la copertura del dominio
parcheggio a camera fissa è debole.

## Limiti di PKLot

- Le annotazioni originali descrivono piazzole `occupied/vacant`, non box delle
  classi operative `person`, `car`, `motorcycle`, `bus` e `truck`.
- I 12.417 frame provengono da due soli parcheggi e tre viste; sequenze a cinque
  minuti producono forte ridondanza temporale.
- Il dataset è diurno, brasiliano e ripreso dall'alto: non copre notte, neve o
  tutte le geometrie delle camere operative.
- Le auto dense e piccole sono utili per stressare il detector, ma potrebbero
  sovrarappresentare una geometria diversa da quella di produzione.

## Inventario ufficiale

L'archivio `PKLot.tar.gz` misura 3.860.376.865 byte e ha SHA-256
`df182f46184fc29fb7ef3cd293b49cae52599365d9fe9415c4e7e7ea7ded9f79`.
La verifica preventiva ha trovato 24.990 membri, tutti file regolari o
directory, senza path traversal o link: 12.417 JPEG e 12.416 XML. Un solo JPEG,
`parking2/sunny/2012-11-06/2012-11-06_18_48_46.jpg`, non ha XML e non è stato
selezionato.

Le viste e le quantità annotate sono:

- `parking1a` / UFPR04: 3.791 frame, 28 piazzole per frame;
- `parking1b` / UFPR05: 4.152 frame, 40 piazzole per frame;
- `parking2` / PUCPR: 4.473 XML e 4.474 JPEG, 100 piazzole per frame.

Ognuna delle tre viste contiene sole, nuvolosità e pioggia e dispone di frame
nelle fasce bassa (≤20%), media (>20% e <70%) e alta (≥70%) occupazione. Il
campione di 27 immagini prende un solo frame per ciascuna cella, usa date
distinte entro ogni vista e sostituisce qualsiasi sovrapposizione temporale con
il pilot.

## Dimensionamento e integrazione in 0.1.4

Il minimo utile da ammettere è **21 immagini revisionate**, mentre le 27 del
task 17 servono a revisionare anche il controllo OOD:

- **18 train**: UFPR04 e UFPR05, tutte le 18 combinazioni
  vista×meteo×densità;
- **3 validation**: PUCPR a bassa densità, una per meteo;
- **6 holdout PKLot**: PUCPR a densità media e alta, non usate per training o
  per la metrica aggregata; misurano separatamente il caso denso e lontano.

0.1.4 contiene oggi 1.621 immagini train e 203 validation. Con 18+3 immagini il
rapporto rimane praticamente invariato: da 88,87/11,13 a 88,83/11,17. I
suggerimenti destinati al train sono 551, di cui 495 `car`; se fossero tutti
confermati, il train passerebbe da 3.858 a 4.409 box e `car` da 716 a 1.211,
cioè il 27,47% delle istanze. Questo resta sotto il limite dichiarato del 30%,
equivalente a circa 630 nuove `car` sul 0.1.4 attuale.

Le tre candidate validation a bassa densità contengono 53 suggerimenti, 52
`car` e un `truck`. Se tutti fossero confermati, la validation avrebbe 206
immagini e 517 box, con 126 `car` (24,37%): sufficiente per aggiungere il dominio
senza lasciare che tre scene dense dominino la metrica aggregata. Le sei scene
holdout contengono 1.231 suggerimenti e devono essere rendicontate come slice
PKLot separata.

Questi sono massimi pre-revisione, non conteggi finali. L'ammissione deve usare
solo le box corrette e deve essere annullata o ridotta se supera 630 `car` nel
train, introduce near-duplicate o peggiora i gate 0.1.4 esistenti.

## Sonda sul modello corrente

Il campione è stato analizzato in sola lettura con
modello ONNX 0.1.3 extended, soglia 0,30 e la pipeline runtime di una vista
completa più quindici crop. Le label PKLot `space-occupied` sono state usate
soltanto come proxy di presenza: descrivono piazzole e non box `car`, quindi i
numeri seguenti non sono precision, recall o mAP.

- 50 immagini, 34 con almeno una piazzola occupata e 2.505 piazzole occupate;
- sola vista completa: 165 detection `car/bus/truck` in 19 immagini, 155 con
  centro dentro una piazzola occupata;
- pipeline completa: 428 detection in 30 immagini, 397 con centro dentro una
  piazzola occupata;
- sei scene contengono complessivamente 1.469 piazzole occupate, ma la pipeline
  produce soltanto 14 detection, 12 centrate su una piazzola occupata;
- nelle 16 scene dichiarate vuote compaiono due detection: senza revisione
  visiva non possono essere classificate come falsi positivi.

La pipeline multi-vista recupera molti oggetti rispetto alla vista intera, ma
resta quasi cieca nelle scene più dense e lontane. Questo rende PKLot un
candidato plausibile per hard-positive mining, non una sorgente da importare in
blocco. La ricevuta completa è
`dataset/workspace/sources/pklot-pilot/runtime-probe.json` (SHA-256
`e9db5271ee39fa5def0b7c58a8c05edcd6ce630967232ce16048c2ec4b1731d4`).

## Gate prima dell'ammissione

1. Correzione umana completa dei task 13 e 17; eliminare duplicati, falsi
   positivi e box parziali e aggiungere gli oggetti mancanti.
2. Marcare il task 17 completato solo dopo aver verificato tutte le 27 immagini.
   Le label di occupazione originali non diventano ground truth del detector.
3. Esportare il task 17 e rifare i controlli di hash, geometria, classi,
   deduplicazione e limite di 630 `car` train.
4. Materializzare soltanto le 18 train e 3 validation previste; conservare le 6
   scene PUCPR medio/alte come slice di valutazione separata.
5. TEST-ID e TEST-OOD restano congelati. Confrontare baseline e candidato a
   parità di modello, seed e configurazione, riportando sia le metriche 0.1.4 sia
   quelle della slice PKLot.
6. Conservare l'espansione soltanto se migliora il criterio dichiarato senza
   regressioni rilevanti; in caso contrario ripristinare 0.1.4 invariato.

## Stato verificato

- CVAT locale 2.71.0 attivo;
- task 12: 17 immagini, stato `annotation`;
- task 13: 50 immagini e 1.729 suggerimenti, stato `annotation`;
- task 17: 27 immagini e 1.835 suggerimenti, stato `annotation`;
- training set invariato;
- nessun export CVAT approvato;
- nessun training avviato con questi dati.
