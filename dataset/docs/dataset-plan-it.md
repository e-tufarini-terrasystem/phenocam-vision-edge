<!--
Scopo architetturale: sintesi operativa italiana della specifica autorevole per
il primo dataset misto di object detection del repository.
Contesto di sistema: rende immediatamente verificabili per il team obiettivo,
fonti, composizione, controlli e separazione dei dati interni.
-->

# Dataset esterno misto: sintesi dell'iterazione 1

Stato: specifica approvata il 2026-08-26. La versione inglese
[`dataset-plan-en.md`](dataset-plan-en.md) è normativa in caso di ambiguità.

Decisione di implementazione del 2026-08-27: questa iterazione costruisce
soltanto il dataset pubblico di training. L'inventario, la validation operativa
e l'eventuale test cieco sui dati interni sono rinviati a una seconda fase e
devono restare visibili come `operational_validation: pending`; il rinvio non
autorizza a sostituirli con uno split pubblico né a inserire immagini interne
nel training.

## Obiettivo

Il primo esperimento deve misurare quanto un dataset esterno progettato con cura
generalizzi al dominio reale PhenoCam/Agrocam **prima** di inserire immagini
interne nel training.

Il training iniziale usa quindi solo fonti pubbliche approvate. Download,
filtraggio, normalizzazione, deduplicazione, bilanciamento, campionamento e
manifest devono poter terminare senza accesso all'archivio interno. Se l'archivio
non è disponibile, il dataset pubblico viene comunque completato e lo stato
riporta `operational_validation: pending`.

## Ontologia compatibile con il runtime

Si mantiene la tabella COCO completa a 80 classi e si generano label solo per:

| ID | Classe YOLO | Tipi sorgente |
|---:|---|---|
| 0 | `person` | persona |
| 1 | `bicycle` | bicicletta, anche senza ciclista |
| 2 | `car` | auto, van, pickup |
| 3 | `motorcycle` | moto, motorino/scooter |
| 5 | `bus` | autobus, coach |
| 7 | `truck` | camion, trattore, mietitrebbia, altri mezzi agricoli semoventi |

Il sottotipo originale (`van`, `pickup`, `tractor`, `combine`, ecc.) resta nei
metadati anche quando viene compilato in una classe COCO più ampia. Non si passa
a due classi in questa iterazione: richiederebbe un esperimento separato su
modello e runtime.

## Fonti e composizione

Le sole fonti ammesse sono:

- **PhenoCam Network v3**, CC BY 4.0: contesto fixed-camera, agricoltura/natura,
  stagioni, negativi e positivi annotati manualmente;
- **Open Images V7**, annotazioni CC BY 4.0 e licenza immagine da verificare per
  ogni file: box diversificati, classi rare, occlusioni e negativi verificati.

COCO, BDD100K, VisDrone, AU-AIR, CrowdHuman, KITTI, Cityscapes e Mapillary sono
esclusi per ridondanza, provenienza o licenze non sufficientemente compatibili.

Composizione raccomandata, calcolata su frame sorgente unici prima dei crop:

| Fonte | Positivi | Negativi | Totale | Quota |
|---|---:|---:|---:|---:|
| PhenoCam v3 | 15 | 706 | 721 | 36,05% |
| Open Images V7 verificato | 1.229 | 50 | 1.279 | 63,95% |
| **Totale** | **1.244** | **756** | **2.000** | **100%** |

Questa composizione sostituisce il contratto provvisorio dopo la revisione umana
completa: fra i 350 candidati positivi PhenoCam solo 15 contenevano target. I 379
positivi mancanti sono selezionati in modo deterministico dal pool Open Images
già licenziato, scaricato e deduplicato; PhenoCam conserva il ruolo di ponte di
dominio fixed-camera soprattutto tramite negativi reali.

Strati primari mutuamente esclusivi: 22,8% negativi ordinari fixed-camera, 15%
hard negative/confusori, 25% positivi comuni, 25% positivi difficili e 12,2%
positivi rari/ambientali. Fra i positivi servono almeno 30% con target piccoli,
20% occlusi o tagliati dal bordo, 10% multipli/sovrapposti e non più di 50% facili.

Minimi di istanze: 900 persone, 750 famiglia car, 270 famiglia truck, 200
biciclette, 200 moto e 180 bus; almeno 2.500 istanze totali.

Questi minimi si applicano congiuntamente ai 1.244 frame positivi. La selezione
supplementare li impone usando i conteggi revisionati della base Open Images e
dei 15 positivi PhenoCam confermati, mantenendo al massimo cinque frame per
gruppo di provenienza Open Images.

## Campionamento e deduplicazione

Il seed è `20260826`. Dopo verifica di licenza, decodifica e metadati, si ordina
stabilmente per fonte, ID, timestamp e URL. Si riempie prima la quota più rara e,
all'interno dello strato, si massimizza la diversità tramite embedding fissato e
versionato. Il modello corrente può solo dare priorità alla revisione con soglia
`0.05`; non produce ground truth.

Si applicano SHA-256 del file, SHA-256 dei pixel RGB normalizzati EXIF, pHash a
64 bit con distanza ≤6, similarità SSCD (≥0,98 candidato automatico; 0,95–0,98
revisione) e legami di metadati. I componenti duplicati o di sequenza ricevono un
solo `group_id` indivisibile. Nessun parent, crop, duplicato o evento attraversa
un confine di dataset.

Ogni frame accettato produce il frame completo; può produrre fino a due crop
positivi e un hard-negative crop usando la geometria esatta del runtime. I crop
ereditano gruppo, licenza e destinazione del parent e non alterano le percentuali
delle fonti.

## Annotazioni e controlli

Si conservano due livelli: annotazione sorgente completa e label YOLO compilata
`class_id x_center y_center width height`. Un negativo non ha un label file. Le
coordinate sono normalizzate e tutti i target visibili devono essere annotati;
frame incompleti o ambigui vengono esclusi.

Ogni immagine deve decodificare in RGB, essere almeno 640×480, avere provenienza
e licenza complete e box validi. Tutti i negativi sono verificati in modo
indipendente; hard negative, classi rare, duplicati candidati e mapping ambigui
ricevono revisione umana. Nessun gruppo correlato può attraversare i confini.

L'output comprende `data.yaml`, immagini/label di training, annotazioni sorgente,
manifest delle fonti, licenze, gruppi e rifiuti, statistiche e checksum. I dati
interni restano in storage e manifest separati e ad accesso ristretto.

## Dati interni e valutazione

Quando disponibili, le immagini interne vengono inventariate per dimensione,
diversità, siti, camere, copertura temporale, classi e failure mode. Si raggruppano
per `site_id + camera_id + sequence/event`, o con l'equivalente più conservativo.

Sono usate principalmente come validation operativa. Un test cieco separato si
crea solo con gruppi indipendenti sufficienti e non viene mai usato per training,
iperparametri, soglie, composizione o selezione iterativa. Come copertura minima
si cercano 60–100 negativi, 50 immagini positive per persone e 75 positive per
veicoli, distribuite su più siti/camere dove possibile.

Se i dati sono insufficienti per validation e test significativi, tutto il
materiale iniziale diventa validation/development e non si formula alcuna
affermazione finale sulle prestazioni in produzione. Le immagini interne già
ispezionate non possono entrare nel test cieco.

Dopo il training pubblico si analizzano miss, oggetti piccoli/lontani, occlusioni,
bordi, mezzi agricoli, falsi positivi ricorrenti, sfondi difficili e pattern di
sito/camera. Solo una seconda proposta approvata potrà introdurre esempi interni
nel training. Nessuna percentuale interna viene fissata prima di misurare
l'archivio e restano sempre separati validation e test indipendenti.

## Criterio di completamento

Il dataset pubblico è accettato con 2.000 frame unici, quote entro tolleranza,
minimi di istanze soddisfatti, provenienza/licenze al 100%, negativi verificati,
zero leakage, manifest riproducibili e checksum validi. Può essere dichiarato
completo anche con validation operativa ancora pending, ma non come prova di
prestazioni in produzione.
