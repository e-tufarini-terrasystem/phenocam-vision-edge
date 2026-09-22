# V6 — primo batch operativo per annotazione

> Historical record for this iteration; use the [current guide](../development.md#current-workflow).
> Commands refer to the original workflow; ignored artifacts require local evidence.

Stato: correzioni completate in CVAT e dataset v6 materializzato. Inclusi 57
frame interni; 3 frame ambigui conservati in CVAT ed esclusi dal dataset.
Autorizzazione: eliminare le copie identiche e creare i task sul CVAT Docker locale.

## Confini e selezione

- Rimossi soltanto i 69 JPEG di `phenozero2` identici ai corrispondenti file
  di `phenozero1`; originali conservati e hash registrati nella ricevuta locale.
- Le 240 immagini TEST-OOD e tutti i loro giorni restano esclusi dal training.
- Ricostruzione deterministica dello split originale con seed 20260831:
  326 immagini Raspberry su 14 giorni e 703 immagini TS02 fino al 27 agosto
  su 24 giorni, coerenti con le 1.029 immagini documentate in v3.
- Giorni sigillati Raspberry: 2025-11-04, 2025-11-05, 2025-12-18 (94 frame).
- Giorni sigillati TS02: 2026-08-09, 12, 14, 21, 27 (147 frame).
  I totali ricostruiti coincidono con i 241 frame e gli 8 giorni documentati;
  il manifest originale non è più disponibile. Per prudenza, tutta la sorgente
  Raspberry e TS02 fino al 27 agosto sono esclusi da questo batch.
- Quote: TS02 24 frame su 12 giorni (28 agosto–8 settembre), PhenoZero1
  24 frame su 2 giorni (8–9 settembre), PhenoZero2 12 frame sull'8 settembre.
  La selezione distribuisce i frame nel tempo e non usa i risultati dei test.
- PhenoZero1 dal 10 settembre e TS02 dal 9 settembre non sono candidati train
  in questo batch. Validation e test v6 devono essere definiti separatamente;
  servono nuovi giorni per PhenoZero2 e nuove acquisizioni Raspberry.

## Invarianti e rischio

Sessanta identità, SHA-256 e immagini decodificate uniche; nessun riuso di
identità, hash o gruppi del test storico. Ogni giorno selezionato ha solo ruolo
train. Le proposte automatiche non sono ground truth e richiedono revisione
completa. Non si avvia il training e non si modifica alcun task precedente.

Il limite principale è la scarsa diversità temporale PhenoZero, soprattutto
PhenoZero2: dodici immagini dello stesso giorno non sono dodici scene indipendenti.
Il batch rappresenta tre camere addestrabili, non tutte le quattro inventariate.
Le anteprime PhenoZero1/2 mostrano lo stesso luogo con resa cromatica diversa:
vanno considerate camere correlate e raggruppate per giorno anche attraverso
i due identificatori. Non costituiscono evidenza di due siti fisici indipendenti.

## Task disponibili

Progetto [Phenocam privacy detector v6](http://localhost:8080/projects/2), ID 2,
con guida di annotazione integrata. Ogni task contiene un solo job.

| Camera | Task | Job | Frame | Box proposte |
| --- | ---: | ---: | ---: | ---: |
| PhenoZero1 | 18 | [43](http://localhost:8080/tasks/18/jobs/43) | 24 | 74 |
| PhenoZero2 | 19 | [44](http://localhost:8080/tasks/19/jobs/44) | 12 | 32 |
| TS02 | 20 | [45](http://localhost:8080/tasks/20/jobs/45) | 24 | 246 |
| Totale | — | — | 60 | 352 |

Proposte dal modello locale `models/yolo26n-v5.onnx`, SHA-256
`252f302257759cae6d40579fb76b74d66f87bdce2997d44f89dba85d03420379`,
soglia 0,30, full frame più 15 crop, CPU con quattro thread. Non è stato usato
YOLO26x: checkpoint e ambiente teacher non erano disponibili localmente.
Classi operative e attributi coincidono con il progetto v3; `ambiguous` resta
disponibile per i casi irrisolti. I task sono nel subset `train`; alla creazione
erano nello stato `annotation`.

Verifiche completate:

- Readback SHA-256 di tutti i 60 JPEG originali da CVAT, identici alle sorgenti.
- Nomi, ordine, dimensioni, classi e coordinate di tutte le 352 box verificati.
- Sessanta identità, hash file e hash pixel unici; nessuna corrispondenza con
  l'intero manifest sorgenti v3, oltre alle esclusioni dei giorni riservati.
- Zero errori di inferenza; anteprime dei tre task ispezionate localmente.
- Task preesistenti invariati. Nessun training avviato.

La ricevuta locale conserva anche gli hash del codice di preannotazione,
inclusa la versione di lavoro del deduplicatore presente all'esecuzione.
Per annotare, correggere tutte le proposte, aggiungere i target mancanti e
marcare completed soltanto dopo avere controllato tutti i frame.

## Revisione visuale dopo annotazione umana

Questa sezione descrive la prima revisione, prima delle successive correzioni.
L'esito finale è riportato nella sezione seguente.

L'11 settembre i tre job risultano `acceptance/completed`: 213 box nel task 18,
106 nel task 19 e 249 nel task 20, totale 568. Nessun track o tag `ambiguous`.
Esportati COCO e backup completi con prefisso
`dataset/workspace/annotation/exports/v6-task<ID>-human-20260911`.

Codex ha esaminato tutti i 60 frame nelle anteprime annotate, approfondendo i
candidati con ritagli alla risoluzione originale. Non è una seconda revisione
umana indipendente. Il controllo formale di tutte le 568 box passa: rettangoli
non ruotati, coordinate finite e valide entro l'immagine, classi ammesse.

Il controllo visuale richiede correzioni: 18 segnalazioni confermate e 7 punti
da risolvere, distribuiti complessivamente su 18 frame. Tra gli errori confermati:
duplicati e frammenti residui, box su pavimentazione/materiale di cantiere/ombra,
una vettura etichettata `person`, furgoni etichettati `bus` o `truck`, un'auto e
due persone mancanti. I casi irrisolti riguardano soprattutto forte occlusione,
una possibile duplicazione e la tassonomia di muletto e mezzi parzialmente visibili.

Rapporto locale con ID delle shape, frame CVAT a base zero, link diretti e
ritagli originale/annotato:

- Rapporto visuale HTML (`dataset/workspace/training-v6/review-2026-09-11/review.html`, local evidence).
- Risultati strutturati e hash degli snapshot (`dataset/workspace/training-v6/review-2026-09-11/review.json`, local evidence).
- Nella stessa cartella: snapshot `task-18.json`, `task-19.json`, `task-20.json`,
  verifica `geometry.json` e anteprime di tutti i frame.

Le annotazioni CVAT non sono state modificate dalla revisione. I frame molto
scuri 18/23 e 19/0 non mostrano target riconoscibili con sicurezza; l'assenza di
segnalazioni non certifica l'assenza di target. Risolvere le segnalazioni in CVAT,
riesportare e ricontrollare prima di materializzare v6. Se un caso resta
irrisolvibile, applicare la regola `ambiguous` e documentare l'eventuale riduzione
del batch: non includerlo tacitamente come negativo.

## Esito finale: CVAT corretto e dataset v6 creato

Le modifiche manuali successive dell'utente sono state lette e preservate.
Codex ha eliminato tre duplicati residui nel task 18 (50102, 50110, 50570),
aggiunto tre auto parzialmente visibili nei frame 17–19, rifinito due box distinte
nel task 20/frame 22 (50412, 50414) e incluso la testa nella box person 50736.
Conservate le decisioni dell'utente su muletto `truck` e cassonato leggero `car`.
Snapshot prima/dopo e readback verificano che le altre shape siano invariate.

Tre frame ricevono il tag `ambiguous` e non entrano nella v6:

- Task 18/frame 10: classe del mezzo fortemente occluso non risolvibile.
- Task 18/frame 23 e task 19/frame 0: buio insufficiente a confermare l'assenza
  completa di target. Non sono stati importati come negativi.

CVAT conserva 564 box complessive; 551 appartengono ai 57 frame ammessi.
Export finali e backup in `dataset/workspace/annotation/exports/`, prefissi
`v6-task18-final2-20260911`, `v6-task19-final-20260911`,
`v6-task20-final-20260911`. Il suffisso `final2` include la rifinitura dei bordi
superiori delle tre auto aggiunte. Gli export precedenti sono conservati.

Artefatto: dataset/dataset-v6/dataset.yaml (`dataset/dataset-v6/dataset.yaml`, local evidence).
Ricostruzione, quando la destinazione non esiste:

```sh
dataset/.venv/bin/python -m scripts.training_v6.dataset
```

| Componente | Immagini |
| --- | ---: |
| Training | 2.020 |
| Validation, identica a v5 | 206 |
| PKLot holdout, identico a v5 e fuori dal YAML train/val | 6 |
| Totale | 2.232 |

Le 2.004 immagini v5 e tutte le loro label sono copiate e verificate, senza
hard link. Aggiunte 57 immagini intere (PhenoZero1 22, PhenoZero2 11, TS02 24)
e 171 crop generati con le regole v5. Totale 7.119 box, incluse 551 box nelle
nuove immagini intere e 166 nelle nuove viste crop. Le classi COCO e il relativo
ordine restano quelli di v5. Nessun training è stato avviato.

L'audit (`dataset/dataset-v6/metadata/audit.json`, local evidence) passa i controlli su
hash, immagini e identità duplicate, geometria, corrispondenza COCO/snapshot,
esclusioni e separazione dei gruppi. La verifica finale confronta tutti i file
immagine/label ereditati con v5 e controlla tutte le 7.119 righe YOLO. Nessun
frame dei test storici è stato aggiunto. Fingerprint v6:
`e1491ab0e3bc073c167026f7e0dd0057c86bede700baa2122a8d3a4126dbfddc`.

Ricevute e prove locali in
`dataset/workspace/training-v6/corrections-2026-09-11/`: `plan.json`,
`receipt.json`, `refinement.json`, `task-<ID>-final.json`, `verification.json`.
La revisione resta umana seguita da correzione visuale AI, non una seconda
revisione umana indipendente. Validation/test interni su giorni separati restano
da preparare prima di misurare il miglioramento operativo di un modello v6.

## Struttura locale prima dell'esecuzione

### Completamento delle correzioni e materializzazione v6

La richiesta successiva dell'utente, che non riesce a completare manualmente
le correzioni, viene eseguita sulle ultime annotazioni CVAT, preservando le sue
modifiche. Invarianti: snapshot prima/dopo, patch limitate alle shape interessate,
nessuna sovrascrittura di v5, validation v5 identica e test storici esclusi.
Il rischio principale è trasformare un oggetto incerto in una label arbitraria:
i frame irrisolvibili ricevono `ambiguous` e non entrano nel dataset.

Struttura prevista prima dell'implementazione, riusando le librerie del builder v5:

```text
dataset/workspace/training-v6/corrections-2026-09-11/
  corrections.py                  (~150 linee produttive; patch CVAT con readback)
  task-<ID>-before/after.json      (snapshot e provenienza)
scripts/training_v6/
  __init__.py                     (~1 linea produttiva)
  sources.py                      (~160 linee produttive; sorgenti revisionate e gate)
  dataset.py                      (~190 linee produttive; materializzazione atomica e audit)
dataset/config/training-v6.json   (hash e regole del batch)
dataset/dataset-v6/               (artefatto generato, immagini escluse da Git)
```

Si riusano conversione YOLO, crop e audit v5. Le verifiche sull'artefatto reale
confrontano integralmente i file v5 ereditati, i conteggi CVAT/YOLO, le esclusioni
`ambiguous`, gli hash e la separazione dei gruppi tra split.
PyYAML, già importato dal builder v5 ma assente dall'ambiente locale corrente,
è stato installato come `6.0.2` e dichiarato nelle dipendenze workstation del
dataset. Nessuna dipendenza aggiunta al runtime edge.

Si riusano screening e bundle CVAT esistenti, senza nuove dipendenze o modifiche
al runtime. Gli script sono ricevute operative locali, non nuove interfacce CLI.

```text
dataset/workspace/training-v6/          (ignorato da Git)
  prepare.py                          (~150 linee produttive; selezione e bundle)
  upload.py                           (~130 linee produttive; task CVAT e readback)
  duplicate-removal-2026-09-11.json     (ricevuta eliminazione)
  reconstructed-sealed-days.json       (ricostruzione conservativa)
  selection.csv                       (60 sorgenti e provenienza)
  selection.json                      (regole, conteggi e hash)
  screening/                          (proposte v5, mai label umane)
  bundles/                            (tre pacchetti COCO e immagini)
  cvat.json                           (ID, job, conteggi e verifica)
```

## Ripristino CVAT

Il proxy Traefik non partiva perché il vecchio percorso bind mount conteneva
una directory vuota al posto di `components/analytics/grafana_conf.yml`.
Ripristinato il file della versione CVAT v2.71.0 dal repository ufficiale e
riavviato il container esistente; nessun volume dati ricreato.
SHA-256 del file: `bb828236e1ff8a6b94cd1fc949c65a94a362a35753cc7253548eec6b7b034127`.
