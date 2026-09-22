# Handoff training 0.1.3 — 31 agosto 2026

> Historical artifact labels use normalized model versions, not filesystem paths.
> Exact commands, identifiers and paths remain in the original document:
> `git cat-file blob 07731af6e97d7431ede5923a7fecfc1d97a6ba25` from the
> [source snapshot](https://github.com/e-tufarini-terrasystem/phenocam-vision-edge/tree/cc63857c12c5553c2e3451854863edf7c0705e2a).

> Historical record for this iteration; use the [current guide](../development.md#current-workflow).
> Commands refer to the original workflow; ignored artifacts require local evidence.

> Aggiornamento 2 settembre 2026: il task 11 è stato completato, esportato e
> importato. Il build affiancato, ora denominato dataset 0.1.3 expanded, contiene 18
> nuove immagini PhenoCam revisionate. Lo stato corrente e i conteggi verificati
> sono in `docs/status/training-0.1.3.md`; il testo seguente resta la fotografia
> storica dell'handoff.

## Stato attuale

Il gate umano interno è concluso. Inventario, split operativo, screening
baseline/0.1.2, embedding, selezioni, revisione CVAT e import delle due coorti
interne sono completati. Il nuovo gate pubblico del task 11 è in revisione; non
è stato ancora avviato alcun training 0.1.3.

Nel progetto CVAT del ciclo 0.1.3 (ID 1) restano tre task attivi
e utili:

| Task | ID | Job | Immagini | Box |
|---|---:|---|---:|---:|
| Representative clean | 9 | 30-32 | 120 | 2.877 |
| Informative clean | 10 | 33-35 | 120 | 3.009 |
| Public YOLO26x high-confidence | 11 | 1 job | 40 | 192 proposte |

- Task 9: <http://localhost:8080/tasks/9>
- Task 10: <http://localhost:8080/tasks/10>
- Task 11: <http://localhost:8080/tasks/11>

La revisione conserva una box separata e stretta per ogni istanza, anche nei
frame affollati. Le box che racchiudono più oggetti non sono ammesse; gli oggetti
non separabili in modo affidabile devono essere marcati `ambiguous`.

## Cosa è stato completato

### Modello 0.1.2 e dataset

- Modello ONNX 0.1.2 stabilizzato e incluso nel package, SHA-256
  `f62fd573fd6315fc34ddc3b0f22f4351be78f05db768babf6bca35b0c69d8962`.
- Viewer dataset migliorato e verificato su 2.000 immagini.
- Controllo manuale del dataset pubblico trasformato in una coda CVAT da 22
  immagini; la revisione finale conserva 4 box.
- Commit già presenti sul branch:
  `0bacc0f`, `dcca648`, `3b528c8`, `5e93ce2`, `544e168`.

### Dati operativi e screening

- 1.029 immagini interne inventariate senza modificare la sorgente.
- Split per sito-giorno senza leakage: 403 `operational_dev`, 385
  `operational_mining`, 241 `sealed_test`.
- Il test resta sigillato: zero immagini test lette durante embedding e
  selezione.
- Screening completato su 5.675 immagini pubbliche e 788 immagini interne con
  entrambi i modelli, senza fallimenti.

### Esito PhenoCam pubblico

Il pilot pubblico da 200 immagini è stato completato con `0/200` positivi. È
stato esportato e archiviato prima della rimozione da CVAT:

- export COCO revisionato del pilot pubblico per il ciclo 0.1.3;
- backup ripristinabile del task del pilot.

I nomi esatti degli archivi sono conservati nel documento originale.

La causa principale era il ranking: soltanto 7 immagini avevano confidenza 0.1.2
`>=0,30`; 161 erano state selezionate per segnali deboli `crop_only`.

Una seconda analisi con il gate stretto trova 58 candidati con accordo fra i
due modelli; 2 erano già nel pilot negativo e ne restano 56. I dieci più forti
mostrano soprattutto veicoli reali ma molto lontani, spesso alti 7-15 pixel.
Non è stato creato un altro task: prima bisogna decidere esplicitamente se
questi oggetti piccoli ma riconoscibili siano target privacy o negativi
operativi. Dopo la decisione, l'eventuale nuovo pilot dovrebbe contenere solo
20 immagini ad alta confidenza.

### Preannotazioni interne clean

I task interni originali proponevano tutte le detection alla soglia `0,01`,
generando migliaia di falsi box. Sono stati sostituiti con bundle che accettano
una proposta soltanto quando:

- baseline e 0.1.2 rilevano entrambi l'oggetto con confidenza `>=0,30`;
- la classe è identica;
- l'IoU fra le due box è `>=0,50`.

Il filtro è in `dataset/builder/mining/suggestions.py`; l'orchestrazione CVAT
rimane sotto il limite di 200 linee produttive. Sono stati aggiunti il flag
`cvat-bundle --clean-suggestions`, la configurazione del gate e il comando
il comando storico documentato nella fonte originale.

### Integrazione YOLO26x — 1 settembre 2026

Dopo la rimozione manuale delle box di gruppo, i task contenevano 1.134 e 1.115
box individuali. Il comando `teacher-coco` ha eseguito YOLO26x a 1.280 pixel,
confidenza `0,50`, su MPS e ha aggiunto soltanto proposte non sovrapposte a box
della stessa classe con IoU `>=0,50`. Il checkpoint ha SHA-256
`9fdd44a31c504547ffb81d2c6d9e6dac3493c8eaa8b0398d3f43bae6c7003e92`.

- task 9: 1.323 candidati, 483 aggiunti, 1.617 box finali;
- task 10: 1.554 candidati, 655 aggiunti, 1.770 box finali.

Il controllo visuale dei 24 frame con più aggiunte conferma che le proposte sono
principalmente istanze individuali reali. Su un frame diurno rappresentativo il
conteggio visuale è circa 55-60 veicoli contro 37 box dopo la fusione: il teacher
migliora la copertura, ma non sostituisce la revisione umana. Il readback COCO da
CVAT conserva tutti i conteggi e le geometrie; l'arrotondamento massimo introdotto
dal server è inferiore a `0,01` pixel.

Gli export e i backup precedenti all'import hanno suffisso
`pre-yolo26x-20260901T075143Z`; gli export di controllo hanno suffisso
`post-yolo26x-20260901T075143Z`. Tutti sono locali e ignorati da Git.

### Recupero tiled raspberrypi2.local — 1 settembre 2026

Il controllo separato per sito ha mostrato auto mancanti nella scena affollata
`raspberrypi2.local`, dovute alla riduzione del frame 4.608×2.592 a 1.280 pixel.
Un secondo passaggio YOLO26x ha quindi elaborato soltanto le 60 immagini del
sito in ciascun task con la griglia esistente 5×3 e sovrapposizione del 20%.
Confidenza, modello e dimensione sono rimasti invariati.

- task 9: 1.959 candidati validi, 588 box di bordo scartate, 487 istanze
  aggiunte; totale 2.104;
- task 10: 1.950 candidati validi, 623 box di bordo scartate, 453 istanze
  aggiunte; totale 2.223.

La fusione sopprime i duplicati anche fra `car`, `bus` e `truck` e quando
l'intersezione copre almeno metà della box più piccola. Le 120 immagini
`sitets02` sono rimaste invariate. Il controllo visuale dei 24 frame con più
aggiunte mostra prevalentemente veicoli individuali reali. Il readback CVAT
conferma tutte le geometrie con IoU minima `0,99909` e scarto massimo inferiore
a `0,01` pixel.

I backup completi immediatamente precedenti al passaggio tiled hanno suffisso
`post-yolo26x-pre-tiled-20260901T082128Z`; SHA-256 task 9
`844ca7a0b64dd017f3906688973a681519de0e2b3305d8e8e0c14e1aa67375db` e task 10
`2fbdc1e3147aae854220ba2ea6ee2b58e0e6e5ba821f29f5670594b51afb872c`.

### Recupero mirato della zona destra — 1 settembre 2026

Dopo il passaggio tiled restavano auto mancanti soprattutto nel parcheggio a
destra. Un confronto su otto frame ha trovato 73 nuove proposte valide a soglia
`0,30` con i crop esistenti, contro 27 usando crop più piccoli a soglia `0,50`;
sotto `0,30` aumentavano invece i casi dubbi. È stato quindi eseguito un ultimo
passaggio limitato ai sei crop con origine oltre il 55% della larghezza, sempre
soltanto su `raspberrypi2.local`.

- task 9: 317 aggiunte, di cui 308 `car`; totale 2.421;
- task 10: 316 aggiunte, di cui 305 `car`; totale 2.539.

Le 633 aggiunte sono distribuite su 47 e 53 frame. I 24 frame più modificati
sono stati controllati visualmente prima dell'import; le nuove box rappresentano
prevalentemente veicoli individuali reali. `sitets02` e tutte le annotazioni
precedenti sono rimasti invariati. Il readback CVAT ha IoU minima `0,99874` e
scarto massimo inferiore a `0,01` pixel.

I backup completi precedenti a questo passaggio hanno suffisso
`post-tiled-pre-right-20260901T085902Z`; SHA-256 task 9
`e28cc526f309d99c5311a73b80b344da58ee1c6d835800bd12ed517fce3afc61` e task 10
`22b182f97d491a16f6bd80608ef566ca2b74a5802669e286a94b5980c6a74bf5`.

### Audit finale delle annotazioni — 1 settembre 2026

La revisione umana ha portato i task 9 e 10 a 2.915 e 3.075 box. L'audit finale
ha esaminato tutte le sovrapposizioni della famiglia veicoli con IoU `>=0,50` o
copertura della box minore `>=0,80`. Le sovrapposizioni fra istanze distinte
sono state conservate; sono stati rimossi soltanto duplicati dello stesso
oggetto, un box di gruppo residuo e tre box spurie interne a veicoli già
annotati. Due furgoni sono stati riclassificati da `truck` a `car`.

- task 9: 38 rimozioni, 2.877 box finali;
- task 10: 66 rimozioni e 2 riclassificazioni, 3.009 box finali.

Il readback CVAT coincide esattamente con gli export corretti. Non restano
coppie della stessa classe con IoU `>=0,50`, duplicati `car`/`truck` con IoU
`>=0,79` o box che contengono almeno due box individuali della stessa classe.
I backup completi precedenti all'intervento hanno suffisso
`human-final-20260901T133323Z`; SHA-256 task 9
`0fbe3147dfd9579e78fae599cde9993cbbd6a6a2b589ac8c128dc4647a71455e` e task 10
`9e19ee2186b30c030c6737a35bc5948ecb6741e4a7edbb3ea4b87e3adf7f52d3`.
I backup completi dello stato corretto hanno suffisso
`human-final-cleanup-20260901T134246Z`; SHA-256 task 9
`ed8331fddb930e84353aced74728686f1672a296d672c35296dce0a2808b1dd8` e task 10
`5fa5d1f75b8348157dd05c520491e7b359208f604be0d9a3ed6d96cb11c3ad05`.

### Import operativo con provenienza — 1 settembre 2026

I due export corretti sono stati importati sotto
workspace del training 0.1.3 (`reviewed/`). Ogni immagine usa il formato
`sito--timestamp--sha12.jpg`, mentre il manifest conserva nome sorgente,
identita, sito, gruppo, split, coorte, checksum e revisori.

- operational dev: 120 immagini, 114 positive, 6 negative e 2.877 box;
- operational mining: 120 immagini, 117 positive, 3 negative e 3.009 box.

I 240 nomi sono unici e contengono `raspberrypi2.local` o `sitets02`; le copie
coincidono con gli SHA-256 sorgente e gli artifact non contengono path locali.
Il report preannotazione/ground truth registra complessivamente 2.085 box
corrispondenti, 49 rimosse e 3.801 aggiunte. Questi dati restano operativi e non
entrano nel training del primo ciclo 0.1.3.

### Dataset 0.1.3 materializzato — 1 settembre 2026

L'artifact oggi denominato dataset 0.1.3 è apribile con il viewer e contiene 2.240
immagini: le 2.000 pubbliche e invariate sotto `train`, più 120
`operational_dev` e 120 `operational_mining`. Le 240 immagini interne restano
escluse dalla voce `train` del file YOLO. Il manifest conserva origine e nome
sorgente; i filename includono sito, timestamp e hash breve.

La verifica confronta byte-per-byte tutti i 3.244 file pubblici di immagini e
label con la 0.1.2, controlla unicità e presenza di ogni path, valida tutte le
label normalizzate e conferma l'intero manifest SHA-256.

## Task rimossi e recuperabilità

Sono stati eliminati da CVAT i task superati o già completati 2-8. Prima della
rimozione sono stati verificati i backup locali:

- task 7 e 8: export COCO e backup con le vecchie preannotazioni;
- task 5: export COCO delle 200 negative e backup completo;
- task 6: export COCO delle 22 immagini/4 box e backup completo;
- task 2-4: già esportati, importati nel dataset e coperti da backup completi.

Tutti gli archivi si trovano in
`dataset/workspace/annotation/exports/`, che è locale e ignorata da Git. La
lista CVAT riletta dopo le eliminazioni conteneva i task 9 e 10; il task 11
pubblico YOLO26x è stato aggiunto successivamente. Le eliminazioni sono quindi
recuperabili con `cvat-cli task
create-from-backup`.

### Screening pubblico YOLO26x — 1 settembre 2026

YOLO26x ha completato 3.059/3.059 frame PhenoCam pubblici ancora eleggibili,
senza errori. Il gate `>=0,50` ha prodotto 48 immagini e 201 box. L'audit
visuale ha escluso 8 immagini e 9 falsi positivi su camera, ombre e vegetazione;
il task 11 contiene quindi 40 immagini e 192 proposte su 4 siti e 33 gruppi.

Il readback COCO da CVAT coincide per conteggi, classi e geometrie, con scarto
massimo `0,0093` pixel. Le proposte restano da revisionare e non fanno ancora
parte del dataset 0.1.3.

Il report `docs/status/dataset-0.1.3-review.md` misura
composizione, bias e leakage prima di ogni modifica. La pipeline pronta per il
post-review assegna ogni immagine a `included`, `reserved` o `rejected`, limita
il primo ingresso a 18 frame (8 per sito dominante, uno per camera-day), applica
SSCD `0,95` e permette un build affiancato senza sovrascrivere il 0.1.3 corrente.

## Verifiche eseguite

- Suite dataset: 43 test passati.
- Suite runtime: 237 test passati, 7 saltati per fixture opzionali assenti.
- Test specifico del gate clean incluso.
- Sintassi di `dataset/commands/cvat-tasks.sh`: valida.
- `git diff --check`: passato.
- Moduli di orchestrazione verificati sotto 200 linee produttive
  (`cvat.py`: 191; `selection.py`: 197; `teacher.py`: 164;
  `teacher_screening.py`: 134; `teacher_views.py`: 57; `review.py`: 186).

## Prossimi passi

1. Completare il task 11 e correggere tutte le proposte YOLO26x.
2. Esportare e importare il task con revisori attribuiti.
3. Aggiornare composizione pubblica 0.1.3 e README con il rendimento umano reale.
4. Proseguire con training ed evaluation usando le coorti interne separate.

Non è stato eseguito alcun push.
