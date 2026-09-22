# Revisione pre-ampliamento del dataset v3

> Historical record for this iteration; use the [current guide](../development.md#current-workflow).
> Commands refer to the original workflow; ignored artifacts require local evidence.

> Errata corrige del 2 settembre 2026: una box `bus` del frame
> `raspberrypi2.local--2025-11-11T091905--be026e5d2e7d.jpg` racchiudeva un
> edificio. La correzione elimina soltanto quella box e riduce di uno i conteggi
> `bus`, `operational_dev` e totali; le box `truck` restano invariate. Le misure
> pre-correzione restano nel corpo come
> fotografia storica; l'esito materializzato corretto è riportato sotto.

Data della misura: 2026-09-01.

Stato: report pre-modifica. L'artifact oggi denominato `dataset/dataset-v3/` non è stato
modificato e il Task CVAT 11 non è ancora ground truth.

## Esito

Il dataset v3 materiale contiene 2.240 immagini e 10.562 annotazioni. Le 2.000
immagini pubbliche costituiscono l'unico training split; le 240 immagini interne
sono separate in `operational_dev` e `operational_mining` e non sono usate
automaticamente per il training.

L'ampliamento PhenoCam è utile soprattutto per `car`: il training pubblico ha
3.051 istanze `person` ma soltanto 772 `car`, mentre i dati operativi sono
composti per il 94,43% da automobili. Le 40 immagini del Task 11 sono però
concentrate su quattro siti, con 38 immagini provenienti da due sole camere.

La prima espansione consigliata è quindi di **massimo 18 immagini PhenoCam
revisionate**, non di tutte le 40 in modo automatico. Il tetto deriva da un
massimo di 8 immagini per ciascuna camera dominante, una per camera-day, una
immagine per ciascuno degli altri due siti e deduplicazione SSCD interna a
coseno `0,95`. Le altre immagini revisionate restano in una riserva tracciata.

## Confini delle misure

Le sezioni usano quattro qualificatori:

- **misurato**: contato direttamente da immagini, label, manifest o artefatti
  correnti;
- **predizione YOLO26x**: proposta del modello, mai ground truth;
- **stima**: proiezione condizionata alla correttezza delle predizioni;
- **raccomandazione**: decisione progettuale ancora da applicare.

Fonti locali principali:

- `dataset/dataset-v3/metadata/source-images.csv`;
- `dataset/dataset-v3/labels/**`;
- `dataset/dataset-v3/metadata/source-annotations.jsonl`;
- `dataset/workspace/sources/phenocam/baseline-screened.csv`;
- `dataset/workspace/training-v3/screening/public-teacher/`;
- `dataset/workspace/training-v3/selection/public-teacher-clean.csv`;
- `output/training-v2/split-audit.json` e relativi elenchi train/validation;
- `dataset/config/dataset-contract.json` e
  `dataset/config/training-v3.json`.

Le percentuali di immagini per classe sono prevalenze e possono sommare oltre
il 100%, perché una stessa immagine può contenere classi diverse. Le percentuali
di annotazioni sommano invece al 100% per ogni split.

## 1. Composizione corrente misurata

| Split materiale | Immagini | Positive | Negative | Annotazioni |
| --- | ---: | ---: | ---: | ---: |
| `train` | 2.000 | 1.244 | 756 | 4.677 |
| `operational_dev` | 120 | 114 | 6 | 2.877 |
| `operational_mining` | 120 | 117 | 3 | 3.009 |
| **Totale** | **2.240** | **1.475** | **765** | **10.562** |

Per provenienza, le immagini sono 1.279 Open Images V7, 721 PhenoCam v3 e 240
interne. Tutte le immagini Open Images e PhenoCam correnti sono nel training
pubblico. Le immagini interne sono 60 per sito in ciascuno split operativo.

### Distribuzione complessiva

| Classe | Immagini con classe | % immagini | Annotazioni | % annotazioni |
| --- | ---: | ---: | ---: | ---: |
| `person` | 808 | 36,07% | 3.231 | 30,59% |
| `bicycle` | 125 | 5,58% | 200 | 1,89% |
| `car` | 665 | 29,69% | 6.330 | 59,93% |
| `motorcycle` | 144 | 6,43% | 241 | 2,28% |
| `bus` | 126 | 5,63% | 180 | 1,70% |
| `truck` | 317 | 14,15% | 380 | 3,60% |

Il totale aggregato è dominato da `car`, ma ciò dipende dagli split operativi e
non descrive la distribuzione effettivamente usata per addestrare il modello.

### Split pubblico `train`

| Classe | Immagini con classe | % immagini | Annotazioni | % annotazioni |
| --- | ---: | ---: | ---: | ---: |
| `person` | 729 | 36,45% | 3.051 | 65,23% |
| `bicycle` | 125 | 6,25% | 200 | 4,28% |
| `car` | 434 | 21,70% | 772 | 16,51% |
| `motorcycle` | 113 | 5,65% | 204 | 4,36% |
| `bus` | 126 | 6,30% | 180 | 3,85% |
| `truck` | 226 | 11,30% | 270 | 5,77% |

Tutti i minimi di istanza del contratto v2 sono rispettati. `car`, `truck`,
`bicycle`, `motorcycle` e `bus` sono però vicini ai rispettivi minimi, mentre
`person` li supera ampiamente. Il deficit più rilevante rispetto al dominio
operativo è quindi la copertura delle automobili in immagini a camera fissa.

### Split `operational_dev`

| Classe | Immagini con classe | % immagini | Annotazioni | % annotazioni |
| --- | ---: | ---: | ---: | ---: |
| `person` | 37 | 30,83% | 98 | 3,41% |
| `bicycle` | 0 | 0,00% | 0 | 0,00% |
| `car` | 114 | 95,00% | 2.706 | 94,06% |
| `motorcycle` | 19 | 15,83% | 25 | 0,87% |
| `bus` | 0 | 0,00% | 0 | 0,00% |
| `truck` | 40 | 33,33% | 47 | 1,63% |

### Split `operational_mining`

| Classe | Immagini con classe | % immagini | Annotazioni | % annotazioni |
| --- | ---: | ---: | ---: | ---: |
| `person` | 42 | 35,00% | 82 | 2,73% |
| `bicycle` | 0 | 0,00% | 0 | 0,00% |
| `car` | 117 | 97,50% | 2.852 | 94,78% |
| `motorcycle` | 12 | 10,00% | 12 | 0,40% |
| `bus` | 0 | 0,00% | 0 | 0,00% |
| `truck` | 51 | 42,50% | 63 | 2,09% |

L'assenza di `bicycle` negli split operativi è coerente con la regola v3 che
non annota una bicicletta senza persona; non implica che la classe possa essere
rimossa dall'ontologia pubblica a sei classi.

## 2. Split di training e leakage

Il dataset materiale non contiene directory `val` o `test` e il suo YAML
dichiara soltanto `train`, `operational_dev` e `operational_mining`. Il notebook
`notebooks/training-v2.ipynb` costruisce però un train/validation temporaneo e deterministico
raggruppando per `group_id`.

### Split derivato usato dal training v2

| Split derivato | Immagini | Open Images | PhenoCam | Positive | Negative | Annotazioni |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| train | 1.600 | 1.018 | 582 | 990 | 610 | 3.733 |
| validation | 400 | 261 | 139 | 254 | 146 | 944 |

| Classe | Train annotazioni | % train | Validation annotazioni | % validation |
| --- | ---: | ---: | ---: | ---: |
| `person` | 2.462 | 65,95% | 589 | 62,39% |
| `bicycle` | 166 | 4,45% | 34 | 3,60% |
| `car` | 612 | 16,39% | 160 | 16,95% |
| `motorcycle` | 138 | 3,70% | 66 | 6,99% |
| `bus` | 143 | 3,83% | 37 | 3,92% |
| `truck` | 212 | 5,68% | 58 | 6,14% |

La composizione è complessivamente coerente, anche se `motorcycle` pesa quasi
il doppio nella validation. Non esistono hash file o pixel identici fra i due
split e nessun `group_id` attraversa il confine.

Il confine camera/sito non è però preservato: 83 siti PhenoCam compaiono sia in
train sia in validation e si misurano 139 coppie cross-split con distanza pHash
non superiore a 6. La divisione per camera-day impedisce la copia della stessa
sequenza giornaliera, ma non impedisce che lo sfondo fisso della stessa camera
sia presente da entrambi i lati. Questa validation misura quindi la
generalizzazione temporale più della generalizzazione a camere nuove.

Anche `operational_dev` e `operational_mining` condividono i due siti interni.
Non hanno duplicati esatti né `group_id` condivisi, ma contengono 222 coppie
cross-split con distanza pHash non superiore a 6. Devono restare strumenti di
development e mining, non essere presentati come test indipendenti.

Il test operativo sigillato non è incluso nell'artifact e non è stato letto da
questa analisi, coerentemente con `test_status: sealed`.

## 3. Anomalie e duplicati correnti

Controlli misurati sulle 2.240 righe e su tutte le label:

- zero duplicati di `image_id`, `source_identity`, path, SHA-256 sorgente,
  SHA-256 pixel decodificati o SHA-256 compilato;
- zero label mancanti, vuote, malformate, fuori intervallo o con box identiche
  duplicate;
- zero `group_id` condivisi fra gli split materiali;
- 16 valori pHash identici interessano 50 immagini; cinque valori attraversano
  `operational_dev` e `operational_mining`;
- molti pHash identici nel pubblico appartengono a immagini PhenoCam molto buie
  e uniformi: sono anomalie percettive, non copie esatte.

Il pHash non deve essere usato da solo per cancellare dati: sulle camere fisse
lo sfondo domina l'hash anche quando gli oggetti cambiano. La regola corretta
resta pHash come segnale economico, SSCD come conferma e revisione umana per i
casi prossimi alla soglia.

## 4. PhenoCam già presente

### Copertura misurata

- 721 immagini, tutte in `train`;
- 157 siti/camere e 654 gruppi camera-day;
- periodo dal 2001-09-01 al 2023-12-30;
- massimo 9 immagini per sito; nessun singolo sito domina il sottoinsieme;
- 706 negative e 15 positive;
- 59 annotazioni: 47 `car`, 7 `person`, 5 `truck`;
- nessuna annotazione `bicycle`, `motorcycle` o `bus`.

| Stagione sorgente | Immagini | Percentuale |
| --- | ---: | ---: |
| inverno | 197 | 27,32% |
| primavera | 186 | 25,80% |
| estate | 161 | 22,33% |
| autunno | 177 | 24,55% |

I principali codici grezzi `primary_veg_type` sono `DB` 207, `GR` 135, `AG`
106, `EN` 97, `SH` 71 e `TN` 53. Il repository conserva i codici PhenoCam ma
non ne versiona una legenda normativa; il report non ne assume quindi
l'espansione semantica.

La copertura oraria è 36 immagini tra 00-05, 215 tra 06-10, 264 tra 11-14, 152
tra 15-18 e 54 tra 19-23. Gli orari sono quelli registrati dalla camera e non
sono stati convertiti in un fuso comune.

Come misura riproducibile di illuminazione, la luminanza media di thumbnail in
scala di grigi classifica 70 immagini sotto 40/255, 43 fra 40 e 79, 605 fra 80
e 179 e 3 almeno a 180. Le soglie sono un'euristica descrittiva, non annotazioni
di giorno/notte o qualità.

La distribuzione stagionale e per sito è buona, ma il sottoinsieme è quasi
interamente negativo. Meteo, neve, foschia e controluce non sono etichette
versionate nel manifest; non possono essere quantificati come ground truth dal
dataset corrente.

## 5. Screening YOLO26x delle nuove immagini

### Pool misurato

Lo screening ha elaborato 3.059 immagini eleggibili, con zero errori, dopo aver
escluso identità, hash e camera-day già usati nel v3 pubblico o nel pilot
precedente. Il pool copre 141 siti, 2.077 camera-day e il periodo 2001-2023.

| Stagione | Immagini eleggibili |
| --- | ---: |
| inverno | 672 |
| primavera | 894 |
| estate | 821 |
| autunno | 672 |

La luminanza euristica comprende 268 immagini molto scure, 143 debolmente
illuminate, 2.637 intermedie e 11 molto luminose. Il pool è quindi ampio e
stagionalmente diversificato, anche se include fotogrammi quasi neri.

### Predizioni YOLO26x, non ground truth

Modello: `yolo26x.pt`, SHA-256
`9fdd44a31c504547ffb81d2c6d9e6dac3493c8eaa8b0398d3f43bae6c7003e92`,
input 1280, dispositivo MPS, soglia di acquisizione 0,30.

| Soglia | Immagini positive | Box | `person` | `car` | `motorcycle` | `bus` | `truck` | Siti |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0,30 | 76 | 286 | 49 | 209 | 1 | 1 | 26 | 12 |
| 0,40 | 63 | 238 | 24 | 193 | 0 | 1 | 20 | 9 |
| 0,50 | 48 | 201 | 8 | 178 | 0 | 1 | 14 | 8 |
| 0,60 | 37 | 167 | 0 | 158 | 0 | 1 | 8 | 3 |
| 0,70 | 32 | 125 | 0 | 120 | 0 | 0 | 5 | 3 |
| 0,80 | 23 | 52 | 0 | 50 | 0 | 0 | 2 | 3 |

La soglia 0,50 è un buon gate di annotazione: riduce molto le proposte `person`
che, nell'audit visivo, erano spesso arbusti, ombre o parti della camera. Non
offre però un modo credibile per rafforzare `bicycle` o `motorcycle`.

### Task CVAT 11

I conteggi di questa sottosezione sono la fotografia pre-review richiesta dal
report. L'esito umano successivo è documentato nella sezione 10.

L'audit visivo pre-CVAT ha eliminato 8 immagini con 9 falsi positivi espliciti.
Il bundle finale contiene 40 immagini e 192 proposte a confidenza almeno 0,50:

- 178 `car`;
- 12 `truck`;
- 1 `bus`;
- 1 `person`.

Le immagini provengono da `nationalcapital` 24, `bitterootvalley` 14,
`borgocioffinorth` 1 e `snodgrass5` 1. Sono 39 estive e una invernale; tutte
hanno luminanza euristica intermedia. Il task è ancora in stato `annotation`,
con zero job completati: questi numeri restano predizioni.

Rispetto al dataset corrente, il massimo coseno SSCD delle 40 immagini è 0,7915,
ben sotto il gate 0,95: sono nuove rispetto al v3. All'interno delle 40, però,
esistono 5 coppie SSCD almeno 0,95 e 99 coppie dello stesso sito con distanza
pHash non superiore a 8. La ridondanza interna è quindi il limite principale.

## 6. Proposta quantitativa di ampliamento

### Raccomandazione

Mantenere tutte le 2.240 immagini correnti e aggiungere al primo ciclo **fino a
18 immagini PhenoCam umanamente revisionate** nel training pubblico.

Criteri, nell'ordine:

1. ammettere solo immagini del Task 11 completate e importate come ground truth;
2. escludere immagini negative, ambigue o con annotazione incompleta;
3. conservare al massimo una immagine per camera-day;
4. conservare al massimo 8 immagini per `nationalcapital` e 8 per
   `bitterootvalley`, più le due immagini uniche degli altri siti se valide;
5. fra coppie SSCD con coseno almeno 0,95, mantenere quella con più target
   verificati, poi quella che aggiunge una classe rara, poi la confidenza YOLO
   più alta solo come spareggio;
6. rieseguire deduplicazione contro tutto il v3 con SHA-256, pHash e SSCD;
7. mantenere le immagini scartate in una riserva revisionata con motivazione,
   senza cancellarle.

Applicando questi vincoli alle sole predizioni disponibili si ottengono 18
immagini, 18 camera-day e 77 box stimate: 67 `car`, 8 `truck`, 1 `bus` e 1
`person`. La composizione reale potrà soltanto essere stabilita dopo la
revisione CVAT.

### Composizione finale prevista

| Voce | Corrente misurato | Dopo il primo ciclo | Natura del dato |
| --- | ---: | ---: | --- |
| immagini totali | 2.240 | fino a 2.258 | raccomandazione |
| immagini training | 2.000 | fino a 2.018 | raccomandazione |
| immagini operative separate | 240 | 240 | invariato |
| immagini PhenoCam | 721 | fino a 739 | raccomandazione |
| annotazioni totali | 10.562 | circa 10.639 | stima YOLO26x |
| annotazioni training | 4.677 | circa 4.754 | stima YOLO26x |

Se tutte le 77 proposte della selezione conservativa fossero confermate, il
training arriverebbe a circa 3.052 `person`, 200 `bicycle`, 839 `car`, 204
`motorcycle`, 181 `bus` e 278 `truck`. `car` crescerebbe del 8,68% senza
avvicinarsi alla dominanza osservata nel dominio operativo. Le classi
`bicycle`, `motorcycle` e `bus` resterebbero sottorappresentate e richiederebbero
un mining separato, non un abbassamento indiscriminato della soglia.

## 7. Strategia di split consigliata

Le nuove immagini appartengono a siti PhenoCam già presenti nel pubblico v2.
Per il primo ciclo devono restare nello stesso lato di training delle altre
immagini di quei siti; non devono essere distribuite casualmente fra train e
validation.

Per un futuro training v3 riproducibile:

- Open Images: mantenere indivisibili i gruppi di provenienza;
- PhenoCam: assegnare l'intero `site_id` a un solo split, non il singolo giorno;
- se i siti disponibili non bastano, usare blocchi temporali contigui per sito e
  rimuovere dal confine le coppie SSCD almeno 0,95;
- non usare `operational_dev` o `operational_mining` come test indipendente;
- lasciare sigillato il test operativo finché non è dichiarato un protocollo di
  valutazione e non sono disponibili abbastanza siti indipendenti.

Il cambio da split per camera-day a split per sito modifica il protocollo di
benchmark e deve essere versionato esplicitamente; non va applicato retroattivamente
al risultato v2 senza conservare il benchmark storico.

## 8. Coerenza con la documentazione

Elementi coerenti:

- formato YOLO `class_id x_center y_center width height`, coordinate normalizzate;
- ID COCO conservati: 0, 1, 2, 3, 5 e 7, con placeholder inutilizzati 4 e 6;
- negativi senza label file;
- box singole per oggetto e divieto delle box di gruppo;
- provenienza completa nel manifest e nomi interni con sito, timestamp e hash;
- immagini operative escluse automaticamente dal training;
- predizioni YOLO dichiarate come suggerimenti e non ground truth.

Incongruenze o limiti da correggere dopo la revisione:

- il README v3 riporta correttamente 2.240 immagini, ma le statistiche
  materializzate espongono le classi solo per gli split operativi e non il
  totale pubblico;
- il YAML materiale non ha `val` o `test`; il vero split di training vive in
  `output/training-v2/` ed è quindi esterno all'artifact;
- lo split v2 rispetta `group_id` ma non il sito PhenoCam, creando leakage di
  sfondo fisso fra train e validation;
- `training-v3.json` elenca cinque classi privacy e omette `bicycle`, mentre il
  dataset pubblico mantiene l'ontologia a sei classi; il confine è intenzionale
  per i task operativi ma deve essere esplicitato nel futuro README ampliato;
- `sealed_test` è un confine del workflow, non uno split distribuito nel v3;
- il selettore YOLO26x verifica la novità SSCD rispetto al dataset esistente ma
  non impone diversità SSCD tra le immagini dello stesso task.

## 9. Gate prima della modifica

1. Completare il Task CVAT 11 correggendo o eliminando tutte le proposte e
   aggiungendo ogni target reale mancante.
2. Esportare COCO e backup completo; verificare conteggi, classi e geometria.
3. Importare l'export in un artifact reviewed immutabile con task, revisori e
   checksum.
4. Applicare i criteri conservativi alle annotazioni umane e produrre un CSV
   `included/reserved/rejected` con motivazione.
5. Materializzare il v3 ampliato in una nuova destinazione o tramite identità di
   build nuova, senza sovrascrivere l'artifact corrente.
6. Rieseguire checksum, validazione label, audit leakage e statistiche complete.
7. Solo allora aggiornare `dataset/docs/dataset-v3/README.md`, README
   materializzato e stato del
   progetto con i numeri misurati finali.

## 10. Esito post-review e ampliamento

Il Task CVAT 11 è stato esportato e importato il 2026-09-01. Il job 36 risulta
`stage=annotation`, `state=completed`: `annotation` è la fase del job, mentre
`completed` è il suo stato. L'export revisionato contiene 206 box contro le 192
proposte YOLO26x: 14 box sono state aggiunte e tutte le proposte sono state
confermate. La ground truth comprende 188 `car`, 16 `truck`, 1 `bus` e 1
`person` su 40 immagini positive.

La policy dichiarata in questo report ha ammesso 18 immagini: 8
`bitterootvalley`, 8 `nationalcapital`, 1 `borgocioffinorth` e 1 `snodgrass5`.
Contengono 104 box umane (`car` 90, `truck` 12, `bus` 1, `person` 1). Le altre
22 immagini restano in riserva con motivazione; nessuna è stata cancellata o
classificata come rifiutata.

La composizione materializzata corretta è quindi 2.258 immagini e 10.666
annotazioni: 2.018 immagini e 4.781 annotazioni nel training pubblico, più i
240 frame e 5.885 annotazioni operative già separati. La provenienza PhenoCam
sale da 721 a 739 immagini. Questi sono conteggi ground truth, non stime del
teacher.

| Classe | Immagini che la contengono | % delle 2.258 immagini | Box | % dei 10.666 box |
| --- | ---: | ---: | ---: | ---: |
| `person` | 809 | 35,828% | 3.232 | 30,302% |
| `bicycle` | 125 | 5,536% | 200 | 1,875% |
| `car` | 680 | 30,115% | 6.420 | 60,191% |
| `motorcycle` | 144 | 6,377% | 241 | 2,260% |
| `bus` | 127 | 5,624% | 181 | 1,697% |
| `truck` | 326 | 14,438% | 392 | 3,675% |

Le percentuali per immagine non sommano a 100% perché un frame può contenere
più classi. Per split, le annotazioni sono 4.781 nel training (44,825%), 2.876
in `operational_dev` (26,964%) e 3.009 in `operational_mining` (28,211%). Il
training conserva 3.052 `person`, 200 `bicycle`, 862 `car`, 204 `motorcycle`,
181 `bus` e 282 `truck`; l'espansione rafforza `car` e `truck`, mentre
`bicycle`, `motorcycle` e `bus` restano le classi meno rappresentate.

L'audit finale non trova duplicati esatti di immagine, identità o path, né
`group_id` presenti in più split. Tutti i 3.244 file immagine/label ereditati
dal training v2 sono byte-identici. L'artifact non introduce split `val` o
`test`: le nuove immagini restano nel training insieme agli altri frame dei
medesimi siti, mentre gli split interni rimangono separati e non addestrabili.
I checksum completi e i 43 test dataset risultano validi.
