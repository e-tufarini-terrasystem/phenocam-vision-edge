<!--
Questo documento definisce il piano tecnico per costruire e validare un dataset
PhenoCam/Agrocam orientato al rilevamento preventivo di persone e veicoli.
-->

# Piano tecnico per un dataset PhenoCam/Agrocam orientato alla tutela della privacy

> Stato: piano aggiornato sulle informazioni disponibili, senza modificare codice
> e senza avviare addestramenti.

## Sintesi esecutiva

Il sistema deve soddisfare tre obiettivi distinti:

- individuare almeno il 95% delle immagini che contengono una o più persone;
- individuare almeno il 95% delle immagini che contengono uno o più veicoli
  rilevanti;
- scartare per errore non più del 5% delle immagini che non contengono né
  persone né veicoli rilevanti.

Con poche centinaia di immagini provenienti da circa tre siti, queste percentuali
potranno essere soltanto stime iniziali. Il risultato dovrà essere presentato
come una prova pilota sulle tre camere note, non come garanzia di funzionamento
su nuove camere.

Il campione presente nel repository contiene solo 16 immagini, quasi tutte dello
stesso sito. Tutte contengono almeno una persona o un veicolo e nessuna dispone
di annotazioni manuali di riferimento. Una verifica visiva provvisoria, usando
la soglia attuale di 0,30, indica:

- persone: rilevate tutte le 4 persone visibili, ma una è stata segnalata due volte;
- veicoli: rilevati circa 15 dei 18 veicoli visibili;
- il filtro si sarebbe attivato in 13 delle 15 immagini contenenti veicoli;
- due veicoli tagliati dal bordo o coperti dalla vegetazione non sono stati
  rilevati;
- non è possibile misurare quante immagini verrebbero scartate per errore,
  perché nel campione non ci sono immagini negative.

L'obiettivo del 95% va verificato dopo un ulteriore addestramento mirato, usando
dati esterni compatibili e il sistema di ritagli già esistente. Non è realistico
garantire subito il 95% anche per soggetti quasi invisibili, estremamente piccoli
o coperti in gran parte. Per questi casi difficili si propone un obiettivo
diagnostico iniziale del 90%, mantenendo il 95% sull'insieme completo.

Poiché in produzione verrà scartata l'intera immagine, la misura principale è la
percentuale di immagini sensibili che attivano il filtro. Tra le immagini prive
di persone e veicoli, quelle scartate per errore non devono superare il 5%. Si
può tollerare temporaneamente un valore fino al 10% soltanto durante una prova
reversibile, nella quale gli originali siano conservati per il controllo e la
cancellazione definitiva sia disabilitata. La sfocatura rimane opzionale.

### Decisioni già acquisite

- archivio atteso: alcune centinaia di immagini;
- copertura: circa tre siti, una camera per sito, da alcuni mesi a circa un anno;
- titolarità da formalizzare tra azienda e università;
- prodotto commerciale con intenzione di pubblicare il codice come open source;
- esecuzione sul dispositivo entro i limiti già descritti nel repository;
- addestramento preferibilmente su MacBook Air M4 con 24 GB di memoria;
- annotatore e approvatore principale: il responsabile del progetto, con un
  possibile secondo revisore.

---

## 1. Stato attuale del modello, delle classi e del flusso di elaborazione

### Modello

| Proprietà | Stato rilevato |
|---|---|
| Architettura | Ultralytics YOLO26n, rilevamento di oggetti |
| Pesi | `yolo26n.pt`, circa 5,3 MB |
| Modello usato sul dispositivo | `yolo26n.onnx`, FP32, circa 9,5 MB |
| Ingresso | `1×3×640×640`, RGB |
| Uscita | massimo 300 righe `x1,y1,x2,y2,confidence,class_id` |
| Classi | 80 classi COCO |
| Parametri/FLOPs dichiarati | 2,4 M parametri, 5,4 GFLOPs per vista |
| Accuratezza generale dichiarata | COCO mAP50–95 sull'intero processo: 40,1 |
| Programma di esecuzione | ONNX Runtime, CPU |
| Hardware previsto | Raspberry Pi 3, 1 GB RAM, quattro Cortex-A53 |

I valori pubblici del modello sono riportati nella
[documentazione ufficiale YOLO26](https://docs.ultralytics.com/models/yolo26).
La misura ufficiale di 38,9 ms su CPU riguarda una singola immagine su hardware
di riferimento, non il Raspberry Pi e non le sedici viste usate dal progetto.

La licenza indicata nel modello ONNX è AGPL-3.0. Ultralytics dichiara che, quando
si usano il suo codice, i suoi modelli o modelli ulteriormente addestrati,
l'intero progetto deve essere distribuito nel rispetto di AGPL-3.0. In
alternativa occorre una licenza Enterprise. Questa decisione va presa prima
dell'addestramento, non dopo:
[licensing Ultralytics](https://www.ultralytics.com/license).

### Esito della verifica AGPL

Commerciale e open source non sono in contraddizione: il software libero può
essere venduto e usato commercialmente. Il progetto GNU lo chiarisce nelle
pagine [Selling Free Software](https://www.gnu.org/philosophy/selling.en.html)
e [Categories of Free and Nonfree Software](https://www.gnu.org/philosophy/categories.en.html).
Il punto decisivo non è quindi il fatturato, ma la disponibilità a rispettare
integralmente il copyleft.

Per questo progetto la strada più semplice è:

1. attribuire e documentare la titolarità dei contributi di azienda e università;
2. applicare al repository e all'applicazione completa una licenza compatibile
   con AGPL-3.0, includendo testo della licenza e avvisi di copyright;
3. pubblicare il sorgente completo necessario a costruire e usare il prodotto,
   incluse modifiche, configurazioni, script di addestramento ed esportazione e,
   secondo la posizione pubblicata da Ultralytics, i pesi ulteriormente
   addestrati che vengono distribuiti;
4. includere nel pacchetto Raspberry Pi la licenza, gli avvisi e un collegamento
   stabile al sorgente corrispondente;
5. offrire accesso al sorgente anche agli utenti che interagiscono con una
   versione modificata attraverso una rete, come richiede la sezione 13
   dell'[AGPL-3.0](https://github.com/ultralytics/ultralytics/blob/main/LICENSE).

Il repository contiene ora un file `LICENSE` con Apache License 2.0. Questa
licenza chiarisce i diritti sul codice rilasciato dagli autori, ma non annulla
gli obblighi AGPL che possono derivare dalla distribuzione del modello
Ultralytics. Prima del rilascio occorre quindi stabilire, con una verifica
legale, se distribuire l'intero prodotto nel rispetto di AGPL-3.0, ottenere una
licenza Enterprise oppure sostituire il componente. La
[documentazione GitHub](https://docs.github.com/en/repositories/managing-your-repositorys-settings-and-features/customizing-your-repository/licensing-a-repository)
spiega il ruolo della licenza del repository. Inoltre, l'elenco tassativo dei
file inclusi nel pacchetto, documentato nello storico piano locale
`plans/02-github-release/m1.md`,
non comprende licenza e avvisi: andrà corretto se verrà approvata la strada
AGPL.

La licenza del codice non autorizza automaticamente la pubblicazione delle
immagini grezze: provenienza, privacy e licenza del dataset restano valutazioni
separate. Se una parte dell'applicazione, integrazioni cliente o modifiche
dovessero rimanere proprietarie, occorrerà valutare una licenza Enterprise o
un'alternativa tecnologica. Prima del rilascio commerciale è opportuna una
verifica legale congiunta azienda–università; questa analisi non è consulenza
legale.

### Classi operative correnti

Sono abilitate:

- `person`;
- `bicycle`;
- `car`;
- `motorcycle`;
- `bus`;
- `truck`.

La configurazione è in `phenocam/classes/configuration.py`.

Non esistono classi COCO dedicate a:

- furgone;
- pick-up;
- trattore;
- mietitrebbia;
- altri macchinari agricoli.

Questo produce ambiguità semantica: nel campione locale un trattore viene
riconosciuto come `truck`, mentre un furgone molto occluso viene riconosciuto
come `car`.

### Flusso di elaborazione

Il programma:

1. normalizza l'orientamento EXIF e converte in RGB;
2. crea una vista completa e 15 ritagli sovrapposti;
3. usa griglia 5×3 per immagini orizzontali o 3×5 per verticali;
4. ridimensiona ogni vista a 640×640 senza deformarla, aggiungendo margini dove
   necessario;
5. esegue il modello sulle 16 viste in sequenza;
6. riporta i riquadri nel sistema di coordinate dell'immagine originale;
7. elimina i risultati con punteggio di confidenza inferiore a 0,30;
8. elimina i duplicati quando i riquadri si sovrappongono abbastanza;
9. considera `car`, `bus` e `truck` equivalenti quando elimina i duplicati;
10. seleziona le classi abilitate e produce annotazione, sfocatura o eliminazione
    condizionale.

Riferimenti nel repository:

- `phenocam/inference/views.py`;
- `phenocam/inference/pipeline.py`;
- `phenocam/inference/detections.py`;
- `docs/development.md`.

Il programma di esecuzione rifiuta modelli che non rispettano l'attuale formato
di ingresso e uscita e l'elenco esatto delle 80 classi COCO. Un futuro modello a
due classi richiederebbe quindi una modifica separata del programma, soggetta
alla fase di approvazione strutturale prevista da `AGENTS.md`.

### Compatibilità hardware

Un ulteriore addestramento di YOLO26n non cambia significativamente il costo, la
memoria o il tempo richiesti per analizzare un'immagine. È quindi la prima
opzione da provare.

Un passaggio a YOLO26s aumenterebbe i parametri da 2,4 M a 9,5 M e i FLOPs per
vista da 5,4 a 20,7 miliardi. Potrebbe migliorare il tasso di rilevamento, ma
deve essere provato sul Raspberry Pi reale. Il repository stesso dichiara ancora non
verificato il limite di 15 secondi per il comando completo.

Il MacBook Air M4 con 24 GB è adeguato per il pilota YOLO26n. Ultralytics
supporta ufficialmente l'addestramento Apple Silicon con `device=mps`:
[modalità di addestramento](https://docs.ultralytics.com/modes/train). Si
raccomandano immagini 640×640, gruppi automatici oppure inizialmente di 8
immagini, interruzione anticipata quando i risultati non migliorano e tre
inizializzazioni casuali per il candidato finale. YOLO26s va provato solo se
YOLO26n non raggiunge i requisiti. Il MacBook Air è privo di ventola e carichi
prolungati possono ridurre le
prestazioni, ma non la correttezza dell'esperimento.

---

## 2. Inventario e valutazione delle immagini disponibili

### Inventario locale

| Elemento | Quantità e copertura |
|---|---|
| Immagini originali | 16 JPEG |
| Siti/camere | 15 immagini `TS_326`, 1 immagine `TS_106` |
| Periodo | dicembre 2025 – luglio 2026 |
| Risoluzione | 14 immagini 800×601; 2 immagini 3280×2464 |
| Orari | 10:00–15:00 |
| Condizioni | prevalentemente giorno, asciutto, luce ordinaria |
| Soggetti rilevanti | tutte le immagini contengono almeno una persona o un veicolo |
| Negativi | 0 |
| Annotazioni manuali di riferimento | nessuna |
| Risultati esistenti | 16 immagini con riquadri e 16 con sfocatura |
| Immagini storiche attese dai test | 6 immagini 4608×2592, attualmente assenti |

Il sito `TS_326` mostra una strada dietro vegetazione e arbusti. `TS_106` mostra
un ambiente agricolo con una persona. Non sono rappresentati adeguatamente:

- parcheggi;
- siti differenti;
- alba e tramonto;
- pioggia, nebbia o controluce;
- notte;
- folla o più persone vicine;
- biciclette e motocicli;
- immagini senza persone o veicoli rilevanti;
- macchinari agricoli diversi dal singolo trattore;
- forte varietà stagionale per più telecamere.

### Audit visivo provvisorio

Alla soglia corrente di 0,30:

- due veicoli parzialmente fuori campo o coperti dalla vegetazione non vengono
  rilevati;
- un terzo veicolo parzialmente visibile resta sotto soglia a circa 0,256;
- un grande furgone coperto in parte è rilevato con un punteggio di confidenza
  appena superiore alla soglia: 0,343;
- una persona e un'automobile ricevono due riquadri duplicati in due immagini
  diverse;
- alcuni veicoli sono classificati con sottotipo errato, pur restando
  riconoscibili come veicoli.

L'analisi dei punteggi di confidenza mostra il compromesso:

| Soglia indicativa | Effetto sul campione |
|---|---|
| 0,30 | tre veicoli non rilevati |
| 0,25 | recupera il veicolo parzialmente visibile a 0,256 |
| 0,20 | recupera anche un'auto occlusa a 0,238, ma introduce una falsa `person` a 0,222 |
| circa 0,04 | può recuperare l'ultimo veicolo tagliato dal bordo, ma segnala per errore numerosi cartelli, testi sovrapposti e strutture |

La semplice riduzione della soglia per tutte le classi non è quindi una
soluzione sufficiente. Servono soglie distinte per persone e veicoli, immagini
negative difficili e rappresentative e una misurazione completa del rapporto
tra soglia, tasso di rilevamento e falsi allarmi.

---

## 3. Principali cause probabili degli errori

1. **Differenza tra dati di origine e immagini reali:** il modello è stato
   addestrato su COCO, mentre le immagini operative provengono da telecamere
   statiche e mostrano vegetazione, campi e strade lontane.
2. **Soggetti coperti o tagliati:** siepi, rami e bordi dell'immagine nascondono
   porzioni rilevanti dei veicoli.
3. **Oggetti piccoli o distanti:** anche dividendo l'immagine in ritagli, alcuni
   soggetti contengono troppo pochi dettagli per essere riconosciuti facilmente.
4. **Ontologia non adatta:** trattori, macchinari e furgoni non corrispondono
   bene alle sei classi abilitate.
5. **Un'unica soglia per tutte le classi:** i punteggi di persone e veicoli non
   sono necessariamente confrontabili allo stesso modo.
6. **Unione dei risultati delle 16 viste:** i ritagli aumentano il tasso di
   rilevamento, ma anche i duplicati e il rischio di eliminare per errore uno di
   due oggetti vicini.
7. **Immagini negative difficili ricorrenti:** cartelli triangolari, pali, data
   e ora sovrapposte all'immagine, veicoli stampati e strutture verticali
   generano rilevazioni con un basso punteggio di confidenza.
8. **Campione locale non rappresentativo:** quasi tutte le immagini provengono
   dallo stesso sfondo e non comprendono negativi.
9. **Assenza di annotazioni manuali di riferimento:** i risultati correnti
   verificano il funzionamento tecnico, non l'accuratezza.

---

## 4. Dataset potenzialmente utilizzabili

Verifica aggiornata al 26 agosto 2026. Per i dataset è più corretto parlare di
“dati con licenza aperta” che di “open source”. Un repository pubblico o una
licenza aperta sugli strumenti non concedono automaticamente diritti sulle
fotografie e sulle annotazioni.

| Dataset | Copertura utile | Licenza verificata | Uso commerciale | Decisione |
|---|---|---|---|---|
| Immagini PhenoCam/Agrocam interne | Ambiente operativo reale | Dati non pubblici; proprietà, informativa e base giuridica ancora da documentare | Possibile solo se azienda e università dispongono dei diritti necessari | **Condizionato:** indispensabili, ma non sono dati aperti |
| [PhenoCam Network v3](https://zenodo.org/records/14854980/files/Fair-use_Statement.pdf?download=1) | 738 siti, immagini agricole e naturali; nessun riquadro annotato | L'archivio, comprese immagini e dati, è pubblicato con CC BY 4.0 | Sì, con attribuzione, citazione richiesta dalla politica del progetto e verifica di privacy e diritti della personalità | **Approvato** |
| [Open Images V7](https://storage.googleapis.com/openimages/web/factsfigures_v7.html#licenses) | Circa 9 milioni di immagini; persone e veicoli già annotati | Annotazioni CC BY 4.0; immagini indicate come CC BY 2.0. Google non garantisce però la licenza effettiva di ogni fotografia e richiede di verificarla singolarmente | Sì soltanto per le immagini la cui licenza CC BY è confermata alla fonte e registrata con autore e attribuzione | **Condizionato immagine per immagine** |
| [COCO 2017](https://github.com/cocodataset/cocodataset.github.io/blob/master/dataset/termsofuse.htm) | Persone e classi COCO di veicoli | Annotazioni CC BY 4.0; COCO non possiede le fotografie, che restano soggette alla licenza Flickr di ciascuna immagine | Non esiste un'autorizzazione commerciale valida per tutto COCO; selezionare solo fotografie con licenza individuale compatibile e verificabile | **Condizionato immagine per immagine** |
| [BDD100K](https://doc.bdd100k.com/license.html) | 100.000 fotogrammi stradali | Il file [BSD-3-Clause del repository](https://github.com/bdd100k/bdd100k/blob/master/LICENSE) parla di “software” e non dimostra una licenza dei dati. Durante questa verifica la pagina ufficiale dell'accordo sui dati non era raggiungibile | Non è stata trovata un'autorizzazione commerciale generale verificabile; serve un'autorizzazione scritta specifica | **Escluso** finché non viene ottenuta e archiviata l'autorizzazione |
| [VisDrone](https://github.com/VisDrone/VisDrone-Dataset) | Persone e veicoli piccoli ripresi da drone | Il repository ufficiale non contiene una licenza per immagini e annotazioni | Nessuna autorizzazione commerciale verificabile | **Escluso** |
| [AU-AIR](https://github.com/freeridering/auair-dataset/blob/master/LICENSE) | Circa 32.000 immagini da drone con persone e veicoli | La licenza MIT usa esplicitamente il termine “Software”; il repository rimanda immagini e annotazioni a file esterni senza assegnare loro una licenza | Nessuna autorizzazione commerciale verificabile per i dati | **Escluso** |
| [CrowdHuman](https://www.crowdhuman.org/download.html) | Persone numerose e fortemente coperte | I termini ufficiali limitano l'uso a ricerca ed educazione non commerciali e vietano la redistribuzione delle immagini | No | **Escluso** |
| [KITTI](https://www.cvlibs.net/datasets/kitti/) | Auto, pedoni e ciclisti | CC BY-NC-SA 3.0; la clausola `NC` vieta l'uso commerciale | No | **Escluso** |
| [Cityscapes](https://www.cityscapes-dataset.com/license/) | Persone e veicoli in ambiente urbano | Licenza dedicata a scopi scientifici non commerciali; vieta anche la distribuzione del dataset e delle versioni modificate | No | **Escluso** |
| [Mapillary Object Dataset](https://www.mapillary.com/dataset/assets/mapillary-object-dataset-research-use-license-2019.pdf) | Immagini stradali annotate | Research Use License: vieta prodotti, servizi, attività commerciali e opere derivate; la licenza commerciale è separata | No con la licenza pubblica | **Escluso**, salvo contratto commerciale scritto |

Conclusione: tra i dataset esterni esaminati, soltanto PhenoCam Network può
essere importato come archivio con una licenza aperta che consente chiaramente
l'uso commerciale. Open Images e COCO sono utilizzabili soltanto come
sottoinsiemi controllati immagine per immagine. Tutti gli altri sono esclusi.

“Approvato” riguarda la licenza dei dati, non sostituisce la verifica su privacy,
immagine delle persone, marchi e altri diritti. Questa classificazione è
prudenziale e non costituisce consulenza legale.

### Selezione raccomandata

Ordine di preferenza:

1. immagini interne PhenoCam/Agrocam, dopo aver documentato titolarità e base
   giuridica;
2. PhenoCam Network CC BY 4.0, con attribuzione e annotazioni interne;
3. sottoinsieme Open Images composto solo da immagini verificate singolarmente;
4. sottoinsieme COCO composto solo da immagini Flickr con licenza compatibile e
   verificata;
5. non scaricare né usare gli altri dataset per il modello commerciale senza
   una licenza scritta aggiuntiva.

Per ogni immagine esterna approvata, il registro deve contenere almeno URL della
pagina originale, autore, licenza e versione, data della verifica, testo di
attribuzione e copia delle informazioni di licenza. Se uno di questi elementi
manca, l'immagine non entra nel dataset.

Per ridurre i dubbi sulle opere derivate, accettare nel progetto soltanto
`CC0` e `CC BY` tra le licenze Creative Commons. Escludere `NC`; escludere anche
`ND` e, in via prudenziale, `SA` finché un legale non conferma come applicarne
gli obblighi ai pesi addestrati. La licenza aperta non sostituisce il controllo
su persone riconoscibili, targhe, marchi o altri diritti presenti nelle immagini.

---

## 5. Strategia di selezione, unificazione e bilanciamento

### Dataset pilota

Obiettivo iniziale:

- inventariare e annotare le poche centinaia di immagini locali disponibili,
  puntando indicativamente a 300–600 immagini se l'archivio lo consente;
- dedicare circa il 35–45% della selezione a immagini negative reali, comprese
  quelle difficili da distinguere;
- includere tutti i positivi locali rari, in particolare persone, soggetti
  coperti, tagliati dai bordi, piccoli o illuminati male;
- integrare inizialmente 1.000–3.000 immagini esterne con licenza verificata,
  privilegiando negativi PhenoCam Network e positivi Open Images. I pesi di
  partenza sono già addestrati su COCO, quindi non serve importare di nuovo
  l'intero COCO;
- mantenere in ogni ciclo di addestramento almeno il 30–50% di campioni locali,
  anche riutilizzandoli, per evitare che le immagini esterne abbiano un peso
  eccessivo.

Le percentuali sono criteri di campionamento e possono sovrapporsi. I dati
locali sono troppo pochi per addestrare da soli un rilevatore robusto: servono
soprattutto ad adattare il modello all'ambiente reale, scegliere le soglie e
valutare il risultato.

### Procedura

1. Inventariare tutte le immagini disponibili con sito, camera, timestamp,
   risoluzione, sequenza, condizioni ambientali, provenienza e licenza.
2. Calcolare hash esatto e similarità percettiva per eliminare duplicati.
3. Eseguire il modello corrente con una soglia diagnostica molto bassa, senza
   considerare le sue previsioni come annotazioni corrette.
4. Dividere le immagini candidate in gruppi: punteggio alto, punteggio tra 0,05
   e 0,30, nessun rilevamento, disaccordo tra immagine completa e ritagli,
   immagini negative difficili e condizioni rare.
5. Annotare manualmente tutte le immagini selezionate.
6. Unificare le categorie dei dataset esterni conservando categoria originale,
   licenza, corrispondenza con le categorie operative, identificatore della
   sorgente e attributi disponibili.
7. Prima di copiare un'immagine esterna, registrare autore, URL originale,
   licenza, versione, data di verifica e prova dell'attribuzione. Se la fonte non
   è raggiungibile o la licenza non è `CC0` o `CC BY`, escluderla.
8. Campionare per sorgente e condizione, non solamente per numero di riquadri.
9. Dopo il primo ulteriore addestramento, cercare gli errori su immagini
   negative difficili provenienti da un nuovo periodo temporale.
10. Non pubblicare immagini locali sensibili insieme al codice: pubblicare
   l'elenco strutturato dei dati e la loro provenienza, mentre l'accesso alle
   immagini segue la base giuridica e il periodo di conservazione approvati.

---

## 6. Schema di annotazione e casi ambigui

### Classificazione raccomandata

Usare come classi operative:

- `person`;
- `motor_vehicle`.

Per `motor_vehicle`, registrare anche il tipo specifico del veicolo. Questa
informazione può servire nei rapporti anche se non viene usata direttamente per
addestrare il modello:

- `car`;
- `van`;
- `pickup`;
- `truck`;
- `bus`;
- `motorcycle`;
- `tractor`;
- `agricultural_machine`;
- `other_land_vehicle`;
- `unknown_vehicle`.

Questa classificazione è coerente con la finalità di tutela della privacy:
confondere un'automobile con un camion è un errore di categoria, ma il filtro si
attiva comunque e quindi non produce una mancata protezione.

Una bicicletta senza persone non è un soggetto rilevante e non deve causare lo
scarto. Un ciclista o una persona che porta una bicicletta appartiene alla classe
`person`; `bicycle` può essere conservato solo come attributo diagnostico. Non
si propone una regola relazionale persona+bicicletta nella prima versione,
perché aggiungerebbe complessità senza migliorare la tutela già fornita dalla
classe `person`.

La compatibilità con l'attuale programma, che si aspetta 80 classi, richiede una
decisione:

- **raccomandato:** usare un modello con due sole classi e modificare in seguito,
  nel modo più piccolo possibile, il formato atteso dal programma;
- **alternativa senza modificare il programma:** mantenere le sei classi COCO,
  accettando una corrispondenza imperfetta per furgoni e macchinari.

### Informazioni minime da registrare

Ogni annotazione deve contenere i seguenti campi tecnici:

- `image_id`;
- `source`;
- `site_id`;
- `camera_id`;
- `sequence_id`;
- `timestamp`;
- `bbox_visible`;
- `class`;
- `vehicle_subtype`, se applicabile;
- `person_context`: none, cyclist, carrying_bicycle, driver, passenger;
- `occlusion`: none, partial, heavy;
- `truncation`: none, partial, heavy;
- `lighting`;
- `weather`;
- `confuser`;
- `ignore`;
- identità dell'annotatore e versione delle linee guida.

### Regole

- Annotare tutte le persone reali visibili, anche sedute, chine, parzialmente
  nascoste o alla guida.
- Annotare persona e veicolo motorizzato separatamente quando entrambi sono
  visibili.
- Annotare veicoli motorizzati parcheggiati, fermi o parzialmente fuori campo.
- Il riquadro usato per l'addestramento deve racchiudere la parte visibile e
  restare nei limiti dell'immagine.
- Un'eventuale `bbox_full` stimata può essere conservata per analisi, ma non
  deve sostituire il riquadro della parte visibile.
- Immagini di persone o veicoli su cartelli, fotografie, pubblicità o schermi
  devono essere marcate `confuser`, cioè possibile fonte di confusione, e non
  come soggetti reali.
- Riflessi in vetri o specchi di una persona o di un veicolo reale presente
  nella scena sono positivi quando il soggetto è riconoscibile.
- Manichini, statue e spaventapasseri sono immagini negative difficili, non
  soggetti rilevanti.
- Un caso visivamente ambiguo va marcato `ignore`, cioè escluso dal calcolo
  principale durante l'addestramento, e riesaminato nel test.
- Se due annotatori non concordano sulla presenza del soggetto, marcare
  `ignore` e non usarlo per decidere se il modello supera i requisiti.
- Trattori e macchinari agricoli semoventi che possono trasportare una persona
  sono `motor_vehicle`; attrezzature trainate o ferme senza cabina sono immagini
  negative difficili.
- Due riquadri riferiti allo stesso soggetto sono un errore di annotazione.
- Test: tutte le immagini devono essere annotate da due persone e i disaccordi
  devono essere risolti insieme, se è disponibile un aiuto.
- Addestramento: doppia annotazione almeno sul 10–20% e controllo mirato di tutte
  le immagini dichiarate negative.
- Se il secondo annotatore non è disponibile, il responsabile rietichetta il
  test dopo 1–2 settimane, in ordine casuale e senza vedere la prima versione.
  Il disaccordo va risolto e riportato; questa procedura misura la coerenza ma
  non elimina la tendenza soggettiva di un singolo annotatore.

---

## 7. Suddivisione tra addestramento, validazione e test

### Principio

La suddivisione avviene prima per gruppo, poi per percentuale. Il gruppo minimo
è:

`site_id + camera_id + sequence_id`

Fotogrammi vicini nel tempo, raffiche, immagini quasi duplicate o immagini della
stessa permanenza di un soggetto devono restare nello stesso insieme.

### Suddivisione raccomandata

Con tre sole camere non conviene riservarne una intera: resterebbero soltanto due
siti per l'addestramento e non sarebbe possibile capire se una differenza nei
risultati dipende dalla camera o dal modello. Per ciascun sito si propongono
invece blocchi temporali contigui e non sovrapposti:

- circa il 55–65% dei primi periodi per l'addestramento;
- circa il 15–20% di un periodo successivo per la scelta delle soglie e delle
  impostazioni;
- circa il 20–25% del periodo più recente per il test finale, che non deve essere
  consultato durante lo sviluppo.

Le percentuali si applicano ai gruppi temporali, non a immagini estratte a caso.
Se l'archivio lo consente, il test dovrebbe raggiungere 60–100 immagini
negative, almeno 50 persone e almeno 75 veicoli
motorizzato. Sono obiettivi di acquisizione, non numeri da inventare o ottenere
duplicando immagini: se non vengono raggiunti si riportano i conteggi reali e
l'incertezza.

Se il sistema rileva 95 immagini positive su 100, la percentuale reale potrebbe
essere inferiore: l'intervallo statistico di Wilson ha un limite inferiore
vicino all'89%. Allo stesso modo, 5 falsi allarmi su 100 immagini negative
producono un limite superiore vicino all'11%. Con questi volumi, “95% di
rilevamento e 5% di falsi allarmi” è quindi un obiettivo misurato sul campione,
non una garanzia statistica forte. Immagini molto simili e vicine nel tempo
riducono ulteriormente l'informazione effettiva.

Regole aggiuntive:

- nessuna immagine esterna nel test operativo;
- test definito e bloccato prima di scegliere le soglie;
- soglie scelte esclusivamente sull'insieme di validazione;
- test eseguito formalmente una sola volta per ogni candidato al rilascio;
- eventuali sottoinsiemi di Open Images o COCO usati solo per l'addestramento;
- risultati separati per ciascuno dei tre siti;
- prova aggiuntiva in tre turni: ogni volta si esclude un sito
  dall'addestramento e lo si usa per la verifica; questa prova resta separata dal
  test finale;
- nessuna dichiarazione di generalizzazione a siti mai visti. La prima futura
  quarta camera va conservata come prova su un sito nuovo prima di usarla per
  l'addestramento.

---

## 8. Variazioni artificiali delle immagini di addestramento

Applicare solo alle immagini di addestramento:

- variazioni moderate di luminosità, gamma e contrasto;
- temperatura colore e saturazione limitate;
- ombre locali e controluce;
- sfocatura da movimento o messa a fuoco;
- foschia, pioggia leggera e riduzione di visibilità;
- rumore del sensore e compressione JPEG;
- variazioni moderate di dimensione e scala;
- ritagli che lascino gli oggetti parzialmente fuori campo;
- coperture artificiali realistiche simili a foglie, rami o pali;
- riflessione orizzontale dell'immagine;
- composizione `mosaic` limitata ai primi cicli di addestramento;
- tecniche `mixup` e `copy-paste` solo se l'immagine risultante rimane plausibile.

Evitare:

- capovolgimento verticale;
- rotazioni ampie;
- deformazioni prospettiche irrealistiche;
- colori estremi;
- inserimento di veicoli o persone senza ombre e dimensioni coerenti;
- qualsiasi variazione artificiale sulle immagini di validazione e test.

Per riprodurre il comportamento del programma, durante l'addestramento ogni
immagine locale dovrebbe generare la vista completa, un numero limitato di
ritagli contenenti soggetti e alcuni ritagli negativi difficili. Non è necessario
generare e salvare tutte le 16 viste a ogni ciclo.

---

## 9. Strategia di ulteriore addestramento e confronto

### Esperimenti

| Esperimento | Scopo |
|---|---|
| E0 | YOLO26n corrente, senza nuovo addestramento; riferimento iniziale completo |
| E1 | Ulteriore addestramento con sole immagini locali o simili a PhenoCam, solo per misurare il contributo dei dati esterni |
| E2 | Ulteriore addestramento con immagini locali, PhenoCam Network e Open Images verificato; candidato principale |
| E3 | YOLO26s con il dataset migliore, solo se E2 non raggiunge i requisiti |
| E4 | Eventuale revisione dei ritagli o delle classi, solo dopo aver dimostrato un errore strutturale |

### Procedura

1. Inizializzare dai pesi correnti.
2. Usare immagini in ingresso di 640×640 pixel, come nell'esecuzione finale.
3. Sul MacBook Air usare `device=mps`, dimensione automatica dei gruppi oppure
   inizialmente 8 immagini per gruppo, da 50 a 100 cicli massimi e interruzione
   anticipata se i risultati non migliorano.
4. Addestrare brevemente il livello finale del modello e poi l'intero modello.
5. Scegliere il modello in base al tasso di rilevamento, rispettando il limite
   dei falsi allarmi sull'insieme di validazione; non usare soltanto la mAP.
6. Ripetere almeno tre volte l'addestramento finale con inizializzazioni casuali
   diverse.
7. Conservare l'elenco dei dati, il commit del codice, la configurazione, il
   valore casuale iniziale, le metriche e l'impronta digitale dei pesi.
8. Esportare il modello in formato ONNX FP32 usando la stessa versione del
   formato attuale e verificare che produca risultati equivalenti al modello
   usato durante l'addestramento.
9. Valutare il programma completo con 16 viste, non soltanto il modello su
   immagini singole.
10. Misurare sul Raspberry Pi il tempo totale per immagine, il picco di memoria,
    la temperatura, eventuali rallentamenti dovuti al calore e la stabilità su
    lunghe serie di immagini.
11. Non ridurre la precisione numerica del modello prima di aver dimostrato che
    il tasso di rilevamento non peggiora.

Il modello E2 deve sostituire E0 solo se migliora il tasso di rilevamento, senza
superare il limite di falsi allarmi e senza peggiorare tempi e consumo di memoria
sul dispositivo.

---

## 10. Misure, soglie e verifica

### Misure principali

Poiché in produzione viene scartata l'intera immagine, calcolare separatamente
per `person` e `motor_vehicle` queste due misure principali:

- **tasso di attivazione corretto:** percentuale delle immagini che contengono
  la classe e attivano il filtro;
- **tasso di falsi allarmi:** percentuale delle immagini prive di persone e
  veicoli rilevanti che vengono scartate per errore.

Calcolare inoltre:

- percentuale dei singoli soggetti rilevati, considerando corretta una
  localizzazione con IoU ≥0,50;
- precisione: quota delle rilevazioni prodotte dal sistema che sono corrette;
- numero medio di falsi rilevamenti per immagine;
- numero medio di soggetti mancati per immagine;
- frequenza con cui lo stesso soggetto viene rilevato più volte;
- intervallo di confidenza al 95%, cioè il margine di incertezza statistica.

Il tasso di attivazione corretto governa la decisione di scarto. La percentuale
dei singoli soggetti rilevati resta obbligatoria per capire gli errori e diventa
la misura principale se viene abilitata la sfocatura selettiva: rilevare una sola
persona su tre può essere sufficiente per scartare l'immagine, ma non per
sfocare tutte le persone presenti.

### Ricerca delle soglie

1. Salvare tutte le previsioni con punteggio almeno 0,01.
2. Provare soglie da 0,01 a 0,80, con incrementi di 0,01.
3. Usare soglie distinte per persone (`t_person`) e veicoli
   (`t_motor_vehicle`).
4. Per ogni coppia di soglie calcolare tasso di rilevamento, tasso di falsi
   allarmi, precisione, falsi rilevamenti per immagine e numero di immagini
   scartate.
5. Sull'insieme di validazione scegliere le soglie più alte che mantengono il
   tasso di rilevamento richiesto e riducono al minimo i falsi allarmi.
6. Non modificare più soglie e regole di elaborazione dopo aver avviato il test.
7. Calcolare gli intervalli di confidenza ricampionando gruppi completi dello
   stesso sito e della stessa sequenza, non singole immagini correlate.

### Requisiti per approvare il modello

| Misura | Requisito |
|---|---|
| Immagini con persone che attivano il filtro | almeno il 95% del campione |
| Immagini con veicoli rilevanti che attivano il filtro | almeno il 95% del campione |
| Immagini negative scartate per errore | non più del 5% del campione |
| Risultati | sempre separati per persone e veicoli motorizzati |
| Condizioni difficili | obiettivo diagnostico di almeno il 90% per ogni gruppo |
| Confronto con E0 | nessun peggioramento significativo |
| Latenza e RAM | entro i limiti confermati sul Pi |
| Duplicati | misurati e non tali da compromettere sfocatura o conteggi |

Se nessuna coppia di soglie rileva almeno il 95% delle immagini positive e
mantiene i falsi allarmi entro il 5%:

1. mantenere come priorità il rilevamento di almeno il 95% delle immagini
   positive;
2. consentire fino al 10% di falsi allarmi solo in una prova controllata nella
   quale gli originali siano conservati e la cancellazione definitiva sia
   disabilitata;
3. aggiungere immagini negative difficili e soggetti coperti;
4. provare YOLO26s sul Pi;
5. non approvare per la produzione un modello con rilevamento inferiore al 95% o
   falsi allarmi superiori al 5% senza un'esplicita accettazione del rischio.

---

## 11. Matrice di test

Le celle possono sovrapporsi; i conteggi non devono essere sommati.

| Dimensione | Condizioni da coprire | Copertura minima |
|---|---|---|
| Ambiente | agricolo, bosco/naturale, strada sullo sfondo, parcheggio, accesso aziendale | tutti i tipi presenti nei tre siti, con conteggio per sito |
| Soggetto | persona; auto/furgone; camion/bus; moto; trattore/macchinario | risultati per classe e sottotipo; bicicletta solo come contesto della persona |
| Dimensione | molto piccolo, piccolo, medio, grande, misurata nel ritaglio in cui il soggetto appare più grande | tutti i gruppi presenti; obiettivo di almeno 20 soggetti per gruppo critico |
| Parte nascosta | nessuna, parziale, pesante | tutti i gruppi presenti; obiettivo di almeno 20 soggetti per gruppo critico |
| Soggetto tagliato dal bordo | nessun taglio, bordo laterale, bordo inferiore o superiore | tutti i casi disponibili; obiettivo di almeno 20 soggetti per classe |
| Luce | giorno, ombra, alba, tramonto, controluce | tutti i periodi disponibili, senza sintetici nel test |
| Meteo | sereno, nuvoloso, pioggia, foschia/nebbia | tutte le condizioni disponibili, con conteggi espliciti |
| Qualità | sfocatura, compressione, sporco o condensa sulla lente, immagine troppo scura | tutti i casi disponibili; obiettivo di almeno 20 immagini per gruppo |
| Numero di soggetti | singolo, multipli, sovrapposti | tutte le immagini con più soggetti disponibili |
| Negativi ordinari | vegetazione, campo vuoto, strada vuota, parcheggio vuoto | obiettivo complessivo 60–100 immagini |
| Immagini negative difficili | cartelli, pali, strutture, attrezzi esclusi, figure stampate, schermi, animali | obiettivo di almeno 30 immagini, coprendo tutte le famiglie disponibili |
| Generalizzazione temporale | periodi bloccati delle tre camere note | un blocco test successivo per ciascun sito |

Per la dimensione utilizzare il lato corto del riquadro nella vista 640×640 in
cui il soggetto appare più grande:

- molto piccolo: meno di 16 pixel;
- piccolo: da 16 a 32 pixel;
- medio: da 33 a 96 pixel;
- grande: più di 96 pixel.

Gli oggetti sui quali due annotatori non riescono a concordare non devono essere
cancellati dal dataset: vanno marcati `unresolvable/ignore`, esclusi dal calcolo
principale e riportati separatamente.

---

## 12. Rischi, limiti e mitigazioni

| Rischio | Mitigazione |
|---|---|
| Obblighi AGPL incompatibili con parti proprietarie | AGPL per l'intera applicazione oppure licenza Enterprise o tecnologia alternativa prima dell'addestramento |
| Titolarità congiunta non formalizzata | accordo azienda–università su copyright, contributi e autorità di rilascio |
| Codice Apache 2.0 distribuito insieme a componenti AGPL | verificare la licenza dell'intero prodotto; Apache 2.0 da sola non risolve gli obblighi Ultralytics |
| Dataset pubblico ma non commercialmente autorizzato | registro di provenienza e approvazione legale per sorgente e immagine |
| Licenza del software scambiata per licenza del dataset | verificare separatamente i diritti su immagini, annotazioni e strumenti; non accettare una licenza MIT o BSD del solo codice come prova sui dati |
| Attribuzione incompleta o fonte rimossa in seguito | conservare nel registro URL, autore, testo e versione della licenza, data della verifica e copia della prova disponibile al momento del download |
| Immagini contenenti dati personali | base giuridica, accesso ristretto, cifratura, periodo di conservazione e minimizzazione |
| Immagini simili presenti in più insiemi | separazione per sito, camera e sequenza; eliminazione dei duplicati esatti e quasi duplicati |
| Test troppo simile all'addestramento | blocchi temporali successivi e separati per ognuna delle tre camere |
| Annotazioni negative incomplete | doppio controllo di tutte le immagini negative |
| Annotazioni incoerenti su trattori e furgoni | classe `motor_vehicle` con sottotipo, senza corrispondenze forzate |
| Modello troppo adattato allo sfondo `TS_326` | più siti, stagioni e camere |
| Troppi falsi allarmi quando si abbassano le soglie | soglie distinte per classe e raccolta mirata delle immagini negative difficili |
| Oggetti fisicamente irriconoscibili | etichetta `ignore`, risultati separati e nessuna promessa artificiale del 95% |
| YOLO26s troppo lento sul Pi | prova preventiva; YOLO26n resta il modello predefinito |
| Differenza tra il modello addestrato e quello esportato in ONNX | confronto numerico e misure sul programma ONNX realmente usato |
| Cambiamenti stagionali o nuove camere | controllo periodico e test su periodi recenti |
| Duplicati prodotti dalle diverse viste | misura dedicata e revisione dell'eliminazione dei duplicati solo se necessaria |
| Validazione debole con poche centinaia di immagini | intervalli di confidenza, conteggi effettivi e nessuna pretesa su siti nuovi |
| Tendenza soggettiva di un solo annotatore | secondo revisore sul test o nuova annotazione differita senza vedere la prima |

Il trattamento delle immagini deve rispettare la limitazione delle finalità, la
minimizzazione dei dati, il periodo di conservazione e il controllo degli
accessi; sono principi richiamati
dalla [Commissione europea sul GDPR](https://commission.europa.eu/law/law-topic/data-protection/information-business-and-organisations/principles-gdpr_en).
Serve comunque una valutazione legale specifica, non soltanto tecnica.

---

## 13. Attività, impegno e criteri di completamento

| Priorità | Attività | Impegno stimato | Criterio di completamento |
|---|---|---:|---|
| P0 | Formalizzare scarto in produzione e sfocatura opzionale | 0,5 giorni | misura principale e costo degli errori formalizzati |
| P0 | Approvare le classi `person` e `motor_vehicle` | 1 giorno | guida di annotazione approvata |
| P0 | Verificare titolarità, AGPL e licenze dei dataset | 3–6 giorni | decisione azienda–università e registro delle sorgenti |
| P0 | Inventariare l'archivio PhenoCam/Agrocam completo | 2–3 giorni | conteggi per sito, periodo, clima e soggetto |
| P0 | Costruire e congelare il test operativo | 3–5 giorni | blocchi temporali e conteggi reali documentati |
| P0 | Misurare E0 e provare le soglie | 2–3 giorni | risultati per classe, errori catalogati e riferimento iniziale riproducibile |
| P1 | Selezionare e annotare il dataset locale | 4–8 giorni | controllo di qualità superato e insiemi separati correttamente |
| P1 | Curare e documentare i dati esterni | 4–8 giorni | ogni immagine ammessa ha prova della licenza CC0 o CC BY, provenienza, attribuzione e corrispondenza tra classi; le sorgenti escluse non sono presenti |
| P1 | Eseguire E1/E2 con tre inizializzazioni | 3–6 giorni | risultati riproducibili e candidato selezionato |
| P1 | Verifica ONNX e Raspberry Pi | 3–5 giorni | equivalenza dei risultati, tempo, memoria e stabilità documentati |
| P1 | Valutazione finale sul test bloccato | 1–2 giorni | requisiti verificati senza modifiche successive |
| P2 | Raccolta degli errori sulle immagini negative e seconda iterazione | 5–10 giorni | meno falsi allarmi senza ridurre il tasso di rilevamento |
| P2 | Aggiornamento documentazione e piano di monitoraggio | 2–3 giorni | schede descrittive di modello e dataset e procedura di rilascio |

Stima complessiva iniziale: circa 4–7 settimane di lavoro per una persona. La
revisione delle annotazioni richiederà probabilmente più tempo del calcolo sul
MacBook.

## Decisioni proposte e punti ancora aperti

| Tema | Proposta o stato | Azione necessaria |
|---|---|---|
| Volume locale | alcune centinaia di immagini, circa tre siti | inventario esatto per sito e mese |
| Suddivisione dei dati | blocchi temporali distinti in ogni sito | verificare che immagini positive e negative siano sufficienti |
| Titolarità immagini | azienda e università coinvolte | documentare proprietario e base giuridica |
| Licenza software | il codice è Apache 2.0; resta da risolvere il componente Ultralytics AGPL | approvazione legale congiunta prima del rilascio commerciale |
| Immagini sensibili | non pubblicarle automaticamente con il codice | definire accessi, periodo di conservazione e base giuridica |
| Risultato in produzione | scarto dell'intera immagine; sfocatura opzionale | confermato |
| Biciclette | bicicletta sola negativa; la persona è il soggetto rilevante | approvare nella guida di annotazione |
| Veicoli | motorizzati e macchine semoventi con operatore sono rilevanti | approvare i casi limite |
| Riflessi | soggetto reale riconoscibile positivo | approvare nella guida |
| Stampe e schermi | immagini negative difficili | approvare nella guida |
| Falsi allarmi | non più del 5% in produzione; fino al 10% solo in una prova reversibile | approvare esplicitamente |
| Computer di addestramento | MacBook Air M4 24 GB, `device=mps` | eseguire una breve prova sotto carico |
| Dispositivo finale | limiti descritti nel repository | misurare su Raspberry Pi reale |
| Annotazioni | responsabile principale + possibile aiuto | trovare un secondo revisore almeno per il test |
| Funzionamento su siti diversi | misurabile per ora solo sulle tre camere note | riservare una futura quarta camera come prova su un sito nuovo |

---

## Compendio dei termini tecnici e delle abbreviazioni

| Termine | Significato nel documento |
|---|---|
| Addestramento (`training`) | Fase nella quale il modello modifica i propri parametri usando immagini annotate. |
| AGPL-3.0 | Licenza libera con copyleft forte. In generale richiede di rendere disponibile il sorgente corrispondente dell'opera coperta, anche agli utenti che interagiscono con una versione modificata attraverso una rete. |
| Annotazione | Informazione aggiunta manualmente a un'immagine: indica quali soggetti sono presenti, la loro classe e il riquadro che li contiene. |
| Apache License 2.0 | Licenza permissiva attualmente applicata al codice del repository. Non elimina gli obblighi di licenza di componenti esterni distribuiti insieme al codice. |
| Audit | Controllo sistematico dei dati, delle annotazioni o dei risultati. |
| Batch | Gruppo di immagini elaborato insieme durante un passo di addestramento. |
| Box, bounding box o `bbox` | Riquadro rettangolare che delimita una persona o un veicolo nell'immagine. |
| BSD | Famiglia di licenze software permissive. La licenza del codice di uno strumento non si estende automaticamente ai dati elaborati dallo strumento. |
| CC0 | Strumento Creative Commons con cui il titolare rinuncia, per quanto possibile, ai diritti d'autore. Consente l'uso commerciale; registrare comunque provenienza e autore. |
| CC BY | Licenza Creative Commons che permette il riuso, anche commerciale, imponendo l'attribuzione e il rispetto delle condizioni previste. |
| CC BY-NC-SA | Licenza Creative Commons che richiede attribuzione, vieta l'uso commerciale e impone la stessa licenza alle opere derivate. |
| `NC`, `ND`, `SA` | Clausole Creative Commons: `NC` vieta l'uso commerciale, `ND` vieta la distribuzione di opere derivate, `SA` impone la stessa licenza alle opere derivate. |
| Classe | Categoria che il modello deve riconoscere, per esempio `person` o `motor_vehicle`. |
| COCO | Dataset generale di immagini e annotazioni sul quale sono state definite le 80 classi del modello di partenza. |
| Confidence o confidenza | Punteggio con cui il modello esprime quanto considera affidabile una rilevazione. Non equivale automaticamente a una probabilità calibrata. |
| `confuser` | Etichetta per un elemento che può ingannare il modello, come una persona stampata su un cartello, ma che non è un soggetto reale. |
| Copyleft | Regola di licenza che impone di mantenere determinate libertà e obblighi anche quando il software viene modificato o distribuito insieme ad altro codice. |
| `copy-paste`, `mixup`, `mosaic` | Tecniche che costruiscono nuove immagini di addestramento combinando parti o intere immagini esistenti. |
| CPU | Processore generale del computer. Sul Raspberry Pi esegue il modello ONNX. |
| Dataset | Insieme organizzato di immagini, annotazioni e informazioni sulla loro provenienza. |
| E0–E4 | Nomi brevi assegnati nel piano ai diversi esperimenti, per poterli confrontare senza ambiguità. |
| EXIF | Metadati incorporati nelle fotografie; possono contenere, tra le altre cose, orientamento e data di acquisizione. |
| Falso negativo | Persona o veicolo realmente presente che il sistema non rileva. |
| Falso positivo o falso allarme | Rilevazione prodotta dal sistema quando il soggetto indicato non è realmente presente. |
| Fine-tuning | Ulteriore addestramento di un modello già addestrato, eseguito con dati più vicini all'ambiente reale del progetto. |
| FLOPs o GFLOPs | Numero indicativo di operazioni matematiche richieste dal modello; GFLOPs significa miliardi di operazioni. Aiuta a confrontare il costo dei modelli, ma non determina da solo il tempo reale. |
| FP32 | Rappresentazione numerica a 32 bit in virgola mobile usata dal modello ONNX attuale. |
| FPR | `False Positive Rate`, cioè tasso di falsi allarmi. Qui indica la percentuale di immagini negative scartate per errore. |
| GDPR | Regolamento generale dell'Unione europea sulla protezione dei dati personali. |
| GB, MB, ms | Gigabyte, megabyte e millisecondi; misurano rispettivamente memoria o spazio e tempo. |
| GPU | Processore specializzato nei calcoli paralleli usati durante l'addestramento dei modelli. |
| Hard negative o immagine negativa difficile | Immagine senza soggetti rilevanti che contiene elementi facilmente confondibili con persone o veicoli. |
| Hash | Impronta digitale calcolata da un file; serve a identificare con precisione dati, codice o pesi del modello. |
| `ignore` | Etichetta per un caso ambiguo che viene conservato nel dataset ma escluso dal calcolo principale delle metriche. |
| Immagine negativa | Immagine che non contiene né persone né veicoli rilevanti. |
| Immagine positiva | Immagine che contiene almeno una persona o un veicolo rilevante. |
| Intervallo di Wilson | Metodo statistico usato per indicare l'incertezza di una percentuale misurata su un numero limitato di esempi. |
| IoU | `Intersection over Union`: misura da 0 a 1 quanto il riquadro previsto coincide con quello annotato. Il valore 0,50 è la soglia proposta per considerare corretta la posizione. |
| JPEG | Formato compresso usato dalle immagini presenti nel repository. Compressioni forti possono cancellare dettagli utili al modello. |
| Manifest | Elenco strutturato delle immagini usate, della loro provenienza, licenza e appartenenza ad addestramento, validazione o test. |
| mAP | `mean Average Precision`: misura riassuntiva comune nei rilevatori di oggetti. È utile per confrontare modelli, ma non sostituisce i requisiti specifici su rilevamento e falsi allarmi. |
| MIT | Licenza software permissiva. Se compare nel repository di uno strumento non copre automaticamente immagini o annotazioni scaricate separatamente. |
| `motor_vehicle` | Classe proposta per veicoli motorizzati e macchine semoventi capaci di trasportare una persona. Non comprende la bicicletta senza persona. |
| MPS | Sistema Apple che permette a PyTorch e Ultralytics di usare la GPU integrata dei Mac con Apple Silicon. |
| ONNX | Formato portabile usato per esportare il modello addestrato e utilizzarlo senza l'ambiente completo di addestramento. |
| ONNX Runtime | Programma che esegue il modello ONNX sul Raspberry Pi. |
| Open data o dati aperti | Dati pubblicati con una licenza che permette accesso, riuso e ridistribuzione. Per i dataset è un termine più preciso di “open source”. |
| Open source | Software distribuito con una licenza che permette di leggere, usare, modificare e ridistribuire il codice rispettando le relative condizioni. Rendere pubblico un repository non basta da solo. |
| P0, P1, P2 | Livelli di priorità delle attività: P0 è indispensabile e viene prima di P1; P2 comprende miglioramenti successivi. |
| `person` | Classe proposta per tutte le persone reali visibili, comprese persone parzialmente coperte, ciclisti, conducenti e passeggeri visibili. |
| Pesi del modello | Valori numerici appresi durante l'addestramento; contengono ciò che il modello ha imparato dai dati. |
| Pipeline o flusso di elaborazione | Successione completa delle operazioni: lettura dell'immagine, creazione dei ritagli, esecuzione del modello, unione dei risultati e decisione finale. |
| Precisione (`precision`) | Percentuale delle rilevazioni prodotte dal sistema che corrisponde a soggetti reali. |
| PyTorch | Libreria software usata da Ultralytics per addestrare il modello prima dell'esportazione in ONNX. |
| Quantizzazione | Riduzione della precisione numerica del modello per diminuire memoria e tempo di esecuzione; può però ridurre l'accuratezza. |
| RAM | Memoria di lavoro disponibile mentre il programma è in esecuzione. |
| Raspberry Pi | Piccolo computer sul quale verrà eseguito il filtro in produzione. |
| Recall o tasso di rilevamento | Percentuale dei casi realmente presenti che il sistema individua. Un recall del 95% significa 95 casi rilevati su 100. |
| RGB | Rappresentazione a tre canali di colore: rosso, verde e blu. |
| Seed | Valore iniziale della componente casuale dell'addestramento. Ripetere con seed diversi permette di verificare che il risultato non dipenda dalla fortuna. |
| Soglia (`threshold`) | Punteggio minimo di confidenza richiesto per accettare una rilevazione. Abbassarla può recuperare più soggetti ma anche aumentare i falsi allarmi. |
| Suddivisione (`split`) | Assegnazione delle immagini agli insiemi di addestramento, validazione e test. |
| Test bloccato | Insieme finale che non deve essere consultato per scegliere modello, soglie o altre impostazioni. |
| Timestamp | Data e ora associate all'acquisizione di un'immagine. |
| Truncation | Soggetto tagliato dal bordo dell'immagine, quindi visibile solo in parte. |
| Validazione | Insieme separato usato per scegliere modello, soglie e impostazioni prima del test finale. |
| YOLO | Famiglia di modelli per rilevare e localizzare oggetti nelle immagini; YOLO26n è la variante piccola attualmente usata. |
