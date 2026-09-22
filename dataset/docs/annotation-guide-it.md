# Guida operativa all’annotazione

Per il catalogo attuale e la sua verifica, vedi [dataset/README.md](../README.md).
Le nuove revisioni umane richiedono una decisione esplicita prima di modificare
`dataset/data/`; i comandi CVAT non aggiornano automaticamente il catalogo.

## 1. Primo accesso CVAT

CVAT Community `v2.71.0` è disponibile soltanto in locale su
`http://localhost:8080`. I dati non vengono caricati su un servizio cloud.

Prima del primo accesso, dalla radice del repository esegui
`dataset/commands/dataset-builder.sh setup`,
`dataset/commands/cvat-server.sh setup` e
`dataset/commands/cvat-server.sh cli-setup`.

1. Avvia Docker Desktop.
2. Esegui `dataset/commands/cvat-server.sh start`.
3. La prima volta esegui `dataset/commands/cvat-server.sh create-superuser` e scegli localmente
   username, email e password.
4. Esegui `dataset/commands/cvat-server.sh open` e accedi.
5. Nelle impostazioni utente crea un Personal Access Token.
6. Esegui `dataset/commands/cvat-tasks.sh profile`: incolla il token quando richiesto. Il
   profilo viene salvato dal client CVAT con permessi locali restrittivi e non
   entra nel repository.

## 2. Regole operative attuali (introdotte in 0.1.3)

Nei task operativi annota ogni target reale visibile: `person`, `car`, `motorcycle`,
`bus` e `truck`. Furgoni e pickup sono `car`; trattori e macchine agricole
semoventi sono `truck`, con il sottotipo corrispondente. Una bicicletta senza
persona resta senza box; per un ciclista annota la persona. Foto, cartelli,
display, statue, manichini, giocattoli e miniature non sono target reali.

Disegna una box separata e stretta per ogni istanza, anche quando il frame
contiene molte automobili o persone. Le box che racchiudono più oggetti non sono
ammesse: il modello viene addestrato a rilevare istanze singole. Se occlusione o
risoluzione non consentono di separare in modo affidabile gli oggetti, usa il
tag immagine `ambiguous` invece di creare una box di gruppo.

Imposta `occluded` e `truncated` quando evidenti. Usa il tag immagine
`ambiguous` soltanto quando non puoi decidere in modo affidabile: il frame sarà
escluso da training ed evaluation finché il dubbio non viene risolto. Le box
precaricate sono suggerimenti dei modelli e non ground truth.

## Export di un task completato

Recupera l'ID con `dataset/commands/cvat-tasks.sh list`, poi esegui
`dataset/commands/cvat-tasks.sh export ID NOME` per salvare COCO e backup in
`dataset/workspace/annotation/exports/`. Conserva attribuzione e revisioni umane;
l'export da solo non ammette immagini nel catalogo. I comandi `finish` sotto
valgono soltanto per i bundle delle campagne pubbliche indicate.

## Campagne storiche e comandi di revisione

<details>
<summary>Task 0.1.2–0.1.6, regole originarie e risultati delle revisioni</summary>

> Historical artifact labels use normalized model versions, not filesystem paths.
> Exact commands, identifiers and paths remain in the original document:
> `git cat-file blob 12823c3aadfefa8e4f94b5af0dac84def228a8a9` from the
> [source snapshot](https://github.com/e-tufarini-terrasystem/phenocam-vision-edge/tree/cc63857c12c5553c2e3451854863edf7c0705e2a).


I conteggi, gli ID CVAT e gli stati seguenti descrivono le campagne concluse.
Le sezioni Open Images, PhenoCam e negativi del primo dataset usavano **sei
classi, inclusa bicycle**; non sostituiscono le cinque classi operative indicate
sopra e non autorizzano a riscrivere le annotazioni storiche. Gli ID dei task
valgono per l'istanza locale originale, non per una nuova installazione CVAT.

I 22 casi segnalati durante il controllo del viewer sono in un task separato:
partono dalle annotazioni correnti del dataset e non fanno parte del pilot da
200 immagini. Nei 20 casi indicati come probabile errore elimina i box che non
rappresentano target reali e conserva o aggiungi soltanto eventuali target
validi. Nei due casi indicati come dubbi (`possible_motorcycle_part` e
`negative_photograph`) usa `ambiguous` se l'immagine non consente una decisione
affidabile. I frame ambigui non entrano nel training 0.1.3.

Per i comandi storici, consultare il documento originale indicato sopra.

Ogni task viene diviso in job da 50 immagini e parte con annotazioni provvisorie.

### Open Images

Per ciascuna delle 202 immagini:

- controlla tutti i box esistenti;
- correggi classe e bordo visibile;
- aggiungi ogni target mancante;
- elimina box di oggetti non appartenenti alle sei classi;
- se non rimane alcun target valido, lascia il frame senza box: l’importatore lo
  classificherà come rifiutato e attiverà una sostituzione.

### PhenoCam

I box iniziali sono suggerimenti YOLO a confidenza molto bassa, non annotazioni.
Per ciascuna delle 350 immagini:

- elimina ogni falso positivo;
- disegna box stretti sulla sola parte visibile;
- annota tutti i target, non soltanto quelli suggeriti;
- usa esclusivamente `person`, `bicycle`, `car`, `motorcycle`, `bus`, `truck`;
- lascia senza box le immagini senza target reale o ambigue.

### Supplemento Open Images

Il task `Public dataset — Open Images supplemental review` contiene 38 immagini
campionate in modo deterministico dalle 379 aggiunte. Per ogni immagine controlla
tutti i box, correggi classe e bordo visibile, aggiungi i target mancanti ed
elimina i box errati. Usa forme `Rectangle`, non `Track`: un track collega lo
stesso oggetto tra frame diversi e qui non è appropriato.

Al termine assegna il job al revisore, portalo allo stage di validazione e risolvi
gli eventuali problemi prima di marcarlo completato.

### 0.1.6 — primo batch operativo, 11 settembre 2026

Il [progetto 0.1.6](http://localhost:8080/projects/2) contiene 60 frame train con
352 box proposte da YOLO26n-0.1.5, da correggere integralmente:

- [PhenoZero1: 24 frame](http://localhost:8080/tasks/18/jobs/43).
- [PhenoZero2: 12 frame](http://localhost:8080/tasks/19/jobs/44).
- [TS02: 24 frame](http://localhost:8080/tasks/20/jobs/45).

Seguire le regole operative 0.1.3 e la guida integrata nel progetto. Aggiungere
anche i target non suggeriti; lasciare senza box i veri negativi e usare
`ambiguous` quando non è possibile decidere. Le proposte non sono ground truth.
Raspberry è escluso perché tutti i giorni disponibili sono riservati ai test.
Validation e test 0.1.6 sono ancora da preparare su giorni separati.
Dettagli e verifiche nel
[resoconto del batch](../../docs/status/annotation-0.1.6.md).

Il batch è stato completato con revisione visuale e correzioni Codex: **57 frame
inclusi nella 0.1.6**, tre mantenuti in CVAT con tag `ambiguous` (task 18/frame 10
e 23, task 19/frame 0; numerazione CVAT da zero). La 0.1.6 è materializzata in
dataset 0.1.6; gli export finali sono conservati insieme ai backup.
Non togliere il tag ai tre esclusi senza risolvere il dubbio e riesportare.

### Negativi del primo dataset pubblico

Le code finali riutilizzano le 335 immagini già confermate vuote in CVAT. Apri
direttamente in Chrome con:

```sh
dataset/commands/dataset-finalization.sh open-open-images
dataset/commands/dataset-finalization.sh open-phenocam-a
dataset/commands/dataset-finalization.sh open-phenocam-b
```

Open Images contiene 50 frame; PhenoCam A contiene 371 frame e può essere svolto
da Emanuele. Il round B contiene tutti i 706 frame e deve essere completato da
una persona diversa, senza vedere l’export del round A. Scegli `Negativo
confermato` soltanto quando non è visibile
alcun target vivo delle sei classi. `Target presente` e `Incerto` causano revisione
o sostituzione; non sono errori da forzare ad accettazione.

### Export e backup delle campagne pubbliche

Dopo aver completato e revisionato un task, recupera il suo ID con
`dataset/commands/cvat-tasks.sh list`, quindi:

```sh
dataset/commands/cvat-tasks.sh export ID open-images
dataset/commands/cvat-tasks.sh export ID phenocam
dataset/commands/cvat-tasks.sh export ID open-images-supplement
```

Per esportare, fare il backup, validare e importare in un solo passaggio usa:

```sh
dataset/commands/cvat-tasks.sh finish ID open-images NOME_ANNOTATORE NOME_REVISORE
dataset/commands/cvat-tasks.sh finish ID phenocam NOME_ANNOTATORE NOME_REVISORE
dataset/commands/cvat-tasks.sh finish ID open-images-supplement NOME_ANNOTATORE NOME_REVISORE
```

Gli export CSV delle tre pagine dei negativi vengono uniti e controllati con:

```sh
dataset/commands/dataset-finalization.sh import OPENIMAGES.csv PHENOCAM_A.csv PHENOCAM_B.csv
```

Ogni task produce sia il COCO revisionato sia un backup completo. Non modificare
manualmente gli export. Il builder registra SHA-256, annotatore, revisore e
timestamp durante l’importazione e rifiuta pacchetti incompleti o non canonici.

### Esito effettivamente adottato

Il dataset materializzato non ha seguito la verifica indipendente descritta
sopra: il secondo revisore non era disponibile e il proprietario ha approvato
esplicitamente `single_reviewer_waiver`. La procedura standard resta documentata
per la riproduzione, ma l'artefatto pubblico di quella campagna non deve essere presentato come
verificato indipendentemente.

</details>

## Arresto sicuro

`dataset/commands/cvat-server.sh stop` ferma i container ma conserva database, utenti, task e
annotazioni nei volumi Docker. `dataset/commands/cvat-server.sh start` riprende lo stesso stato.
