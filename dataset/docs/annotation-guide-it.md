# Guida operativa all’annotazione

## 1. Primo accesso CVAT

CVAT Community `v2.71.0` è disponibile soltanto in locale su
`http://localhost:8080`. I dati non vengono caricati su un servizio cloud.

1. Avvia Docker Desktop.
2. Esegui `dataset/commands/cvat-server.sh start`.
3. La prima volta esegui `dataset/commands/cvat-server.sh create-superuser` e scegli localmente
   username, email e password.
4. Esegui `dataset/commands/cvat-server.sh open` e accedi.
5. Nelle impostazioni utente crea un Personal Access Token.
6. Esegui `dataset/commands/cvat-tasks.sh profile`: incolla il token quando richiesto. Il
   profilo viene salvato dal client CVAT con permessi locali restrittivi e non
   entra nel repository.

## 2. Creazione dei task

### Regole v3

Nei task v3 annota ogni target reale visibile: `person`, `car`, `motorcycle`,
`bus` e `truck`. Furgoni e pickup sono `car`; trattori e macchine agricole
semoventi sono `truck`, con il sottotipo corrispondente. Una bicicletta senza
persona resta senza box; per un ciclista annota la persona. Foto, cartelli,
display, statue, manichini, giocattoli e miniature non sono target reali.

Disegna una box separata e stretta per gli oggetti isolati o grandi in primo
piano. Quando un frame contiene molte automobili o persone e annotare ogni
istanza sarebbe oneroso, usa una o poche box per coprire i gruppi: `car` per le
automobili e `person` per le persone. Dividi un gruppo soltanto quando una box
unica includerebbe molto spazio vuoto; non è necessario separare decine di
piccoli oggetti vicini. Il gruppo è un target privacy valido e non richiede
`ambiguous`.

Imposta `occluded` e `truncated` quando evidenti. Usa il tag immagine
`ambiguous` soltanto quando non puoi decidere in modo affidabile: il frame sarà
escluso da training ed evaluation finché il dubbio non viene risolto. Le box
precaricate sono suggerimenti dei modelli e non ground truth.

I 22 casi segnalati durante il controllo del viewer sono in un task separato:
partono dalle annotazioni correnti del dataset e non fanno parte del pilot da
200 immagini. Nei 20 casi indicati come probabile errore elimina i box che non
rappresentano target reali e conserva o aggiungi soltanto eventuali target
validi. Nei due casi indicati come dubbi (`possible_motorcycle_part` e
`negative_photograph`) usa `ambiguous` se l'immagine non consente una decisione
affidabile. I frame ambigui non entrano nel training v3.

```sh
dataset/commands/cvat-tasks.sh upload-open-images
dataset/commands/cvat-tasks.sh upload-phenocam
dataset/commands/cvat-tasks.sh upload-open-images-supplement
dataset/commands/cvat-tasks.sh upload-v3
dataset/commands/cvat-tasks.sh read-v3
dataset/commands/cvat-tasks.sh list
```

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

## 3. Negativi

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

## 4. Export e backup

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

## 5. Arresto sicuro

`dataset/commands/cvat-server.sh stop` ferma i container ma conserva database, utenti, task e
annotazioni nei volumi Docker. `dataset/commands/cvat-server.sh start` riprende lo stesso stato.

## 6. Esito effettivamente adottato

Il dataset materializzato non ha seguito la verifica indipendente descritta
sopra: il secondo revisore non era disponibile e il proprietario ha approvato
esplicitamente `single_reviewer_waiver`. La procedura standard resta documentata
per la riproduzione, ma l'artefatto corrente non deve essere presentato come
verificato indipendentemente.
