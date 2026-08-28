# Guida operativa all’annotazione

## 1. Primo accesso CVAT

CVAT Community `v2.71.0` è disponibile soltanto in locale su
`http://localhost:8080`. I dati non vengono caricati su un servizio cloud.

1. Avvia Docker Desktop.
2. Esegui `dataset/cvat.sh start`.
3. La prima volta esegui `dataset/cvat.sh create-superuser` e scegli localmente
   username, email e password.
4. Esegui `dataset/cvat.sh open` e accedi.
5. Nelle impostazioni utente crea un Personal Access Token.
6. Esegui `dataset/cvat-tasks.sh profile`: incolla il token quando richiesto. Il
   profilo viene salvato dal client CVAT con permessi locali restrittivi e non
   entra nel repository.

## 2. Creazione dei task

```sh
dataset/cvat-tasks.sh upload-openimages
dataset/cvat-tasks.sh upload-phenocam
dataset/cvat-tasks.sh list
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

Al termine assegna il job al revisore, portalo allo stage di validazione e risolvi
gli eventuali problemi prima di marcarlo completato.

## 3. Negativi

Apri direttamente in Chrome con:

```sh
dataset/negative-reviews.sh open-openimages
dataset/negative-reviews.sh open-phenocam-a
dataset/negative-reviews.sh open-phenocam-b
```

Il round B PhenoCam deve essere completato da una persona diversa senza vedere
l’export del round A. Scegli `Negativo confermato` soltanto quando non è visibile
alcun target vivo delle sei classi. `Target presente` e `Incerto` causano revisione
o sostituzione; non sono errori da forzare ad accettazione.

## 4. Export e backup

Dopo aver completato e revisionato un task, recupera il suo ID con
`dataset/cvat-tasks.sh list`, quindi:

```sh
dataset/cvat-tasks.sh export ID openimages
dataset/cvat-tasks.sh export ID phenocam
```

Per esportare, fare il backup, validare e importare in un solo passaggio usa:

```sh
dataset/cvat-tasks.sh finish ID openimages NOME_ANNOTATORE NOME_REVISORE
dataset/cvat-tasks.sh finish ID phenocam NOME_ANNOTATORE NOME_REVISORE
```

Gli export CSV delle tre pagine dei negativi vengono uniti e controllati con:

```sh
dataset/negative-reviews.sh import OPENIMAGES.csv PHENOCAM_A.csv PHENOCAM_B.csv
```

Ogni task produce sia il COCO revisionato sia un backup completo. Non modificare
manualmente gli export. Il builder registra SHA-256, annotatore, revisore e
timestamp durante l’importazione e rifiuta pacchetti incompleti o non canonici.

## 5. Arresto sicuro

`dataset/cvat.sh stop` ferma i container ma conserva database, utenti, task e
annotazioni nei volumi Docker. `dataset/cvat.sh start` riprende lo stesso stato.
