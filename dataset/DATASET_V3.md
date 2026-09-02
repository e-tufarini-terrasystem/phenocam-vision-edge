# Phenocam Vision dataset v3

Questo artifact mantiene la struttura YOLO del dataset v2, aggiunge una piccola
espansione pubblica PhenoCam revisionata ed è apribile direttamente con
`dataset/viewer.html`.

## Composizione

- `images/train` e `labels/train`: 2.018 immagini pubbliche, cioè le 2.000 del
  dataset v2 conservate senza modifiche e 18 positive PhenoCam revisionate;
- `images/operational_dev` e `labels/operational_dev`: 120 immagini interne
  revisionate per development operativo;
- `images/operational_mining` e `labels/operational_mining`: 120 immagini
  interne informative revisionate.

Il file `yolo-dataset.yaml` usa esclusivamente `images/train` per il training.
Gli split interni sono inclusi nello stesso artifact per consultazione,
development ed evaluation, ma non entrano automaticamente nel training.

In totale l'artifact contiene 2.258 immagini e 10.667 annotazioni: 4.781 nel
training e 5.886 nei due split operativi. Le 18 nuove immagini aggiungono 104
box umane (`car` 90, `truck` 12, `bus` 1, `person` 1). Le altre 22 immagini del
Task CVAT 11 restano nella riserva revisionata e non sono materializzate.

## Provenienza

I nomi delle immagini interne espongono sito, timestamp e hash breve, per
esempio:

```text
raspberrypi2.local--2025-10-30T121905--f4ab422e2908.jpg
sitets02--2026-08-08T061104--7f9b7bd389d0.jpg
```

Anche i nuovi nomi PhenoCam espongono sito e timestamp. Le 18 immagini
provengono da `bitterootvalley` (8), `nationalcapital` (8),
`borgocioffinorth` (1) e `snodgrass5` (1).

`metadata/source-images.csv` è il manifest canonico. Per ogni immagine conserva
identità e nome originali, sito, timestamp, gruppo, split, coorte, task CVAT,
revisori e checksum. I path assoluti della workstation non sono inclusi.

## Viewer

Aprire `dataset/viewer.html` e selezionare l'intera cartella dell'artifact v3.
Il viewer associa automaticamente ogni immagine alla label nello stesso split.
La ricerca per nome consente di filtrare
immediatamente `raspberrypi2.local`, `sitets02`, `open-images` o `phenocam`.

## Metadata

- `source-images.csv`: una riga per ciascun frame;
- `source-annotations.jsonl`: annotazioni sorgente e compilate;
- `dataset-statistics.json`: conteggi complessivi e per split;
- `acceptance-audit.json`: confini fra training e dati operativi;
- `build.json`: hash degli input usati dal materializzatore;
- `*-images.txt`: liste deterministiche per split;
- `checksums.sha256`: integrità di ogni file distribuito, escluso se stesso.

## Integrità

Dalla cartella dell'artifact eseguire:

```sh
shasum -a 256 -c metadata/checksums.sha256
```

Le immagini operative sono dati privati e non devono essere redistribuite.
