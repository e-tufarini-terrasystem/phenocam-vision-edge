# Prossimi passi del dataset pubblico

Stato di riferimento: 28 agosto 2026.

## Principio operativo

Il builder automatizza acquisizione, controlli riproducibili, screening,
deduplicazione, conteggi, sostituzioni e materializzazione. Le decisioni che
stabiliscono il ground truth restano umane: presenza o assenza di target,
correzione delle classi e disegno completo dei box.

## Sequenza

1. **Audit supplementare CVAT.** Controlla il task 4: 38 immagini e 126 box,
   correggendo classi, estensione e completezza del frame; non usare track.
2. **Import e audit automatici.** Valida identità, decisioni, revisori, box,
   classi, completezza, quote e minimi di istanza. Produce una coda deterministica
   di sostituzioni per i frame respinti.
3. **Revisione negativi.** Verifica i negativi Open Images e PhenoCam; i
   negativi PhenoCam accettati richiedono due revisori indipendenti.
4. **Deduplicazione congiunta finale.** Rigenera embedding e gruppi sui 2.000
   frame accettati e blocca duplicati o leakage.
5. **Materializzazione YOLO.** Produce full frame, crop approvati, label,
   `data.yaml`, manifest, licenze, statistiche e checksum.
6. **Acceptance audit.** Dichiara il dataset completo soltanto quando tutti i
   gate della specifica passano. La validation operativa interna resta separata
   e può rimanere `pending`.

## Comandi correnti

```sh
dataset/workflow.sh setup
dataset/workflow.sh status
dataset/workflow.sh preflight
dataset/workflow.sh test
dataset/workflow.sh prepare-openimages-review
dataset/workflow.sh prepare-openimages-supplement
dataset/workflow.sh annotation-bundles
```

`status` non modifica dati. `prepare-openimages-review` è idempotente rispetto a
uno screening completo e riprende un checkpoint compatibile. Se trova decisioni
umane nel pacchetto esistente, si ferma invece di sovrascriverle.

## Confine corrente

L’import dispone già di controlli su identità, completezza, classi, box,
revisori e checksum. I passi finali richiedono invece i risultati
effettivi delle revisioni: la logica finale di sostituzione, deduplicazione,
materializzazione e acceptance audit verrà completata e verificata su quegli
output prima di iniziare il training. Non è sostituibile con conteggi o
predizioni del modello corrente.
