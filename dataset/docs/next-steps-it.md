# Prossimi passi del dataset pubblico

Stato di riferimento: 28 agosto 2026.

## Principio operativo

Il builder automatizza acquisizione, controlli riproducibili, screening,
deduplicazione, conteggi, sostituzioni e materializzazione. Le decisioni che
stabiliscono il ground truth restano umane: presenza o assenza di target,
correzione delle classi e disegno completo dei box.

## Sequenza

1. **Revisione negativi Open Images.** Verifica i 50 frame della pagina
   `openimages-a.html` e scarica il CSV.
2. **Prima verifica PhenoCam.** Emanuele verifica i 371 nuovi negativi nella
   pagina `phenocam-a.html`; le altre 335 prime verifiche provengono già da CVAT.
3. **Seconda verifica PhenoCam.** Una persona diversa verifica in cieco tutti i
   706 frame della pagina `phenocam-b.html` e scarica il CSV.
4. **Import automatico.** Controlla identità, completezza, decisioni e revisori
   distinti con `dataset/finalize.sh import`.
5. **Deduplicazione congiunta finale.** Rigenera embedding e gruppi sui 2.000
   frame accettati e blocca duplicati o leakage.
6. **Materializzazione YOLO.** Produce full frame, crop approvati, label,
   `data.yaml`, manifest, licenze, statistiche e checksum.
7. **Acceptance audit.** Dichiara il dataset completo soltanto quando tutti i
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
dataset/finalize.sh prepare
```

`status` non modifica dati. `prepare-openimages-review` è idempotente rispetto a
uno screening completo e riprende un checkpoint compatibile. Se trova decisioni
umane nel pacchetto esistente, si ferma invece di sovrascriverle.

Terminati i tre export CSV:

```sh
dataset/finalize.sh import OPENIMAGES.csv PHENOCAM_A.csv PHENOCAM_B.csv
```

## Confine corrente

La riconciliazione positiva è completa: un solo frame non revisionato è stato
sostituito dopo il task 4 e tutte le soglie sono nuovamente soddisfatte. La
deduplicazione finale, la materializzazione e l’acceptance audit dipendono ora
soltanto dagli export effettivi delle tre revisioni negative. Non sono
sostituibili con conteggi o predizioni del modello corrente.
