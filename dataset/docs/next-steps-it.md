# Prossimi passi del dataset pubblico

Stato di riferimento: 28 agosto 2026.

## Principio operativo

Il builder automatizza acquisizione, controlli riproducibili, screening,
deduplicazione, conteggi, sostituzioni e materializzazione. Le decisioni che
stabiliscono il ground truth restano umane: presenza o assenza di target,
correzione delle classi e disegno completo dei box.

## Sequenza

1. **Sostituzioni Open Images.** Emanuele verifica i 17 rimpiazzi nella pagina
   `resolution/openimages-a.html`.
2. **Sostituzioni PhenoCam.** Emanuele verifica i 5 rimpiazzi nella pagina
   `resolution/phenocam-a.html`.
3. **Seconda verifica PhenoCam definitiva.** Dopo l’import dei rimpiazzi, una
   persona diversa verifica in cieco tutti i 706 frame della nuova pagina
   `resolution/phenocam-b.html`.
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

Il primo tentativo ha trovato 17 target Open Images e 5 PhenoCam; il round B era
stato firmato dallo stesso revisore e non può valere come verifica indipendente.
Le decisioni sono state conservate per eliminare i falsi negativi. Terminati i
due export di sostituzione, il builder genera il round B definitivo.

Al termine del round definitivo l’import viene eseguito con:

```sh
dataset/finalize.sh import-final PHENOCAM_B_DEFINITIVO.csv
```

## Confine corrente

La riconciliazione positiva è completa: un solo frame non revisionato è stato
sostituito dopo il task 4 e tutte le soglie sono nuovamente soddisfatte. La
deduplicazione finale, la materializzazione e l’acceptance audit dipendono ora
soltanto dagli export effettivi delle tre revisioni negative. Non sono
sostituibili con conteggi o predizioni del modello corrente.
