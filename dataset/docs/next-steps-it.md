# Stato conclusivo del dataset pubblico

Stato di riferimento: 28 agosto 2026.

## Principio operativo

Il builder automatizza acquisizione, controlli riproducibili, screening,
deduplicazione, conteggi, sostituzioni e materializzazione. Le decisioni che
stabiliscono il ground truth restano umane: presenza o assenza di target,
correzione delle classi e disegno completo dei box.

## Esito corrente

La revisione manuale è conclusa. Non essendo disponibile un secondo revisore,
il proprietario ha approvato esplicitamente l'uso del singolo passaggio di
Emanuele. La deroga è registrata come `single_reviewer_waiver` e il dataset non
deve essere descritto come verificato indipendentemente.

La composizione finale è stata importata e materializzata con
`dataset/commands/dataset-finalization.sh build` in `dataset/training-dataset/`. L'audit ha
stato `completed_with_single_reviewer_waiver`; la validation operativa interna
rimane separata e `pending`.

## Sequenza eseguita

1. **Sostituzioni Open Images.** Emanuele ha verificato i 17 rimpiazzi nella pagina
   `resolution/open-images-a.html`.
2. **Sostituzioni PhenoCam.** Emanuele ha verificato i 5 rimpiazzi nella pagina
   `resolution/phenocam-a.html`.
3. **Seconda verifica PhenoCam definitiva.** Non eseguita; sostituita dalla
   deroga esplicita `single_reviewer_waiver`.
4. **Import automatico.** Ha controllato identità, completezza e decisioni con
   `dataset/commands/dataset-finalization.sh accept-single-review`.
5. **Deduplicazione congiunta finale.** Rigenera embedding e gruppi sui 2.000
   frame accettati e blocca duplicati o leakage.
6. **Materializzazione YOLO.** Ha prodotto immagini JPEG compilate, label,
   `yolo-dataset.yaml`, metadati, licenze, statistiche e checksum.
7. **Acceptance audit.** Completato con tutti i gate automatici superati e la
   limitazione della singola revisione visibile. La validation operativa interna
   resta separata e `pending`.

## Comandi di riproduzione

```sh
dataset/commands/dataset-builder.sh setup
dataset/commands/dataset-builder.sh status
dataset/commands/dataset-builder.sh preflight
dataset/commands/dataset-builder.sh test
dataset/commands/dataset-builder.sh prepare-open-images-review
dataset/commands/dataset-builder.sh prepare-open-images-supplement
dataset/commands/dataset-builder.sh annotation-bundles
dataset/commands/dataset-finalization.sh prepare
dataset/commands/dataset-finalization.sh accept-single-review
dataset/commands/dataset-finalization.sh build
```

`status` non modifica dati. `prepare-open-images-review` è idempotente rispetto a
uno screening completo e riprende un checkpoint compatibile. Se trova decisioni
umane nel pacchetto esistente, si ferma invece di sovrascriverle.

Il primo tentativo ha trovato 17 target Open Images e 5 PhenoCam; il round B era
stato firmato dallo stesso revisore e non può valere come verifica indipendente.
Le decisioni sono state conservate per eliminare i falsi negativi. Terminati i
due export di sostituzione, il builder genera il round B definitivo.

Con un secondo revisore, il percorso standard alternativo resta:

```sh
dataset/commands/dataset-finalization.sh import-final PHENOCAM_B_DEFINITIVO.csv
```

## Confine corrente

Il dataset pubblico di training è materializzato e riproducibile. Non costituisce
una validazione del modello sul dominio operativo interno; quella fase resta
`pending`. La deroga sulla seconda revisione impedisce inoltre di descrivere i
negativi come verificati indipendentemente.
