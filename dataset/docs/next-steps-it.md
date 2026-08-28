# Prossimi passi del dataset pubblico

Stato di riferimento: 28 agosto 2026.

## Principio operativo

Il builder automatizza acquisizione, controlli riproducibili, screening,
deduplicazione, conteggi, sostituzioni e materializzazione. Le decisioni che
stabiliscono il ground truth restano umane: presenza o assenza di target,
correzione delle classi e disegno completo dei box.

## Sequenza

1. **Preflight automatico.** Verifica modelli, selezioni 900/1.100, percorsi
   locali, calibrazione SSCD e revisione del duplicato Open Images.
2. **Screening Open Images riprendibile.** Analizza 900 immagini con checkpoint
   ogni 25 e conserva le colonne `baseline_*` nel pacchetto di revisione.
3. **Revisione umana Open Images.** Controlla i 250 frame obbligatori; le
   predizioni automatiche servono soltanto per ordinare il lavoro.
4. **Annotazione umana PhenoCam.** Annota i 350 candidati positivi e verifica i
   750 candidati negativi; i negativi accettati richiedono un secondo revisore.
5. **Import e audit automatici.** Valida identità, decisioni, revisori, box,
   classi, completezza, quote e minimi di istanza. Produce una coda deterministica
   di sostituzioni per i frame respinti.
6. **Deduplicazione congiunta finale.** Rigenera embedding e gruppi sui 2.000
   frame accettati e blocca duplicati o leakage.
7. **Materializzazione YOLO.** Produce full frame, crop approvati, label,
   `data.yaml`, manifest, licenze, statistiche e checksum.
8. **Acceptance audit.** Dichiara il dataset completo soltanto quando tutti i
   gate della specifica passano. La validation operativa interna resta separata
   e può rimanere `pending`.

## Comandi correnti

```sh
dataset/workflow.sh setup
dataset/workflow.sh status
dataset/workflow.sh preflight
dataset/workflow.sh test
dataset/workflow.sh prepare-openimages-review
dataset/workflow.sh annotation-bundles
```

`status` non modifica dati. `prepare-openimages-review` è idempotente rispetto a
uno screening completo e riprende un checkpoint compatibile. Se trova decisioni
umane nel pacchetto esistente, si ferma invece di sovrascriverle.

## Confine corrente

Il passo 5 dispone già di importatori con controlli su identità, completezza,
classi, box, revisori e checksum. I passi 6–8 richiedono invece i risultati
effettivi delle annotazioni: la logica finale di sostituzione, deduplicazione,
materializzazione e acceptance audit verrà completata e verificata su quegli
output prima di iniziare il training. Non è sostituibile con conteggi o
predizioni del modello corrente.
