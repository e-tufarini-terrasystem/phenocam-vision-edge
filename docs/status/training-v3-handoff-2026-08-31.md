# Handoff training v3 — 31 agosto 2026

## Stato attuale

Il lavoro è arrivato al gate umano obbligatorio. Inventario, split operativo,
screening baseline/v2, embedding, selezioni e bundle CVAT sono completati. Non
sono ancora stati importati i task v3 nel dataset e non è stato avviato alcun
training v3: queste operazioni devono attendere la conclusione della revisione
umana.

Nel progetto CVAT `Phenocam privacy detector v3` (ID 1) restano soltanto i due
task attivi e utili:

| Task | ID | Job | Immagini | Box proposte |
|---|---:|---|---:|---:|
| Representative clean | 9 | 30-32 | 120 | 1.019 |
| Informative clean | 10 | 33-35 | 120 | 1.115 |

- Task 9: <http://localhost:8080/tasks/9>
- Task 10: <http://localhost:8080/tasks/10>

Le box sono soltanto suggerimenti. Vanno eliminate quando palesemente errate e
vanno aggiunti tutti i target reali mancanti. Ogni oggetto richiede una box
separata: non usare una box unica per gruppi di macchine. Se non è possibile
separare oggetti minuscoli o sovrapposti in modo affidabile, il frame va marcato
`ambiguous`.

## Cosa è stato completato

### Modello v2 e dataset

- Modello ONNX v2 stabilizzato e incluso nel package, SHA-256
  `f62fd573fd6315fc34ddc3b0f22f4351be78f05db768babf6bca35b0c69d8962`.
- Viewer dataset migliorato e verificato su 2.000 immagini.
- Controllo manuale del dataset pubblico trasformato in una coda CVAT da 22
  immagini; la revisione finale conserva 4 box.
- Commit già presenti sul branch:
  `0bacc0f`, `dcca648`, `3b528c8`, `5e93ce2`, `544e168`.

### Dati operativi e screening

- 1.029 immagini interne inventariate senza modificare la sorgente.
- Split per sito-giorno senza leakage: 403 `operational_dev`, 385
  `operational_mining`, 241 `sealed_test`.
- Il test resta sigillato: zero immagini test lette durante embedding e
  selezione.
- Screening completato su 5.675 immagini pubbliche e 788 immagini interne con
  entrambi i modelli, senza fallimenti.

### Esito PhenoCam pubblico

Il pilot pubblico da 200 immagini è stato completato con `0/200` positivi. È
stato esportato e archiviato prima della rimozione da CVAT:

- COCO: `dataset/workspace/annotation/exports/v3-public-pilot-reviewed.coco.zip`;
- backup ripristinabile:
  `dataset/workspace/annotation/exports/v3-public-pilot-task-backup.zip`.

La causa principale era il ranking: soltanto 7 immagini avevano confidenza v2
`>=0,30`; 161 erano state selezionate per segnali deboli `crop_only`.

Una seconda analisi con il gate stretto trova 58 candidati con accordo fra i
due modelli; 2 erano già nel pilot negativo e ne restano 56. I dieci più forti
mostrano soprattutto veicoli reali ma molto lontani, spesso alti 7-15 pixel.
Non è stato creato un altro task: prima bisogna decidere esplicitamente se
questi oggetti piccoli ma riconoscibili siano target privacy o negativi
operativi. Dopo la decisione, l'eventuale nuovo pilot dovrebbe contenere solo
20 immagini ad alta confidenza.

### Preannotazioni interne clean

I task interni originali proponevano tutte le detection alla soglia `0,01`,
generando migliaia di falsi box. Sono stati sostituiti con bundle che accettano
una proposta soltanto quando:

- baseline e v2 rilevano entrambi l'oggetto con confidenza `>=0,30`;
- la classe è identica;
- l'IoU fra le due box è `>=0,50`.

Il filtro è in `dataset/builder/mining/suggestions.py`; l'orchestrazione CVAT
rimane sotto il limite di 200 linee produttive. Sono stati aggiunti il flag
`cvat-bundle --clean-suggestions`, la configurazione del gate e il comando
`dataset/commands/cvat-tasks.sh upload-v3-clean`.

## Task rimossi e recuperabilità

Sono stati eliminati da CVAT i task superati o già completati 2-8. Prima della
rimozione sono stati verificati i backup locali:

- task 7 e 8: export COCO e backup con le vecchie preannotazioni;
- task 5: export COCO delle 200 negative e backup completo;
- task 6: export COCO delle 22 immagini/4 box e backup completo;
- task 2-4: già esportati, importati nel dataset e coperti da backup completi.

Tutti gli archivi si trovano in
`dataset/workspace/annotation/exports/`, che è locale e ignorata da Git. La
lista CVAT è stata riletta dopo le eliminazioni e contiene soltanto i task 9 e
10. Le eliminazioni sono quindi recuperabili con `cvat-cli task
create-from-backup`.

## Verifiche eseguite

- Suite dataset: 37 test passati.
- Suite runtime: 237 test passati, 7 saltati per fixture opzionali assenti.
- Test specifico del gate clean incluso.
- Sintassi di `dataset/commands/cvat-tasks.sh`: valida.
- `git diff --check`: passato.
- Moduli di orchestrazione verificati sotto 200 linee produttive
  (`cvat.py`: 185; `selection.py`: 152).

## Prossimi passi

1. Completare i job 30-32 del task 9 e poi i job 33-35 del task 10.
2. Esportare e verificare entrambi i task completati prima di importarli.
3. Decidere la regola per i target piccoli/lontani prima di creare altre code
   PhenoCam.
4. Importare le revisioni v3, produrre il report preannotazione/ground truth e
   soltanto dopo proseguire con dataset derivato, training ed evaluation.

Non è stato eseguito alcun push.
