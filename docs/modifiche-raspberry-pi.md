<!--
Scopo: preservare lo storico degli adattamenti e delle misure Raspberry Pi.
Responsabilita: registrare interventi, risultati e contesto hardware originario.
Contesto: integra i documenti tecnici con l'evidenza storica del deploy Raspberry Pi.
-->

# Modifiche per Raspberry Pi 3

## Esito

Il progetto è installabile ed eseguibile sulla configurazione richiesta. Il
test è stato svolto direttamente su una board `aarch64` con quattro Cortex-A53
a 1,4 GHz, 905 MiB di RAM utilizzabile, Python 3.13.5 e filesystem della
microSD da 6,9 GiB.

Il modello non è stato sostituito né quantizzato: l'inferenza usa il file
originale `models/yolo26n.onnx` da 9.941.954 byte, input 640x640 e uscita end-to-end
`[1, 300, 6]`.

## Interventi applicati

- Rimosso Ultralytics dal percorso di inferenza sulla board.
- Sostituito il solo wrapper runtime con ONNX Runtime CPU, mantenendo il grafo
  YOLO26n ONNX originale.
- Implementati letterbox RGB, normalizzazione, conversione NCHW e annotazione
  con NumPy e Pillow, senza OpenCV.
- Aggiunte una vista completa e otto viste ritagliate adattive con overlap
  nominale del 20% (`4×2` per immagini orizzontali/quadrate, `2×4` per verticali).
- Riportate le detection nello spazio globale, aggregate tutte le classi e
  applicata NMS per classe con IoU 0,50 prima della selezione finale di
  `phenocam/classes/configuration.py`.
- Conservati CLI, validazione dei percorsi, configurazione delle classi COCO e
  messaggi/exit status esistenti.
- Aggiunta validazione del contratto del modello: singolo input float,
  metadata `detect`/`end2end`, 80 classi COCO e uscita a sei colonne.
- Configurato ONNX Runtime in modalità sequenziale, con arena CPU e memory
  pattern disabilitati per contenere la RAM.
- Impostati quattro thread CPU come default; `YOLO_NUM_THREADS=1..4` permette
  di ridurre il carico.
- Separate le dipendenze di deploy (`requirements/runtime.txt`) da Ultralytics,
  necessario solo per un eventuale export (`requirements/export.txt`).
- Aggiornati README e `.gitignore`; la virtualenv non viene inclusa nel
  progetto versionato.

## Dipendenze runtime minime testate

| Dipendenza diretta | Versione | Funzione |
|---|---:|---|
| NumPy | 2.5.1 | Preparazione tensore e gestione output |
| ONNX Runtime | 1.27.0 | Esecuzione CPU del modello ONNX |
| Pillow | 12.3.0 | Decodifica, resize, disegno e salvataggio |

Le dipendenze transitive installate automaticamente da ONNX Runtime sono
`flatbuffers`, `packaging` e `protobuf`. `pip check` ha restituito
`No broken requirements found`.

Ultralytics, PyTorch e OpenCV non sono installati nella virtualenv di deploy.
La virtualenv misura 157.568 KiB (circa 154 MiB); repository, ambiente, modelli,
immagini e output di test occupano complessivamente circa 187 MiB.

## Regressione multi-vista sulle immagini

Le sei immagini sono elaborate con il modello reale e producono JPEG temporanei
4608x2592 non vuoti. La configurazione classi attiva `person` e `car`; ogni
conteggio multi-vista deve coincidere esattamente con lo snapshot approvato ed
essere maggiore della corrispondente baseline a vista completa.

| File | Person baseline | Car baseline | Person snapshot | Car snapshot | Esito | ONNX Pi | Wall Pi |
|---|---:|---:|---:|---:|---|---|---|
| `raspberrypi2.local_2025-11-19_121905.jpg` | 0 | 7 | 3 | 23 | PASS | not verified | not verified |
| `raspberrypi2.local_2025-11-19_141905.jpg` | 0 | 4 | 1 | 25 | PASS | not verified | not verified |
| `raspberrypi2.local_2025-11-19_151905.jpg` | 2 | 5 | 4 | 28 | PASS | not verified | not verified |
| `raspberrypi2.local_2025-12-17_131905.jpg` | 0 | 10 | 3 | 32 | PASS | not verified | not verified |
| `raspberrypi2.local_2025-12-18_141905.jpg` | 1 | 9 | 4 | 36 | PASS | not verified | not verified |
| `raspberrypi2.local_2025-12-19_124905.jpg` | 1 | 17 | 3 | 39 | PASS | not verified | not verified |

I conteggi di riferimento sono uno snapshot di regressione, non una ground truth né una misura di accuracy, precision, recall o mAP.
Lo snapshot non dimostra che le singole box siano corrette.

not verified — external verification: reference Raspberry Pi 3 is unavailable.
Di conseguenza il tempo ONNX sommato e il wall time completo multi-vista non
sono riportati come misure Raspberry Pi; il limite di 15 secondi resta un
criterio esterno non verificato.

Il benchmark precedente a vista singola, su due immagini e tre esecuzioni per
immagine, aveva misurato 0,848 s medi per `session.run()` e 3,803-4,048 s per il
comando completo. Questi valori sono soltanto contesto storico e non descrivono
la pipeline multi-vista corrente.

## Risorse e compatibilità

- Picco storico a vista singola: 214.224 KiB RSS, circa 209 MiB; il valore
  multi-vista non è stato rimisurato sulla board.
- Swap del processo a fine benchmark: 0 KiB.
- RAM fisica rilevata: 926.816 KiB (905 MiB).
- Filesystem microSD: 7.314.300.928 byte; al termine 3.267.219.456 byte liberi
  (circa 3,04 GiB).
- Modello ONNX: circa 9,5 MiB; virtualenv: circa 154 MiB.

RAM e microSD erano sufficienti per l'esecuzione a vista singola testata. La
pipeline corrente mantiene le nove viste sequenziali, ma il nuovo picco RSS non
è verificato sulla board; il benchmark storico non ha usato swap.

## Verifiche finali

- Compilazione Python di tutti i moduli riuscita.
- Installazione da `requirements/runtime.txt` riuscita.
- Integrità dipendenze con `pip check` riuscita.
- Tutti i sei output di regressione aperti e verificati con Pillow come JPEG
  4608x2592; snapshot esatti e assenza indipendente di duplicati IoU `>=0,50`.
- Suite locale completa: 115 test superati senza errori o fallimenti.
- Verifica esterna del wall time multi-vista su Raspberry Pi 3: non verificata.
- Confronto thread storico a vista singola: 1 thread 1,931 s; 2 thread 1,209 s;
  4 thread 0,874 s. Quattro thread restano il default configurato.
