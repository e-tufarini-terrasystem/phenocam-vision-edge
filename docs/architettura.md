<!--
Scopo: descrivere la struttura statica e il flusso complessivo del software.
Responsabilita: definire componenti, preprocessing, errori e invarianti tra moduli.
Contesto: collega l'interfaccia CLI alla pipeline dettagliata in inferenza.md.
-->

# Architettura

## Modello del sistema

Il software e un processo sincrono e headless che trasforma un'immagine locale
in una nuova immagine annotata. Ogni invocazione gestisce esattamente un input,
un modello e un output. Non mantiene stato tra processi e non usa rete, database,
code, worker o inferenze parallele.

```mermaid
flowchart LR
    CLI[Argomenti CLI] --> Validazione[Validazione percorsi]
    Config[classes.py] --> Selezione[Selezione classi]
    Modello[Modello ONNX] --> Sessione[Sessione ONNX CPU]
    Validazione --> Coordinamento[Transazione di inferenza]
    Selezione --> Coordinamento
    Sessione --> Coordinamento
    Immagine[Immagine locale] --> Coordinamento
    Coordinamento -->|source RGB originale| Gamma[Gamma adattiva opzionale]
    Gamma -->|model image| Viste[1 vista completa + 8 crop]
    Viste --> Detection[Detection globali]
    Detection --> NMS[NMS per classe]
    NMS --> Filtro[Filtro classi abilitate]
    Filtro --> Output[Immagine annotata]
    Coordinamento -->|source originale| Output
```

L'invariante transazionale principale e: **l'output viene scritto soltanto se
tutte le nove viste sono state preparate ed eseguite correttamente**. Un errore
in una vista interrompe il comando; non viene prodotto un risultato parziale.

## Responsabilita dei file

| File | Responsabilita unica |
|---|---|
| `run.py` | Definisce il confine del processo, traduce gli errori in messaggi pubblici e restituisce lo stato di uscita. |
| `arguments.py` | Costruisce la CLI e valida i tre percorsi ricevuti. |
| `classes.py` | Contiene l'inventario COCO fisso e i soli booleani configurabili dall'operatore. |
| `selection.py` | Valida configurazione e metadata delle classi, quindi risolve i nomi abilitati negli ID del modello. |
| `inference/__init__.py` | Coordina l'intera transazione multi-vista e somma il tempo delle chiamate ONNX. |
| `inference/runtime.py` | Configura ONNX Runtime, verifica il contratto statico del modello ed esegue un tensore. |
| `inference/views.py` | Decodifica l'immagine e produce viste normalizzate con geometria inversa. |
| `inference/gamma.py` | Valida la configurazione gamma e prepara l'unica model image usata dalle nove viste. |
| `inference/detections.py` | Valida le righe del modello, ricostruisce coordinate globali ed elimina duplicati. |
| `inference/output.py` | Disegna le detection selezionate e salva un file di output verificato. |
| `inference/errors.py` | Definisce gli errori del dominio inferenza esposti al confine CLI. |
| `batch.sh` | Applica il comando singolo ai file supportati presenti direttamente in `images/`. |
| `export_onnx.py` | Rigenera il modello ONNX FP32 su workstation. |
| `export_onnx_int8.py` | Esegue l'export ONNX INT8 sperimentale con dati di calibrazione. |

## Sequenza di una richiesta

```mermaid
sequenceDiagram
    participant U as Operatore
    participant R as run.py
    participant S as selection.py
    participant O as ONNX Runtime
    participant P as Pipeline inferenza
    participant F as File output

    U->>R: --input, --output, --model
    R->>R: valida i percorsi
    R->>S: carica e valida classes.py
    R->>O: crea sessione CPU
    O-->>R: input, output e metadata
    R->>S: risolve nomi abilitati in ID
    R->>P: carica la source RGB normalizzata EXIF
    P->>P: valida gamma e sceglie una model image
    loop vista completa e otto crop
        R->>P: prepara tensore
        P->>O: session.run()
        O-->>P: righe [x1,y1,x2,y2,conf,id]
        P->>P: valida e converte in coordinate globali
    end
    P->>P: NMS per classe
    P->>F: disegna gli ID abilitati sulla source originale
    F-->>R: file regolare non vuoto
    R-->>U: Execution time e stato 0
```

## Confini di fiducia

Il sistema considera non affidabili tutti i dati esterni, anche quando sono
file locali:

- la CLI non puo assumere che i percorsi esistano o identifichino oggetti del
  tipo atteso;
- `classes.py` e importato dinamicamente, ma struttura, ordine, nomi e tipi
  devono coincidere con l'inventario canonico;
- metadata, descrittori dei tensori e valori restituiti dal modello vengono
  controllati prima dell'uso;
- l'immagine deve essere decodificabile e avere dimensioni positive;
- le sei costanti gamma devono avere tipi, valori e ordine validi anche quando
  la funzione e disabilitata;
- ogni detection deve contenere sei numeri finiti e produrre un rettangolo con
  area positiva;
- il salvataggio e riuscito solo se il percorso finale e un file regolare non
  vuoto.

Gli errori delle librerie non attraversano il confine pubblico con dettagli
interni, stack trace o percorsi sensibili. Il comando espone messaggi fissi per
configurazione classi, configurazione gamma, compatibilita del modello,
inferenza e scrittura. `GammaConfigurationError` identifica soltanto costanti
gamma versionate non valide.

## Stato e proprieta

Gli oggetti `Arguments`, `View` e `Detection` sono dataclass immutabili. Questo
rende esplicito che percorsi validati, geometria di una vista e detection
normalizzate non vengono modificati dopo la costruzione.

La transazione conserva la source RGB originale per dimensioni e rendering. La
gamma puo produrre una distinta model image, condivisa da tutte le nove viste;
non modifica mai la source. La sola mutazione intenzionale del dato applicativo
e il disegno sulla source immediatamente prima del salvataggio. La sessione ONNX
viene creata una volta per processo e riutilizzata in sequenza per tutte le viste.

## Dipendenze

Il percorso di runtime ha tre dipendenze dirette:

- Pillow possiede decodifica, orientamento EXIF, resize, disegno e scrittura;
- NumPy possiede il tensore di input e la validazione dell'array di output;
- ONNX Runtime esegue il grafo esclusivamente con `CPUExecutionProvider`.

Ultralytics e una dipendenza separata, usata soltanto dagli script di export.
PyTorch e OpenCV non fanno parte del percorso di deploy.
