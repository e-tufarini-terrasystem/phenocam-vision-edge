<!--
Scopo: spiegare come installare, configurare ed eseguire il software.
Responsabilita: documentare il contratto operativo di CLI, batch, output ed export.
Contesto: traduce l'architettura in procedure ripetibili per operatore e manutentore.
-->

# Operativita

## Installazione runtime

Il deploy richiede Raspberry Pi OS a 64 bit, Python 3.11 o successivo e un
ambiente virtuale. Dalla radice del repository:

```sh
sudo apt update
sudo apt install --no-install-recommends python3-venv
python3 -m venv .venv
.venv/bin/python -m pip install --no-cache-dir -r requirements/runtime.txt
```

`--no-cache-dir` evita di conservare le wheel sulla microSD. Il requirements
seleziona un pin NumPy compatibile con la versione di Python e installa inoltre
ONNX Runtime e Pillow. Il modello `models/yolo26n.onnx` deve restare disponibile;
`models/yolo26n.pt` e gli script di export non sono necessari al runtime.

Su macOS Apple Silicon si puo creare allo stesso modo una virtualenv con Python
3.13 e installare `requirements/runtime.txt`, senza `sudo apt`.

## Comando singolo

Il modulo si esegue dalla radice del repository e richiede sempre tutte e tre
le opzioni:

```sh
.venv/bin/python -m phenocam \
  --input images/esempio.jpg \
  --output output/esempio.jpg \
  --model models/yolo26n.onnx
```

| Opzione | Validazione |
|---|---|
| `--input` | Deve essere un file locale esistente e decodificabile come immagine. |
| `--output` | La directory padre deve esistere; il percorso non puo essere una directory o la stessa identita del file di input. |
| `--model` | Deve essere un file locale esistente con estensione `.onnx`, senza distinzione tra maiuscole e minuscole. |

Il confronto tra input e output risolve percorsi equivalenti e link simbolici;
se l'output esiste gia, controlla anche l'identita reale dei file. Questo evita
che il salvataggio distrugga accidentalmente l'immagine sorgente.

La CLI non crea la directory di output. Un file di output distinto gia esistente
viene sovrascritto. Il comando non apre finestre grafiche e non accetta URL,
stream standard, video, webcam o directory.

## Configurazione delle classi

`phenocam/classes/configuration.py` elenca tutte le classi COCO per categoria:

```python
("vehicle", (
    ("bicycle", True),
    ("car", True),
    ("motorcycle", False),
    # ...
)),
```

Modificare esclusivamente il booleano associato alle classi desiderate. Non
cambiare nomi, ordine, categorie, parentesi o tipo dei contenitori e mantenere
almeno una classe su `True`. Il file viene validato e caricato a ogni comando.

Disabilitare una classe impedisce soltanto che le sue box vengano disegnate. Non
riduce il numero di chiamate ONNX, il tempo del modello o la memoria necessaria
alla preparazione dell'immagine.

## Thread CPU

Il default usa fino a quattro core. Per ridurre carico e temperatura, accettando
una latenza probabilmente maggiore:

```sh
YOLO_NUM_THREADS=2 .venv/bin/python -m phenocam \
  --input images/esempio.jpg \
  --output output/esempio.jpg \
  --model models/yolo26n.onnx
```

Sono validi soltanto `1`, `2`, `3` e `4`. Qualunque altro valore usa il default.
Le nove viste restano sequenziali: questa variabile controlla i thread interni
usati da ONNX Runtime, non crea nove inferenze concorrenti.

## Risultato e stato di uscita

In caso di successo il processo stampa una sola riga simile a:

```text
Execution time: 1.234 s
```

Il numero e tempo ONNX sommato, non wall time completo. L'immagine salvata
mantiene le dimensioni della sorgente dopo l'eventuale correzione EXIF.

| Stato | Significato |
|---:|---|
| `0` | Inferenza completata e output non vuoto scritto. |
| `1` | Percorso non valido, configurazione incompatibile, errore di inferenza o scrittura. |
| `2` | Sintassi CLI non valida, opzione mancante o argomento sconosciuto gestito da `argparse`. |

I diagnostici applicativi vengono scritti su standard error e omettono dettagli
interni. I principali sono:

```text
error: input image does not exist or is not a file
error: model does not exist or is not a file
error: model must be an ONNX file
error: output directory does not exist
error: output path must be a file
error: input and output paths must differ
error: class configuration is invalid
error: model classes are incompatible
error: inference failed
error: output image could not be written
```

## Elaborazione batch

`scripts/batch.sh` elabora i file immagine presenti direttamente in `images/`:

```sh
./scripts/batch.sh
```

Lo script:

1. individua la radice del progetto senza dipendere dalla directory corrente;
2. preferisce `.venv/bin/python`, altrimenti cerca `python3`;
3. crea `output/`;
4. considera JPG, JPEG, PNG, WEBP, BMP, TIF e TIFF senza distinzione tra
   maiuscole e minuscole;
5. invoca `.venv/bin/python -m phenocam` separatamente per ciascun file, usando
   sempre
   `models/yolo26n.onnx`;
6. conserva il nome originale nell'output;
7. continua dopo un errore individuale, ma termina con stato 1 se almeno una
   immagine fallisce.

Le sottodirectory non vengono visitate. Anche i file con estensione supportata
restano dati non affidabili: la decodifica effettiva avviene nella pipeline.

## Export opzionale

La rigenerazione del modello e un'attivita da workstation separata dal deploy:

```sh
python3 -m venv .venv-export
.venv-export/bin/python -m pip install -r requirements/export.txt
.venv-export/bin/python scripts/export/fp32.py
.venv-export/bin/python scripts/export/int8.py
```

`scripts/export/fp32.py` carica `models/yolo26n.pt` con Ultralytics e produce ONNX con opset 20.
`scripts/export/int8.py` richiede inoltre i dati di calibrazione `coco8.yaml` e
produce una variante quantizzata sperimentale. Il runtime documentato e testato
continua a usare `models/yolo26n.onnx`; sostituirlo richiede di verificare nuovamente
contratto, snapshot e prestazioni.

## Diagnosi essenziale

- Se il modello e rifiutato, controllare che sia end-to-end, detection, con un
  solo input float statico, un solo output a sei colonne e 80 classi COCO.
- Se la configurazione e rifiutata, ripristinare struttura e nomi di
  `phenocam/classes/configuration.py` e cambiare soltanto i booleani.
- Se la scrittura fallisce, controllare esistenza e permessi della directory
  padre e che l'estensione sia supportata da Pillow.
- Se il Pi rallenta durante esecuzioni ripetute, controllare temperatura,
  alimentazione e processi concorrenti; ridurre i thread limita il carico ma
  non garantisce una latenza inferiore.
