<!--
Scopo: spiegare come installare, configurare ed eseguire il software.
Responsabilita: documentare CLI, batch, due output, metadata opzionali ed export.
Contesto: traduce output, sostituzione metadata e pairing batch in procedure ripetibili.
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

Il modulo si esegue dalla radice del repository. `--input` e `--model` sono
obbligatori; occorre richiedere almeno uno dei due output:

```sh
.venv/bin/python -m phenocam \
  --input input/esempio.jpg \
  --annotated-output output/esempio_annotated.jpg \
  --model models/yolo26n.onnx
```

Solo privacy, oppure entrambi con una singola inferenza:

```sh
.venv/bin/python -m phenocam \
  --input input/esempio.jpg \
  --privacy-output output/esempio_privacy.jpg \
  --model models/yolo26n.onnx

.venv/bin/python -m phenocam \
  --input input/esempio.jpg \
  --annotated-output output/esempio_annotated.jpg \
  --privacy-output output/esempio_privacy.jpg \
  --meta input/esempio.meta \
  --model models/yolo26n.onnx
```

| Opzione | Validazione |
|---|---|
| `--input` | Deve essere un file locale esistente e decodificabile come immagine. |
| `--annotated-output` | Opzionale; la directory padre deve esistere e il percorso deve identificare un file distinto. |
| `--privacy-output` | Opzionale; valgono gli stessi vincoli dell'annotato. |
| `--model` | Deve essere un file locale esistente con estensione `.onnx`, senza distinzione tra maiuscole e minuscole. |
| `--meta` | Opzionale; deve essere un file regolare esistente, non simbolico, con suffisso `.meta` case-insensitive e distinto da input, modello e output. |

`--output` non esiste piu e viene rifiutato da `argparse`. Il confronto risolve
percorsi equivalenti e link simbolici; per file esistenti controlla anche gli
hard link. Ogni output deve essere distinto dall'input e, quando sono richiesti
entrambi, i due output devono essere distinti tra loro.

La CLI non crea directory. Un output esistente viene sovrascritto. Con entrambi,
l'annotato viene salvato e verificato prima del privacy; se il secondo fallisce,
il primo resta presente. Il comando non apre finestre e non accetta URL, stream
standard, video, webcam o directory.

## Metadata delle detection

Il file indicato con `--meta` deve essere creato dall'operatore: il comando non
lo crea. Dopo la scrittura e verifica di tutti gli output richiesti, ogni vecchia
sezione esatta lowercase `[detection]` viene rimossa e una sola sezione corrente
viene aggiunta atomicamente in fondo, preservando gli altri byte:

```text
[detection]
detected=true|false
software_name=phenocam-detection
software_version=1.0.0
model_id=yolo26n
model_version=1.0.0
annotated_image=<percorso CLI o vuoto>
privacy_image=<percorso CLI o vuoto>
classes=<nomi rilevati separati da virgola o vuoto>
<class-key>_count=<intero positivo, solo classi rilevate>
total_count=<somma, oppure 0>
```

I conteggi usano soltanto detection finali dopo soppressione globale e filtro
delle classi abilitate; classi e campi conteggio seguono l'ordine COCO. Con zero
detection selezionate si ottengono `detected=false`, `classes=` e
`total_count=0`, senza campi `*_count`. I percorsi mantengono esattamente la
spelling CLI. Esecuzioni ripetute sostituiscono il risultato, senza storico.

Se il commit metadata fallisce, gli output completati restano presenti e il
precedente file metadata rimane disponibile. Il comando termina con stato `1`
senza stampare contenuto, percorso temporaneo, eccezione o stack trace.

## Configurazione delle classi

`phenocam/classes/configuration.py` elenca tutte le classi COCO per categoria:

```python
("vehicle", (
    ("bicycle", True),
    ("car", True),
    ("motorcycle", True),
    # ...
)),
```

La configurazione versionata abilita `person`, `bicycle`, `car`, `motorcycle`, `bus` e `truck`.
Modificare soltanto i booleani, mantenendo invariati nomi, ordine, categorie e
tuple; almeno una classe deve restare `True`. Il file viene validato a ogni comando.

La stessa selezione controlla box/testo dell'annotato e regioni sfocate del
privacy. Il filtro avviene dopo l'inferenza: disabilitare una classe non riduce
le sedici chiamate ONNX, il tempo del modello o la memoria. Una configurazione
valida con zero detection selezionate scrive comunque gli output invariati;
disabilitare tutte le classi rende invece la configurazione invalida.

Il privacy non contiene box, nomi o confidenze. Usa rettangoli: margine 10% per
lato, clipping all'immagine e Gaussian blur con raggio pari al massimo tra 8 px
e il 10% del lato corto della regione. Non usa segmentazione o maschere.

## Thread CPU

Il default usa fino a quattro core. Per ridurre carico e temperatura, accettando
una latenza probabilmente maggiore:

```sh
YOLO_NUM_THREADS=2 .venv/bin/python -m phenocam \
  --input input/esempio.jpg \
  --annotated-output output/esempio_annotated.jpg \
  --model models/yolo26n.onnx
```

Sono validi soltanto `1`, `2`, `3` e `4`. Qualunque altro valore usa il default.
Le sedici viste restano sequenziali: `YOLO_NUM_THREADS` controlla i thread
interni usati da ONNX Runtime, non crea sedici inferenze concorrenti.

## Risultato e stato di uscita

In caso di successo il processo stampa una sola riga simile a:

```text
Execution time: 1.234 s
```

Il numero e tempo ONNX sommato, non wall time completo. L'immagine salvata
mantiene le dimensioni della sorgente dopo l'eventuale correzione EXIF.

| Stato | Significato |
|---:|---|
| `0` | Inferenza, output richiesti ed eventuale metadata completati. |
| `1` | Percorso non valido o errore di configurazione, inferenza, output o metadata. |
| `2` | Sintassi CLI non valida, opzione mancante o argomento sconosciuto gestito da `argparse`. |

I diagnostici applicativi vengono scritti su standard error e omettono dettagli
interni. I principali sono:

```text
error: input image does not exist or is not a file
error: model does not exist or is not a file
error: model must be an ONNX file
error: at least one output path is required
error: output directory does not exist
error: output path must be a file
error: input and output paths must differ
error: output paths must differ
error: metadata file does not exist or is not a file
error: metadata file must use the .meta extension
error: output path cannot be stored in metadata
error: metadata path must differ from input, model, and output paths
error: class configuration is invalid
error: model classes are incompatible
error: inference failed
error: output image could not be written
error: metadata file could not be updated
```

## Elaborazione batch

`scripts/batch.sh` elabora i file immagine presenti direttamente in `input/`:

```sh
./scripts/batch.sh
```

Lo script:

1. individua la radice del progetto senza dipendere dalla directory corrente;
2. preferisce `.venv/bin/python`, altrimenti cerca `python3`;
3. crea `output/`;
4. considera JPG, JPEG, PNG, WEBP, BMP, TIF e TIFF senza distinzione tra
   maiuscole e minuscole;
5. invoca `.venv/bin/python -m phenocam` una volta per ciascun file, chiedendo
   entrambi gli output con `models/yolo26n.onnx`;
6. genera `<stem>_annotated.<ext>` e `<stem>_privacy.<ext>` in `output/`,
   preservando l'estensione originale;
7. passa `input/<stem>.meta` tramite `--meta` solo quando e un file regolare non
   simbolico; un match mancante non viene creato e non e un errore;
8. continua dopo un errore individuale, incluso il commit metadata, ma termina
   con stato 1 se almeno una
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
contratto, integrazione reale e prestazioni.

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
