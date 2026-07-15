<!--
Scopo: descrivere come il comportamento del software viene verificato.
Responsabilita: collegare invarianti, test automatici, snapshot e limiti delle prove.
Contesto: separa le garanzie riproducibili dalle misure dipendenti dalla piattaforma.
-->

# Verifica

## Esecuzione della suite

I test usano `unittest` e si eseguono dalla radice del repository:

```sh
.venv/bin/python -m unittest discover -s tests
```

Il gate locale completo aggiunge compilazione, integrita delle dipendenze,
contratto del comando, sintassi batch e hash degli asset:

```sh
.venv/bin/python -m compileall -q phenocam scripts tests
.venv/bin/python -m pip check
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m phenocam --help
sh -n scripts/batch.sh
test -x scripts/batch.sh
test -d input
shasum -a 256 models/yolo26n.onnx models/yolo26n.pt
shasum -a 256 requirements/runtime.txt requirements/export.txt
```

Il gate strutturale verifica inoltre `input/`, l'assenza della precedente
directory locale e dei due file di preprocessing rimossi, e l'assenza di
riferimenti obsoleti nella documentazione e negli import, senza introdurre
alias di compatibilita.

La maggior parte della suite usa file temporanei, immagini sintetiche e mock.
In questo modo verifica i contratti senza modificare
`phenocam/classes/configuration.py`, senza scrivere
nelle directory operative e, dove non serve, senza caricare ONNX Runtime.

## Copertura per confine

| Test | Contratto principale |
|---|---|
| `tests/test_arguments.py` | Opzioni richieste, tipi di percorso, directory padre e identita distinta tra input e output. |
| `tests/test_selection.py` | Inventario COCO esatto, soli booleani modificabili, almeno una classe e metadata del modello completi. |
| `tests/test_runtime.py` | Thread, opzioni della sessione, contratto tensoriale e validazione dell'output dinamico. |
| `tests/test_views.py` | EXIF, RGB, letterbox, forma del tensore, geometria adattiva, copertura e ordine delle viste. |
| `tests/test_detections.py` | Soglia, valori non validi, conversione globale, clipping, IoU, NMS per classe e pareggi deterministici. |
| `tests/test_pipeline.py` | Identita della source tra caricamento, nove viste e rendering, tempi e assenza di output parziale. |
| `tests/test_command.py` | Messaggi pubblici, separazione stdout/stderr e stati del processo. |
| `tests/test_reference_images.py` | Output JPEG reali, conteggi snapshot e assenza di duplicati sopra la soglia IoU. |

## Invarianti verificati

La suite protegge in particolare questi comportamenti:

- i dati restituiti ai moduli successivi sono immutabili o trattati come tali;
- configurazione, modello, immagine e output vengono validati ai rispettivi
  confini;
- la stessa source RGB alimenta le nove viste, fornisce le dimensioni globali e
  rimane lo sfondo del rendering;
- ogni immagine produce una vista completa e otto crop deterministici;
- le nove chiamate usano una sessione, avvengono in sequenza e devono riuscire
  tutte;
- le coordinate vengono ricostruite e limitate prima di accettare l'area della
  box;
- la NMS opera per classe con IoU 0,50 e criteri di pareggio stabili;
- il filtro dell'operatore non altera inferenza, fusione o NMS;
- eccezioni di terze parti non divulgano dettagli nei messaggi applicativi;
- un successo richiede un file di output regolare e non vuoto.

## Snapshot sulle immagini di riferimento

Quando sono disponibili, le immagini di riferimento vengono elaborate con
modello e dipendenze reali.
Per ciascuna, il test confronta i conteggi multi-vista con valori approvati e
verifica che gli output siano JPEG non vuoti con dimensioni attese.

I conteggi esistenti devono restare invariati con la source RGB passata
direttamente alle nove viste. Sono uno **snapshot di regressione**: segnalano
cambiamenti nel comportamento del modello, nella geometria o nel
post-processing, ma non sono ground truth e non misurano accuracy, precision,
recall o mAP. Un conteggio uguale non dimostra che posizione, classe e
confidenza di ogni box siano semanticamente corrette.

La configurazione locale di `phenocam/classes/configuration.py` puo essere
modificata dall'operatore.
Prima di interpretare un fallimento degli snapshot occorre verificare che la
selezione attesa dal test non sia stata alterata intenzionalmente.

## Prestazioni

Il valore `Execution time` e verificabile come somma dei nove intervalli
`session.run()`, ma non rappresenta il tempo completo percepito dall'operatore.
Per una misura end-to-end occorre cronometrare il processo esternamente,
includendo avvio, sessione, I/O e post-processing.

Le misure storiche a vista singola non descrivono la pipeline multi-vista
corrente. Il limite di 15 secondi sul comando completo e un criterio di
accettazione ancora da verificare sulla board Raspberry Pi 3 di riferimento.
Stato: `not verified — external verification: reference Raspberry Pi 3 is unavailable`.
Non deve essere dedotto da risultati macOS o da hardware piu recente.

Per una verifica attendibile sulla board si devono registrare almeno:

- versione di Raspberry Pi OS, architettura e versione Python;
- versioni esatte delle dipendenze;
- stato di alimentazione e throttling termico;
- valore ONNX stampato e wall time completo per ogni immagine;
- picco RSS della pipeline multi-vista;
- esito e dimensioni dei sei output di riferimento.

## Limiti della verifica

I test non trasformano in supportate funzioni che il programma non implementa:
batch ONNX, elaborazione concorrente, video, webcam, URL, input standard,
modelli con classi personalizzate o output YOLO grezzo restano fuori contratto.

La suite locale verifica correttezza strutturale e regressioni note. Compatibilita
con filesystem particolari, immagini eccezionalmente grandi, build diverse di
ONNX Runtime e comportamento termico richiedono prove nell'ambiente reale.

Per risultati e vincoli gia registrati consultare
[Modifiche per Raspberry Pi](modifiche-raspberry-pi.md) e
[Limitazioni Raspberry Pi](limitazioni-raspberry-pi.md).
