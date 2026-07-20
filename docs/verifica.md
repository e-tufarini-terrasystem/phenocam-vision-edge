<!--
Scopo: descrivere come il comportamento del software viene verificato.
Responsabilita: collegare invarianti, rendering privacy, test e limiti delle prove.
Contesto: separa le garanzie locali sui due output dalle verifiche esterne.
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
| `tests/test_arguments.py` | Due output opzionali, presenza minima, ordine e identita distinte tra input/output/output. |
| `tests/test_selection.py` | Inventario COCO esatto, soli booleani modificabili, almeno una classe e metadata del modello completi. |
| `tests/test_runtime.py` | Thread, opzioni della sessione, contratto tensoriale e validazione dell'output dinamico. |
| `tests/test_views.py` | EXIF, RGB, letterbox, tensore, quindici crop, copertura e ordine delle sedici viste. |
| `tests/test_detections.py` | Soglia di confidenza uniforme e inclusiva, valori non validi, conversione globale, clipping, IoU, copertura della box minore e dominio `car`, `bus`, `truck`. |
| `tests/test_output.py` | Selezione, copie indipendenti, geometria/raggio privacy, ordine, persistenza e fallimento parziale. |
| `tests/test_pipeline.py` | Una sessione, sedici run e una soppressione globale per ogni combinazione di output; timing e ordine dei confini. |
| `tests/test_command.py` | Delega delle due destinazioni, messaggi pubblici, stream e stati del processo. |
| `tests/test_reference_images.py` | Inventario, output JPEG, vincitori noti e invarianti finali per IoU e copertura nei domini di soppressione. |

## Invarianti verificati

La suite protegge in particolare questi comportamenti:

- i dati restituiti ai moduli successivi sono immutabili o trattati come tali;
- configurazione, modello, immagine e output vengono validati ai rispettivi
  confini;
- la stessa source RGB alimenta le sedici viste e rimane non mutata, mentre ogni
  prodotto nasce da una copia indipendente;
- ogni immagine produce una vista completa e quindici crop deterministici;
- le sedici chiamate usano una sessione, avvengono in sequenza e devono riuscire
  tutte;
- le coordinate vengono ricostruite e limitate prima di accettare l'area della
  box;
- ogni classe usa la stessa soglia di confidenza inclusiva pari a 0,30 prima
  della conversione delle coordinate;
- la soppressione usa IoU o copertura della box minore a 0,50, con un dominio
  condiviso per `car`, `bus`, `truck` e domini separati per le altre classi;
- il filtro dell'operatore non altera inferenza, fusione o soppressione;
- output annotato, privacy o entrambi eseguono sempre una sessione, sedici run e
  una soppressione prima di una sola delega finale;
- il privacy usa esattamente margine 10%, floor/ceil, clipping, coordinate
  destre/inferiori esclusive e raggio `max(8 px, 10% del lato corto)`;
- le sovrapposizioni vengono sfocate in ordine e le classi disabilitate ignorate;
- l'annotato viene scritto prima del privacy e resta presente se il secondo
  salvataggio fallisce;
- eccezioni di terze parti non divulgano dettagli nei messaggi applicativi;
- un successo richiede ogni file richiesto regolare e non vuoto, anche con zero
  detection selezionate.

## Integrazione sulle immagini di riferimento

Quando sono disponibili, le immagini di riferimento vengono elaborate con
modello e dipendenze reali. Il test garantisce l'inventario esatto delle sei
JPEG, dimensioni sorgente `4608×2592`, durata restituita di tipo `float`, output
JPEG regolare non vuoto con le stesse dimensioni e IoU strettamente inferiore a
`0,50` per ogni coppia finale nello stesso dominio. Ogni detection finale deve
inoltre avere confidenza maggiore o uguale a 0,30. Anche la copertura della box
minore deve essere strettamente inferiore a `0,50`. Appartengono allo stesso
dominio le detection della stessa classe e tutte le coppie i cui nomi sono tra
`car`, `bus` e `truck`; classi diverse fuori da questo gruppo non vengono
confrontate.

Le regressioni richiedono i tre vincitori `car` 0,70, 0,75 e 0,77 e il vincitore
`truck` 0,91 nelle rispettive coordinate con tolleranza di due pixel. Non
asseriscono conteggi totali: le immagini non sono ground truth. Questi controlli
non dimostrano rifiuto universale dei cartelli, precision, recall, accuracy o
mAP. La selezione in `phenocam/classes/configuration.py` modifica soltanto quali
box compaiono nell'output; il privacy resta coperto con immagini sintetiche in
`test_output.py`. Se le immagini ignorate non sono disponibili, lo stato e:
`not verified — external verification: reference images are unavailable`.

## Confronto aggregato osservato

| Geometria | Inferenze totali | Persone | Auto | Tempo ONNX relativo |
|---|---:|---:|---:|---:|
| `4×2` corrente | 9 | 18 | 183 | `1,00×` |
| `5×3` nuova | 16 | 22 | 210 | `1,67×` |

I conteggi piu alti e la confidenza media sostanzialmente invariata non
distinguono oggetti recuperati da falsi positivi. Queste osservazioni non
dimostrano maggiore accuracy, precision, recall o mAP e non sono criteri di
accettazione automatici.

## Prestazioni

Il valore `Execution time` e verificabile come somma dei sedici intervalli
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
