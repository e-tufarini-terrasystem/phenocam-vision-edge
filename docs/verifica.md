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

La maggior parte della suite usa file temporanei, immagini sintetiche e mock.
In questo modo verifica i contratti senza modificare `classes.py`, senza scrivere
nelle directory operative e, dove non serve, senza caricare ONNX Runtime.

## Copertura per confine

| Test | Contratto principale |
|---|---|
| `tests/test_arguments.py` | Opzioni richieste, tipi di percorso, directory padre e identita distinta tra input e output. |
| `tests/test_selection.py` | Inventario COCO esatto, soli booleani modificabili, almeno una classe e metadata del modello completi. |
| `tests/test_runtime.py` | Thread, opzioni della sessione, contratto tensoriale e validazione dell'output dinamico. |
| `tests/test_views.py` | EXIF, RGB, letterbox, forma del tensore, geometria adattiva, copertura e ordine delle viste. |
| `tests/test_gamma.py` | Configurazione, mediana su istogramma, soglie, percorsi identita e LUT gamma RGB. |
| `tests/test_detections.py` | Soglia, valori non validi, conversione globale, clipping, IoU, NMS per classe e pareggi deterministici. |
| `tests/test_inference.py` | Unica model image, proprieta della source, nove viste, tempi e assenza di output parziale. |
| `tests/test_run.py` | Messaggi pubblici, separazione stdout/stderr e stati del processo. |
| `tests/test_reference_images.py` | Output JPEG reali, conteggi snapshot e assenza di duplicati sopra la soglia IoU. |

## Invarianti verificati

La suite protegge in particolare questi comportamenti:

- i dati restituiti ai moduli successivi sono immutabili o trattati come tali;
- configurazione, modello, immagine e output vengono validati ai rispettivi
  confini;
- la configurazione gamma viene validata anche se disabilitata e la source non
  viene trasformata sul percorso di default;
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

Le immagini di riferimento vengono elaborate con modello e dipendenze reali.
Per ciascuna, il test confronta i conteggi multi-vista con valori approvati e
verifica che gli output siano JPEG non vuoti con dimensioni attese.

Con `ADAPTIVE_GAMMA_ENABLED = False`, i conteggi esistenti devono restare
invariati. Sono uno **snapshot di regressione**: segnalano cambiamenti nel
comportamento del modello, nella geometria o nel post-processing, ma non sono
ground truth e non misurano accuracy, precision, recall o mAP. Un conteggio
uguale non dimostra che posizione, classe e confidenza di ogni box siano
semanticamente corrette e non autorizza ad abilitare la gamma.

La configurazione locale di `classes.py` puo essere modificata dall'operatore.
Prima di interpretare un fallimento degli snapshot occorre verificare che la
selezione attesa dal test non sia stata alterata intenzionalmente.

## Gate di accuratezza esterno

Il default gamma puo diventare attivo soltanto se, sullo stesso dataset
annotato e con identici modello, nove viste e procedura di valutazione, la
variante adattiva ottiene mAP50 complessiva superiore alla baseline e recall
complessiva non inferiore. Il controllo visivo dell'utente e aggiuntivo: puo
rifiutare un risultato, ma non sostituisce le due metriche.

Dataset annotato ed evaluator non fanno parte del repository. Stato corrente:
`not verified — external verification: annotated dataset and evaluator are not part of the repository`.

## Prestazioni

Il valore `Execution time` e verificabile come somma dei nove intervalli
`session.run()`, ma non rappresenta il tempo completo percepito dall'operatore.
Per una misura end-to-end occorre cronometrare il processo esternamente,
includendo avvio, sessione, I/O e post-processing.

Le misure storiche a vista singola non descrivono la pipeline multi-vista
corrente. Il limite di 15 secondi sul comando completo e un criterio di
accettazione ancora da verificare sulla board Raspberry Pi 3 di riferimento.
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
[Modifiche per Raspberry Pi](../MODIFICHE_RASPBERRY_PI.md) e
[Limitazioni Raspberry Pi](../LIMITAZIONI_RASPBERRY_PI.md).
