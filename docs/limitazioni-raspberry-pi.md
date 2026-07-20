<!--
Scopo: preservare i vincoli Raspberry Pi e i confini delle funzioni non supportate.
Responsabilita: distinguere compatibilita, prestazioni e verifiche esterne.
Contesto: integra i documenti tecnici con i limiti specifici del deploy Raspberry Pi.
-->

# Limitazioni riscontrate su Raspberry Pi 3

## Piattaforma supportata

Il test copre Raspberry Pi OS **64 bit** (`aarch64`) con Python 3.13.5. È
richiesto Python 3.11 o successivo; il requirements include un pin NumPy
separato per Python 3.11, ma il benchmark è stato eseguito con il pin Python
3.13. PyPI pubblica wheel CPython 3.11 ARM64 per
[NumPy 2.3.5](https://pypi.org/project/numpy/2.3.5/),
[ONNX Runtime 1.27.0](https://pypi.org/project/onnxruntime/1.27.0/) e
[Pillow 12.3.0](https://pypi.org/project/pillow/12.3.0/). Raspberry Pi OS a 32
bit (`armv7l`) non è stato verificato e non va considerato compatibile senza
una prova specifica o una build dedicata di ONNX Runtime.

## Prestazioni

- Il comando multi-vista esegue sedici chiamate ONNX sequenziali nella stessa
  sessione: una sull'immagine completa e quindici su ritagli adattivi sovrapposti.
- `Execution time` somma i sedici intervalli `session.run()` e non comprende
  preparazione delle viste, soppressione, rendering, scrittura, import o
  caricamento della sessione. Il wall time completo comprende invece l'intero
  processo CLI.
- Il rapporto ONNX osservato `1,67×` rispetto alla precedente griglia `4×2` e
  informativo: non e una soglia di accettazione e non predice la latenza
  end-to-end sul Raspberry Pi.
- not verified — external verification: reference Raspberry Pi 3 is unavailable.
  Il limite di 15 secondi per comando multi-vista resta un obiettivo di
  accettazione esterno non verificato, non un risultato misurato su workstation.
- Il precedente valore medio di 0,848 s e il confronto da 0,874 s con quattro
  thread a 1,209 s con due e 1,931 s con uno descrivono soltanto il benchmark
  storico a vista singola; non rappresentano il comando multi-vista corrente.
- Il tempo può aumentare con throttling termico, alimentazione insufficiente,
  servizi concorrenti o microSD più lenta. Il dispositivo di controllo
  `vcgencmd` non era accessibile nel sandbox; la temperatura letta da sysfs dopo
  i test era circa 61,8 °C, ma non è stato possibile certificare lo stato di
  throttling.

## Memoria

- Il picco storico a vista singola è circa 209 MiB RSS. Il consumo multi-vista
  non è stato rimisurato sulla board; le viste sono comunque eseguite in
  sequenza e viene preparato un solo tensore alla volta. Questo limita la memoria
  simultanea delle viste, ma non elimina il costo delle sedici esecuzioni. Le
  esecuzioni parallele restano escluse perché aumenterebbero il consumo.
- Le immagini sorgente sono 4608x2592: la decodifica RGB occupa molta più RAM
  del JPEG compresso. File con risoluzioni molto superiori possono aumentare il
  picco prima del resize a 640x640.
- ONNX Runtime usa solo `CPUExecutionProvider`; non viene sfruttata una GPU o
  un acceleratore dedicato.

## Spazio microSD

Una microSD nominale da 8 GB espone qui un filesystem da circa 6,9 GiB. Il
progetto completo con virtualenv, entrambi i modelli, immagini e output di test
occupa circa 187 MiB ed è compatibile con i 3,04 GiB rimasti sul sistema di
prova.

Il margine dipende però dallo spazio già occupato dal sistema operativo. Le
installazioni con meno di circa 250 MiB liberi non hanno un margine prudente per
virtualenv, download temporanei e output. `--no-cache-dir` evita di conservare
le wheel scaricate. Per un deploy minimo si possono omettere `models/yolo26n.pt` e i
file di export, risparmiando almeno 5,3 MiB oltre alle dipendenze di export.

## Vincoli del modello e della pipeline

- Il runtime accetta il contratto del modello incluso: YOLO detection
  end-to-end, input float statico `[1,3,640,640]`, output `[1,N,6]` e 80 classi
  COCO. Un ONNX con output YOLO grezzo o classi personalizzate viene rifiutato.
- La soglia di confidenza inclusiva e fissata a 0,30 per tutte le classi; non e
  esposta nella CLI.
- Una vista completa e quindici ritagli adattivi usano griglie `5×3` per
  immagini orizzontali o quadrate e `3×5` per immagini verticali, con 20% di
  overlap nominale e coordinate globali. La soppressione usa IoU o copertura
  della box minore a 0,50; `car`, `bus` e `truck` condividono un dominio, mentre
  le altre classi competono soltanto con se stesse.
- Il filtro `phenocam/classes/configuration.py` agisce soltanto dopo le sedici
  inferenze, la fusione e la
  soppressione: disabilitare classi non riduce tempo CPU o RAM del modello.
- I conteggi di riferimento non sono asseriti: non sono ground truth ne una
  misura di accuracy, precision, recall o mAP.
- Senza riaddestramento o un secondo classificatore, il post-processing riduce
  i falsi positivi noti a bassa confidenza ma non puo garantire che ogni oggetto
  simile a un cartello venga rifiutato in scene eterogenee.
- È supportata una sola immagine locale per processo. Non sono implementati
  batch, directory, video, webcam, URL o standard input.
- La directory di output deve esistere e un file esistente viene sovrascritto.
- L'export ONNX non è una funzione di deploy: richiede Ultralytics e una stack
  molto più pesante. Va eseguito su workstation e non sulla microSD da 8 GB.

## Aspetti non modificati

Non sono state applicate quantizzazione INT8, riduzione della risoluzione del
modello, conversione a TFLite o sostituzione con un'altra architettura. Queste
tecniche potrebbero ridurre latenza o spazio, ma avrebbero modificato il modello
o il sistema di inferenza oltre il vincolo richiesto.
