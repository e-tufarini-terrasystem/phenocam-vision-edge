<!--
Scopo: documentare l'algoritmo di inferenza multi-vista dall'immagine alle annotazioni.
Responsabilita: rendere espliciti preprocessing, geometria, contratti e filtri.
Contesto: dettaglia la transazione coordinata da phenocam/inference/pipeline.py.
-->

# Pipeline di inferenza

## 1. Configurazione delle classi

Prima di caricare il modello, `enabled_class_names()` importa
`phenocam/classes/configuration.py` e
verifica l'intera costante `COCO_CLASSES`. Sono invarianti:

- contenitore esterno, categorie e voci sono tuple;
- esistono esattamente le 12 categorie e le 80 classi COCO previste;
- nomi, appartenenza alle categorie e ordine non cambiano;
- ogni stato e esattamente un booleano Python;
- almeno una classe e abilitata.

L'operatore puo modificare soltanto i valori `True` e `False`. La configurazione
viene riletta a ogni chiamata, senza cache, e restituisce una tupla immutabile di
nomi abilitati nell'ordine canonico.

Dopo la validazione del modello, `model_class_ids()` verifica che i metadata
contengano gli stessi 80 nomi associati a chiavi intere contigue da 0 a 79. La
selezione avviene per nome, percio il modello puo associare ID diversi ai nomi
purche l'inventario sia completo e senza duplicati. Gli ID finali sono ordinati.

**Invariante:** la selezione non riduce il lavoro della rete. Tutte le classi
partecipano alle nove inferenze e alla NMS; il filtro viene applicato soltanto
durante il disegno.

## 2. Sessione e contratto ONNX

`create_session()` configura una sessione con:

- solo `CPUExecutionProvider`;
- esecuzione sequenziale;
- ottimizzazioni del grafo abilitate;
- un thread tra operazioni;
- memory pattern e arena CPU disabilitati per contenere la memoria sul Pi;
- log di ONNX Runtime limitati agli errori.

Il numero di thread interni e `min(4, numero CPU)` per impostazione predefinita.
`YOLO_NUM_THREADS` lo sostituisce soltanto se e un intero tra 1 e 4; valori
mancanti, non numerici o fuori intervallo riportano silenziosamente al default.

Prima dell'inferenza `model_contract()` richiede:

| Proprieta | Contratto |
|---|---|
| Input | Un solo tensore `float` con forma statica `[1, 3, H, W]` e dimensioni positive. |
| Output | Un solo tensore `float` tridimensionale con forma `[1, N, 6]`. |
| Metadata `task` | Valore `detect`. |
| Metadata `end2end` | Stringa `True`. |
| Metadata `names` | Rappresentazione letterale di una mappa di classi valida. |

La dimensione del modello non e scritta nel codice: `W` e `H` vengono lette
dall'input ONNX. Il modello incluso usa 640x640. Un modello con output YOLO
grezzo, input dinamico, classi personalizzate o piu tensori viene rifiutato.

## 3. Caricamento dell'immagine

Pillow apre il file in un context manager. `ImageOps.exif_transpose()` applica
l'orientamento dichiarato nei metadata EXIF, poi l'immagine viene convertita in
RGB e caricata interamente prima di chiudere il file sorgente.

Un file non decodificabile o un'immagine senza dimensioni positive produce un
errore di inferenza generico. Il dettaglio dell'eccezione e il percorso non
vengono inseriti nel messaggio pubblico.

Lo stesso oggetto RGB alimenta direttamente tutte le nove viste, fornisce le
dimensioni per la normalizzazione globale e rimane lo sfondo del rendering.

## 4. Geometria delle nove viste

La prima vista contiene tutta l'immagine e ha priorita 0. Seguono otto crop in
ordine per righe:

- immagine orizzontale o quadrata: 4 colonne per 2 righe;
- immagine verticale: 2 colonne per 4 righe.

Con dimensione sorgente `D`, numero di celle `n` e overlap nominale `o = 0,20`,
la dimensione del crop sull'asse e:

```text
crop = ceil(D / (n - (n - 1) * o))
```

La distanza disponibile `D - crop` viene divisa uniformemente tra i `n - 1`
intervalli. Ogni posizione e arrotondata, mentre prima e ultima sono ancorate
esplicitamente a `0` e `D - crop`. Questa scelta garantisce copertura completa,
coordinate nei limiti e comportamento deterministico anche con dimensioni non
divisibili. Immagini minuscole possono generare crop ripetuti: restano comunque
nove tentativi intenzionali.

Ogni `View` conserva origine e dimensione del crop, fattore di scala, padding e
priorita. Questi valori costituiscono la trasformazione inversa necessaria per
riportare le box nello spazio dell'immagine completa.

## 5. Letterbox e tensore

Ogni vista mantiene le proporzioni. Date le dimensioni del modello `(Wm, Hm)` e
della vista `(Wv, Hv)`:

```text
scala = min(Wm / Wv, Hm / Hv)
larghezza_ridimensionata = max(1, round(Wv * scala))
altezza_ridimensionata = max(1, round(Hv * scala))
offset_x = (Wm - larghezza_ridimensionata) // 2
offset_y = (Hm - altezza_ridimensionata) // 2
```

Il resize usa interpolazione bilineare. L'immagine ridimensionata viene centrata
su uno sfondo RGB `(114, 114, 114)`. NumPy converte poi i pixel in `float32`,
traspone da HWC a CHW, aggiunge la dimensione batch e divide per 255. Il risultato
e un array contiguo con forma `[1, 3, Hm, Wm]` e valori tra 0 e 1.

Le viste sono prodotte da un generatore: crop e tensori vengono preparati uno
alla volta, invece di conservare nove input contemporaneamente.

## 6. Esecuzione e tempo misurato

Per ogni vista `run_tensor()` chiama:

```text
session.run((nome_output,), {nome_input: tensore})
```

Il cronometro racchiude soltanto questa chiamata. Il valore stampato al termine
e la somma dei nove intervalli ONNX; non include avvio Python, sessione, lettura
immagine, preparazione, post-processing, disegno o scrittura.

Anche dopo il controllo statico del modello, il risultato dinamico deve essere
una lista o tupla con un solo `numpy.ndarray`, tipo `float32`, tre dimensioni e
forma `[1, N, 6]`. L'intera transazione fallisce se questo contratto non vale.

## 7. Normalizzazione delle detection

Ogni riga e interpretata come:

```text
[x1, y1, x2, y2, confidenza, id_classe]
```

La riga viene ignorata se non ha sei valori convertibili in numeri finiti, se
la confidenza e minore di 0,25, se l'ID non e un intero esatto o se non compare
nella mappa del modello.

Le coordinate del modello vengono prima liberate dal padding e dalla scala,
poi traslate con l'origine del crop:

```text
x_globale = (x_modello - offset_x) / scala + crop_x
y_globale = (y_modello - offset_y) / scala + crop_y
```

Ogni coordinata viene limitata a `[0, larghezza - 1]` o
`[0, altezza - 1]`. Dopo il clipping sono accettati soltanto rettangoli con
`x2 > x1` e `y2 > y1`. La detection immutabile conserva inoltre priorita della
vista e posizione originale della riga per risolvere i pareggi.

## 8. NMS globale per classe

Le detection delle nove viste vengono raggruppate per ID di classe. In ogni
gruppo i candidati sono ordinati per:

1. confidenza decrescente;
2. priorita crescente della vista;
3. priorita crescente della riga.

Un candidato viene scartato quando la sua Intersection over Union con una box
gia accettata e maggiore o uguale a 0,50. Classi diverse non si sopprimono tra
loro. L'ordinamento finale aggiunge l'ID classe come ultimo criterio, rendendo
il risultato riproducibile anche in presenza di pareggi.

## 9. Filtro e output

Soltanto ora gli ID non abilitati in `phenocam/classes/configuration.py` vengono
esclusi. Le detection
selezionate sono disegnate sull'immagine RGB completa con rettangolo, nome della
classe e confidenza a due decimali. Dimensione del font e spessore della linea
scalano rispetto al lato minore dell'immagine. La classe con ID 0 usa un colore
rosso-arancio; le altre un azzurro.

Pillow deduce il formato dal percorso di output. Un file esistente puo essere
sovrascritto. Dopo `save()`, il codice verifica che il percorso identifichi un
file regolare e che la sua dimensione sia maggiore di zero.
