<!--
Scopo: documentare l'algoritmo multi-vista fino ai rendering annotato e privacy.
Responsabilita: rendere espliciti preprocessing, geometria, contratti e output finali.
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
partecipano alle sedici inferenze e alla soppressione globale; il filtro viene
applicato soltanto durante il rendering finale.

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

Lo stesso oggetto RGB alimenta direttamente tutte le sedici viste, fornisce le
dimensioni per la normalizzazione globale e resta la sorgente non mutata dei
rendering finali.

## 4. Geometria delle sedici viste

La prima vista contiene tutta l'immagine e ha priorita 0. Seguono quindici crop
con priorita da 1 a 15 in ordine per righe:

- immagine orizzontale o quadrata: 5 colonne per 3 righe (`5×3`);
- immagine verticale: 3 colonne per 5 righe (`3×5`).

Con dimensione sorgente `D`, numero di celle `n` e overlap nominale `o = 0,20`,
la dimensione del crop sull'asse e:

```text
crop = ceil(D / (n - (n - 1) * o))
```

La distanza disponibile `D - crop` viene divisa uniformemente tra i `n - 1`
intervalli. Ogni posizione e arrotondata, mentre prima e ultima sono ancorate
esplicitamente a `0` e `D - crop`. Questa scelta garantisce copertura completa,
coordinate nei limiti e comportamento deterministico anche con dimensioni non
divisibili.

Per una sorgente `4608×2592`, la griglia `5×3` produce crop `1098×997`,
origini X `0, 878, 1755, 2632, 3510` e origini Y `0, 798, 1595`.
Immagini minuscole possono generare origini coincidenti: tutti i quindici
tentativi restano intenzionali e mantengono le rispettive priorita.

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
alla volta, invece di conservare sedici input contemporaneamente.

## 6. Esecuzione e tempo misurato

Per ogni vista `run_tensor()` chiama:

```text
session.run((nome_output,), {nome_input: tensore})
```

Il cronometro racchiude soltanto questa chiamata. Il valore stampato al termine
e la somma dei sedici intervalli ONNX; non include avvio Python, sessione, lettura
immagine, preparazione, post-processing, rendering o scrittura.

Anche dopo il controllo statico del modello, il risultato dinamico deve essere
una lista o tupla con un solo `numpy.ndarray`, tipo `float32`, tre dimensioni e
forma `[1, N, 6]`. L'intera transazione fallisce se questo contratto non vale.

## 7. Normalizzazione delle detection

Ogni riga e interpretata come:

```text
[x1, y1, x2, y2, confidenza, id_classe]
```

La riga viene ignorata se non ha sei valori convertibili in numeri finiti, se
l'ID non e un intero esatto o se non compare nella mappa del modello. Ogni
classe richiede confidenza maggiore o uguale a 0,30; la soglia e inclusiva.

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

## 8. Soppressione globale delle detection

Le detection delle sedici viste vengono divise in domini di soppressione.
`car`, `bus` e `truck` condividono un unico dominio e possono quindi competere
anche con etichette diverse; ogni altra classe ha un dominio separato per ID.
In ogni dominio i candidati sono ordinati per:

1. confidenza decrescente;
2. priorita crescente della vista;
3. priorita crescente della riga.

L'IoU e l'area di intersezione divisa per l'area di unione. La copertura della
box minore e la stessa intersezione divisa per la minore delle due aree. Un
candidato viene scartato quando, rispetto a una box gia accettata nel dominio,
l'IoU e maggiore o uguale a 0,50 **oppure** la copertura della box minore e
maggiore o uguale a 0,50. Le soglie sono inclusive.

Il vincitore conserva classe, confidenza e coordinate originali: non avvengono
fusione, media o unione delle box. L'ordinamento finale resta confidenza
decrescente, priorita della vista crescente, priorita della riga crescente e ID
di classe crescente. La selezione configurata dall'operatore avviene soltanto
dopo questa soppressione.

Il post-processing geometrico e le soglie di confidenza riducono i falsi
positivi noti, ma non puo garantire accuratezza semantica in scene arbitrarie.

## 9. Selezione e output finali

Soltanto dopo la soppressione `write_outputs()` converte una volta gli ID
abilitati in un insieme e filtra le detection. La stessa selezione governa
entrambi i prodotti; non cambia inferenza, fusione o soppressione. La source RGB
normalizzata non viene mutata: ogni prodotto richiesto parte da una copia
indipendente.

### Output annotato

Le detection selezionate sono disegnate con rettangolo, nome della classe e
confidenza a due decimali. Font e linea scalano rispetto al lato minore. La
classe con ID 0 usa rosso-arancio, le altre azzurro. L'aspetto coincide con il
precedente output annotato.

### Output privacy

Il privacy parte dalla source senza annotazioni e non aggiunge box, nomi o
confidenze. Soltanto le detection conservate raggiungono questo output: la
soppressione non cambia geometria o intensita dello sfocamento. Per ogni
detection selezionata, nell'ordine finale deterministico:

1. calcola larghezza `x2 - x1` e altezza `y2 - y1`;
2. espande ciascun lato del 10%, usando `_PRIVACY_MARGIN_RATIO = 0.10`;
3. applica floor a sinistra/alto e ceil a destra/basso;
4. limita il rettangolo ai bordi, con destra/basso esclusivi secondo Pillow;
5. usa raggio `max(8 px, 0.10 * lato corto della regione finale)`, fissato da
   `_PRIVACY_MIN_BLUR_RADIUS = 8` e `_PRIVACY_BLUR_RADIUS_RATIO = 0.10`;
6. applica `GaussianBlur` al crop corrente e lo reinserisce nello stesso punto.

Le sovrapposizioni ricevono quindi piu blur in sequenza. Le detection di classi
disabilitate non modificano pixel, anche quando si sovrappongono a una regione
abilitata. Il contratto e rettangolare: non usa maschere o segmentazione.

### Persistenza

Con entrambi gli output, l'annotato viene renderizzato, salvato e verificato
prima di iniziare il privacy. Un errore successivo non rimuove l'annotato gia
completato. Zero detection selezionate non e un errore: ogni prodotto richiesto
viene comunque salvato ed e pixel-equivalente alla source normalizzata nei
formati lossless.

Pillow deduce il formato dalla destinazione e puo sovrascrivere file esistenti.
Dopo ogni `save()`, il percorso deve essere un file regolare non vuoto; qualunque
errore di rendering, filtro, I/O o verifica diventa `OutputWriteError` senza
dettagli privati.
