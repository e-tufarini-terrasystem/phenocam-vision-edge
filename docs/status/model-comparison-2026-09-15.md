# Nuova valutazione: YOLO26n base, precedente 0.1.5 e nuovo 0.1.6

> Historical artifact labels use normalized model versions, not filesystem paths.
> Exact commands, identifiers and paths remain in the original document:
> `git cat-file blob 2790fb729849ba3b59913c625b751145927d5bd0` from the
> [source snapshot](https://github.com/e-tufarini-terrasystem/phenocam-vision-edge/tree/cc63857c12c5553c2e3451854863edf7c0705e2a).

> Historical record for this iteration; use the [current guide](../development.md#current-workflow).
> Commands refer to the original workflow; ignored artifacts require local evidence.

Valutazione eseguita il 15 settembre 2026. Tre artefatti già fissati, stessa pipeline e nuova esecuzione completa: nessun riaddestramento o ricerca della soglia.

## Conclusione

Il nuovo artefatto migliora il rilevamento delle **auto su PKLot e Raspberry**, anche rispetto al precedente specializzato 0.1.5, alla stessa configurazione operativa. Non risulta migliore in generale.

- **PKLot:** TP 282 → 882, FP 50 → 91, FN 1.004 → 404; F1 0,349 → 0,781. Tutti i 6 gruppi migliorano; restano immagini dello stesso parcheggio.
- **Raspberry:** TP 1.730 → 2.042, FP 112 → 243; F1 0,519 → 0,574. Tutte le 11 giornate migliorano, ma la recall resta al 42,3%: 2.786 auto mancate su 4.828.
- **TS02:** TP 595 → 597, FP 18 → 29; F1 0,886 → 0,881. Variazione piccola e intervallo appaiato che include zero: nessun vantaggio chiaro.
- **Scene miste, auto:** TP 45 → 41, FP 40 → 81; F1 0,566 → 0,418. Regressione osservata anche nell’intervallo appaiato, interamente negativo.
- **Persone:** nessun progresso generale. Nelle scene miste TP 153 → 135 e FP 228 → 90: F1 sale, ma si trovano meno persone. Raspberry passa da 47 a 45 TP; TS02 resta a 31 TP. PKLot trova soltanto 1 delle 12 apparizioni di persone annotate.

Gli intervalli rafforzano la lettura del campione storico; non autorizzano una conclusione su camere nuove o una promozione automatica del modello.

## Protocollo fissato prima delle misure

- 652 immagini per modello: validation 206, PKLot holdout 6, TEST-ID 200, TEST-OOD 240 (Raspberry 120 e TS02 120).
- Soglia **0,47**, configurazione operativa corrente di `scripts/batch.sh`. Il confronto storico del 14 settembre usava 0,35: i numeri non vanno mescolati.
- Una vista intera e 15 ritagli, ingresso 640 × 640. Deduplicazione: IoU 0,50 oppure copertura del box più piccolo 0,50 solo fra viste diverse.
- Matching a IoU ≥ 0,50 e classe corretta. Cinque classi operative: persone, auto, moto, autobus e camion; biciclette escluse dal punteggio.
- CPU ONNX Runtime, Apple M4, 4 thread, ambiente `.venv-export`. Esecuzioni sequenziali: questa misura riguarda la qualità, non la velocità.
- Nessuna sovrapposizione esatta di hash tra le immagini valutate e il training 0.1.6. Questo controllo non dimostra indipendenza di camere, giorni o fotogrammi simili.
- I test sono storici e già consultati; la validation era stata usata per selezionare il modello. È una nuova misura, non un nuovo test cieco.

TP = oggetti trovati correttamente; FP = falsi allarmi; FN = oggetti mancati. Precision = TP/(TP+FP); recall = TP/(TP+FN); F1 = 2TP/(2TP+FP+FN). I conteggi misurano apparizioni nelle fotografie, non individui o veicoli unici.

## Risultati per classe e contesto

### Auto

| Contesto | Modello | TP | FP | FN | Precision | Recall | F1 |
|---|---|---:|---:|---:|---:|---:|---:|
| PKLot holdout | base | 230 | 46 | 1056 | 0.833 | 0.179 | 0.294 |
| PKLot holdout | 0.1.5 | 282 | 50 | 1004 | 0.849 | 0.219 | 0.349 |
| PKLot holdout | 0.1.6 | 882 | 91 | 404 | 0.906 | 0.686 | 0.781 |
| Raspberry | base | 1644 | 93 | 3184 | 0.946 | 0.341 | 0.501 |
| Raspberry | 0.1.5 | 1730 | 112 | 3098 | 0.939 | 0.358 | 0.519 |
| Raspberry | 0.1.6 | 2042 | 243 | 2786 | 0.894 | 0.423 | 0.574 |
| TS02 | base | 576 | 16 | 154 | 0.973 | 0.789 | 0.871 |
| TS02 | 0.1.5 | 595 | 18 | 135 | 0.971 | 0.815 | 0.886 |
| TS02 | 0.1.6 | 597 | 29 | 133 | 0.954 | 0.818 | 0.881 |
| Scene miste TEST-ID | base | 44 | 37 | 30 | 0.543 | 0.595 | 0.568 |
| Scene miste TEST-ID | 0.1.5 | 45 | 40 | 29 | 0.529 | 0.608 | 0.566 |
| Scene miste TEST-ID | 0.1.6 | 41 | 81 | 33 | 0.336 | 0.554 | 0.418 |

### Persone

| Contesto | Modello | TP | FP | FN | Precision | Recall | F1 |
|---|---|---:|---:|---:|---:|---:|---:|
| PKLot holdout | base | 0 | 0 | 12 | 0.000 | 0.000 | 0.000 |
| PKLot holdout | 0.1.5 | 0 | 0 | 12 | 0.000 | 0.000 | 0.000 |
| PKLot holdout | 0.1.6 | 1 | 2 | 11 | 0.333 | 0.083 | 0.133 |
| Raspberry | base | 45 | 1 | 96 | 0.978 | 0.319 | 0.481 |
| Raspberry | 0.1.5 | 47 | 1 | 94 | 0.979 | 0.333 | 0.497 |
| Raspberry | 0.1.6 | 45 | 0 | 96 | 1.000 | 0.319 | 0.484 |
| TS02 | base | 30 | 5 | 9 | 0.857 | 0.769 | 0.811 |
| TS02 | 0.1.5 | 31 | 3 | 8 | 0.912 | 0.795 | 0.849 |
| TS02 | 0.1.6 | 31 | 3 | 8 | 0.912 | 0.795 | 0.849 |
| Scene miste TEST-ID | base | 158 | 230 | 145 | 0.407 | 0.521 | 0.457 |
| Scene miste TEST-ID | 0.1.5 | 153 | 228 | 150 | 0.402 | 0.505 | 0.447 |
| Scene miste TEST-ID | 0.1.6 | 135 | 90 | 168 | 0.600 | 0.446 | 0.511 |

### Totale delle cinque classi

| Contesto | Modello | TP | FP | FN | Precision | Recall | F1 |
|---|---|---:|---:|---:|---:|---:|---:|
| PKLot holdout | base | 230 | 58 | 1083 | 0.799 | 0.175 | 0.287 |
| PKLot holdout | 0.1.5 | 282 | 62 | 1031 | 0.820 | 0.215 | 0.340 |
| PKLot holdout | 0.1.6 | 883 | 98 | 430 | 0.900 | 0.673 | 0.770 |
| Raspberry | base | 1693 | 97 | 3321 | 0.946 | 0.338 | 0.498 |
| Raspberry | 0.1.5 | 1781 | 116 | 3233 | 0.939 | 0.355 | 0.515 |
| Raspberry | 0.1.6 | 2088 | 244 | 2926 | 0.895 | 0.416 | 0.568 |
| TS02 | base | 638 | 23 | 233 | 0.965 | 0.732 | 0.833 |
| TS02 | 0.1.5 | 658 | 22 | 213 | 0.968 | 0.755 | 0.848 |
| TS02 | 0.1.6 | 650 | 32 | 221 | 0.953 | 0.746 | 0.837 |
| Scene miste TEST-ID | base | 244 | 304 | 199 | 0.445 | 0.551 | 0.492 |
| Scene miste TEST-ID | 0.1.5 | 240 | 309 | 203 | 0.437 | 0.542 | 0.484 |
| Scene miste TEST-ID | 0.1.6 | 212 | 198 | 231 | 0.517 | 0.479 | 0.497 |

## Differenze appaiate e incertezza

Bootstrap percentile appaiato: 10.000 ricampionamenti, seed 20260915. Si ricampionano gli stessi gruppi per i due modelli, mantenendo insieme le immagini della stessa camera e giornata. Per ogni campione si sommano TP/FP/FN e poi si calcola F1; non si fa la media delle F1 dei giorni.

Gli intervalli al 95% sono esplorativi e condizionati ai siti e giorni osservati. Non eliminano l’esposizione precedente ai test, la dipendenza fra giorni o la selezione già avvenuta; non costituiscono una prova confermativa su nuove camere. Non vengono riportati p-value o una probabilità che il modello sia migliore. Il ricampionamento non misura la variabilità fra riaddestramenti.

| Contesto, auto | Riferimento | Δ F1 nuovo − riferimento | Intervallo 95% | Gruppi migliori / uguali / peggiori |
|---|---|---:|---|---|
| PKLot holdout | base | +0.486 | [+0.380; +0.574] | 6 / 0 / 0 |
| PKLot holdout | 0.1.5 | +0.432 | [+0.317; +0.525] | 6 / 0 / 0 |
| Raspberry | base | +0.073 | [+0.060; +0.088] | 11 / 0 / 0 |
| Raspberry | 0.1.5 | +0.055 | [+0.041; +0.070] | 11 / 0 / 0 |
| TS02 | base | +0.009 | [-0.008; +0.023] | 12 / 1 / 6 |
| TS02 | 0.1.5 | -0.006 | [-0.019; +0.008] | 7 / 4 / 8 |
| Scene miste TEST-ID | base | -0.149 | [-0.265; -0.033] | 5 / 182 / 6 |
| Scene miste TEST-ID | 0.1.5 | -0.148 | [-0.265; -0.035] | 3 / 184 / 6 |

Un intervallo che attraversa zero non distingue chiaramente il segno del cambiamento nel campione. Un intervallo interamente positivo descrive un vantaggio nel campione storico, senza dimostrare superiorità generale.

## Validation e decisione operativa

Sulla validation la F1 globale sale da 0,455 (0.1.5) a 0,468 (0.1.6), ma i TP scendono da 232 a 216 e i FN salgono da 257 a 273; i FP scendono da 298 a 218. Migliorare F1 può significare meno falsi allarmi anche trovando meno oggetti.

Il nuovo modello rimane sperimentale. Questa misura del singolo artefatto non modifica l’esito storico dei criteri di accettazione della ricetta o delle repliche di training. Per adottarlo occorre fissare il contesto, un limite accettabile di falsi allarmi e oggetti mancati, scegliere eventuali soglie sulla validation e verificare una volta su nuove giornate annotate senza riadattare il modello al test.

Il file locale workspace del training 0.1.6 (`reconstructed-sealed-days.json`) riferisce 241 immagini tenute separate (94 Raspberry, 147 TS02). Non è stato trovato localmente un export umano per valutarle; CVAT su localhost:8080 non era raggiungibile e Docker non era attivo. Le annotazioni sul server non sono state verificate: non si afferma che non esistano. Non sono stati usati riquadri prodotti dal modello come verità di riferimento.

## Riproducibilità e controlli

Artefatti in `output/model-comparison-2026-09-15/` (directory ignorata da Git):

- `run.py`, `commands.json`, `state.json`: orchestrazione, 12 comandi eseguiti e completamento.
- `protocol.json`, `environment.txt`, `receipts/`: hash dei modelli, del codice, dei manifest e di tutte le immagini e annotazioni usate; ambiente e condizioni congelate.
- `<split>/<modello>/runtime-metrics.json`, `errors.csv`, `worst-examples/`: metriche, errori per immagine e bounding box reali.
- `analysis.py`, `analysis.json`, `summary.csv`, `per-image.csv`: ricostruzione dei conteggi e intervalli appaiati.
- `data-audit.json`: controllo di sovrapposizione esatta e disponibilità delle nuove annotazioni.

Tutti gli hash sono stati verificati dopo le esecuzioni. Ogni TP/FP/FN ricostruito dagli errori coincide con le metriche originali per classe e per insieme. Controllati anche identità del confronto appaiato, simmetria invertendo i modelli e un caso F1 noto; gli intervalli coincidono con `scipy.stats.bootstrap(method="percentile")` entro 1e-12 su un campione di controllo (`math-checks.json`).

| Modello | SHA-256 ONNX |
|---|---|
| base | `09fa4b119751eafe13962607f5a9742aff9f8e3e3e3a950c057ecb8d40cd82ec` |
| 0.1.5 | `252f302257759cae6d40579fb76b74d66f87bdce2997d44f89dba85d03420379` |
| 0.1.6 | `72521182fa0c90fec60fb1cd9f3b2ac113d16de476aaeea3a328508b7b2d0b30` |

## Fonti del metodo

- [SciPy: bootstrap appaiato e intervalli percentile](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.bootstrap.html).
- [scikit-learn: dipendenza e gruppi nella valutazione](https://scikit-learn.org/stable/modules/cross_validation.html#cross-validation-iterators-for-grouped-data).

La presentazione di riferimento è `presentazioni/PhenoCam-Vision-Edge.pptx` e `.pdf`, aggiornata anche con i risultati del confronto. La cartella `presentazioni/`, inclusi i materiali di lavorazione in `.build/`, è locale ed esclusa da Git. I file `PhenoCam-Vision-Edge-Detection-Conclusioni` restano copie compatibili dello stesso contenuto. La pagina 14 usa la scena TS02 del 20 agosto 2026 (base: TP 4, FP 0, FN 2; nuovo: TP 6, FP 0, FN 0), come esempio illustrativo selezionato, con fotografie intere e senza ingrandimenti aggiuntivi. Include una nota finale sulle condizioni commerciali AGPL/Enterprise Ultralytics; non modifica la licenza Apache 2.0 del repository né certifica la conformità del progetto.

Il ricontrollo dei documenti ha confermato i conteggi delle 12 valutazioni e gli
intervalli appaiati. Corrette due descrizioni a pagina 17: il batch contiene
16 immagini, con accumulo di 4 batch dopo il warmup (64 immagini nominali per
aggiornamento); la chiusura di Mosaic era prevista all'epoca 26 delle 30
programmate e non viene raggiunta dal run scelto, terminato all'epoca 13.
La stessa pagina rende esplicita la copertura tra viste: 80% nella selezione
storica, 50% nel nuovo confronto. La pagina 16 distingue ora anche nel testo
visibile Apache 2.0 del repository e AGPL/Enterprise dei pesi YOLO.
Note, PDF e presentazione sono allineati; metriche e conclusioni restano quelle
del confronto verificato.

La revisione linguistica successiva uniforma addestramento, validazione e
riquadri, sostituisce «giri sugli esempi» con «epoche di addestramento» e chiarisce
le descrizioni dei ritagli, dei pesi preaddestrati e dei seed. Le didascalie ora
parlano di auto «rilevate correttamente» e di immagini annotate di nuove giornate
e camere. Aggiornate anche le note e le copie `Detection-Conclusioni`; numeri,
immagini e fonti sono invariati. Il comando locale
`node presentazioni/.build/export.mjs` rigenera entrambi i formati dalla
revisione corrente. Verificati testo del PDF, impaginazione delle 19 pagine e
allineamento delle copie.
