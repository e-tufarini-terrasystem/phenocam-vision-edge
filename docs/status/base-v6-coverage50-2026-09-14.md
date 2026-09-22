# Base contro v6: auto e parcheggi con deduplicazione al 50%

> Historical record for this iteration; use the [current guide](../development.md#current-workflow).
> Commands refer to the original workflow; ignored artifacts require local evidence.

Confronto del 14 settembre 2026.

V6 migliora nettamente su PKLot e moderatamente sul parcheggio difficile di `raspberrypi2.local`. Su TS02 il bilancio è quasi invariato; nelle scene generiche del TEST-ID le auto peggiorano. Il vantaggio è quindi concreto ma dipende dalla scena. Non basta, da solo, a dichiarare raggiunta una soglia di qualità operativa.

## Protocollo

- Stesse 652 immagini di valutazione per entrambi i modelli: validation 206, PKLot holdout 6, TEST-ID 200, TEST-OOD interno 240.
- Confidenza primaria **0,35**, già impostata dall’utente in `scripts/batch.sh` prima del confronto. Nessuna nuova ricerca della soglia.
- Deduplicazione: IoU 0,50; copertura della scatola più piccola 0,50 solo fra viste diverse. Stessa pipeline: immagine intera più 15 crop.
- Matching delle annotazioni a IoU almeno 0,50, con classe corretta. Classi operative: person, car, motorcycle, bus, truck.
- Modelli ONNX eseguiti sequenzialmente sullo stesso Apple M4, CPU ONNX Runtime, 4 thread, ambiente `.venv-export`. Questa è una misura della qualità, non un benchmark prestazionale.
- V6 è il checkpoint neck-head seed42 già selezionato nella precedente validation. Nessun nuovo training o cambiamento dei modelli.
- Solo sulla validation è stata aggiunta la soglia fissa 0,47 come analisi descrittiva di sensibilità.

Precision indica la quota di rilevamenti corretti; recall la quota di oggetti annotati trovati. F1 combina precision e recall, con 1 come valore migliore. TP sono rilevamenti corretti, FP falsi positivi, FN oggetti mancati. I conteggi riguardano apparizioni nelle immagini, non veicoli unici.

## Auto: risultati pertinenti al caso d’uso

| Insieme | Immagini | Auto annotate | Precision base → v6 | Recall base → v6 | F1 base → v6 |
|---|---:|---:|---:|---:|---:|
| PKLot holdout | 6 | 1286 | 85.3% → 88.8% | 27.5% → 75.7% | 0.416 → 0.817 |
| raspberrypi2.local | 120 | 4828 | 91.5% → 84.9% | 40.5% → 48.7% | 0.561 → 0.619 |
| sitets02 | 120 | 730 | 96.1% → 91.9% | 83.4% → 85.9% | 0.893 → 0.888 |
| TEST-ID, scene miste | 200 | 74 | 47.4% → 29.3% | 60.8% → 59.5% | 0.533 → 0.393 |

| Insieme | TP base → v6 | FP base → v6 | FN base → v6 |
|---|---:|---:|---:|
| PKLot holdout | 354 → 974 | 61 → 123 | 932 → 312 |
| raspberrypi2.local | 1953 → 2352 | 182 → 417 | 2875 → 2476 |
| sitets02 | 609 → 627 | 25 → 55 | 121 → 103 |
| TEST-ID, scene miste | 45 → 44 | 50 → 106 | 29 → 30 |

- **PKLot:** 620 auto corrette in più e 62 falsi positivi in più. La recall aumenta di 48,2 punti percentuali, con un miglioramento anche della precision. Rimangono 312 auto mancate su 1.286.
- **raspberrypi2.local:** 399 auto corrette in più, ma 235 falsi positivi in più. La recall cresce di 8,3 punti e la precision scende di 6,5 punti. Il modello trova ancora meno della metà delle auto annotate: progresso utile, margine residuo ampio.
- **TS02:** 18 auto corrette in più e 30 falsi positivi in più. La F1 delle auto passa da 0,893 a 0,888: sostanziale parità con un lieve peggioramento del bilancio complessivo.
- **TEST-ID:** 44 auto corrette contro 45, con FP da 50 a 106. Il vantaggio sui parcheggi non si estende automaticamente alle scene generiche.

## Consistenza fra giornate

Confronto appaiato della F1 delle auto, aggregando TP/FP/FN all’interno di ciascun gruppo. Per le camere interne il gruppo è camera-giorno; il risultato è descrittivo e non un test di significatività statistica.

| Insieme | Gruppi migliori | Uguali | Peggiori |
|---|---:|---:|---:|
| PKLot, 6 immagini/gruppi | 6 | 0 | 0 |
| raspberrypi2.local, 11 giornate | 11 | 0 | 0 |
| TS02, 19 giornate | 8 | 3 | 8 |

Le giornate della stessa camera non sono domini indipendenti. Il conteggio mostra quanto il miglioramento sia distribuito fra le giornate disponibili, senza trasformarle in nuove camere di prova.

## Risultati su tutte le classi operative

| Insieme | TP base → v6 | FP base → v6 | FN base → v6 | Precision base → v6 | Recall base → v6 | F1 base → v6 |
|---|---:|---:|---:|---:|---:|---:|
| val | 254 → 247 | 412 → 321 | 235 → 242 | 38.1% → 43.5% | 51.9% → 50.5% | 0.440 → 0.467 |
| pklot_holdout | 354 → 978 | 89 → 136 | 959 → 335 | 79.9% → 87.8% | 27.0% → 74.5% | 0.403 → 0.806 |
| test_id | 258 → 245 | 390 → 295 | 185 → 198 | 39.8% → 45.4% | 58.2% → 55.3% | 0.473 → 0.498 |
| test_ood | 2700 → 3101 | 257 → 483 | 3185 → 2784 | 91.3% → 86.5% | 45.9% → 52.7% | 0.611 → 0.655 |

La F1 globale cresce su tutti e quattro gli insiemi. Su validation e TEST-ID ciò deriva da meno falsi positivi, con una piccola perdita di recall. Non si sommano gli insiemi in un unico punteggio: i parcheggi affollati dominerebbero il numero di oggetti e nasconderebbero i compromessi fra scene e classi.

## Effetto della deduplicazione e della confidenza

Il confronto principale usa il 50% per entrambi i modelli. La tabella seguente riporta la F1 globale della stessa validation, includendo i risultati storici all’80%. Sono stati verificati gli stessi hash dei modelli e del manifest, lo stesso ambiente Python e la sola modifica della costante di copertura nel codice runtime.

| Copertura cross-view | Confidenza | F1 base | F1 v6 |
|---|---:|---:|---:|
| 80% | 0.35 | 0.431 | 0.460 |
| 80% | 0.47 | 0.444 | 0.463 |
| 50% | 0.35 | 0.440 | 0.467 |
| 50% | 0.47 | 0.449 | 0.468 |

V6 supera la base con entrambe le confidenze e con entrambe le coperture. Il vantaggio non è spiegato soltanto dal passaggio alla deduplicazione al 50%. Questi dati non sono stati usati per scegliere una nuova soglia.

## Limiti e interpretazione

- PKLot contiene molte auto ma soltanto 6 immagini holdout, tutte dello stesso sito. L’entità del risultato non equivale a una prova su molti parcheggi indipendenti.
- Validation già usata per selezionare v6; PKLot e test storici già consultati durante lo sviluppo. Non sono nuovi test ciechi e non vengono presentati come tali.
- Il TEST-OOD interno comprende 120 immagini per camera. Raspberry contiene 4.828 delle 5.558 auto: il miglioramento aggregato non dimostra un miglioramento di TS02.
- Si misura questo artefatto v6 contro la base. La precedente valutazione della stabilità della ricetta, fra seed e contro v5, resta distinta e invariata.
- Per stabilire se la qualità sia sufficiente nell’uso operativo occorrono limiti accettabili di auto mancate/falsi positivi e una verifica su giornate nuove, senza scegliere modello o soglia su quelle immagini.

## Evidenze e verifica

Tutte le 8 valutazioni sono terminate con successo. Lo script di analisi verifica hash di codice, modelli e manifest, numero di immagini e soglie; ricostruisce i conteggi per immagine dagli errori e dalle annotazioni, verificando la corrispondenza esatta dei totali e di ogni classe con i JSON del valutatore.

- Protocollo congelato (`output/base-v6-coverage50-2026-09-14/protocol.json`, local evidence), comandi (`output/base-v6-coverage50-2026-09-14/commands.json`, local evidence), ambiente (`output/base-v6-coverage50-2026-09-14/environment.txt`, local evidence).
- Metriche complete e audit (`output/base-v6-coverage50-2026-09-14/analysis.json`, local evidence), confronto per gruppo (`output/base-v6-coverage50-2026-09-14/paired-groups.csv`, local evidence), confronto storico della copertura (`output/base-v6-coverage50-2026-09-14/validation-coverage-comparison.json`, local evidence).
- Analisi riproducibile (`output/base-v6-coverage50-2026-09-14/analyze.py`, local evidence): eseguire `.venv/bin/python output/base-v6-coverage50-2026-09-14/analyze.py` dalla radice del repository.
- I comandi di inferenza richiedono `YOLO_NUM_THREADS=4 YOLO_AUTOINSTALL=false`; per ripeterli usare nuove directory di output, preservando questi risultati.
- I JSON e gli `errors.csv` originali sono nelle sottocartelle `{val,pklot_holdout,test_id,test_ood}/{base,v6}` della directory di evidenze.
- **Attenzione alle immagini di validation:** il valutatore salva `errors.csv` e `worst-examples` alla migliore delle due soglie fornite, qui 0,47 per entrambi. I numeri primari a 0,35 vengono dai JSON; la ricostruzione appaiata esclude la validation. Nei tre benchmark storici gli errori e le immagini usano invece la sola soglia 0,35.

Non sono state modificate la pipeline, le impostazioni operative o le evidenze del precedente esperimento di training. La sola nuova logica è lo script di analisi conservato con questi risultati.
