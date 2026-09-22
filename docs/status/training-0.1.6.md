# Training 0.1.6 — protocollo e risultati

> Historical artifact labels use normalized model versions, not filesystem paths.
> Exact commands, identifiers and paths remain in the original document:
> `git cat-file blob d1816f34078214feb2227b9fcd53e16237822e21` from the
> [source snapshot](https://github.com/e-tufarini-terrasystem/phenocam-vision-edge/tree/cc63857c12c5553c2e3451854863edf7c0705e2a).

> Historical record for this iteration; use the [current guide](../development.md#current-workflow).
> Commands refer to the original workflow; ignored artifacts require local evidence.

Nota di consolidamento: il modello 0.1.6 è distribuito come
`yolo26n-phenocam`, versione `0.1.6`, con byte PT e ONNX invariati e stato
**sperimentale**. La scheda runtime contiene soltanto identità, versione e hash
ONNX; i riferimenti di provenienza sono conservati qui:

- Ricevuta storica del modello 0.1.6 al commit
  `da14b7d6317d1d158859999947b819f26c332f11`.
- SHA-256 della ricevuta:
  `1725300acc3bb4ce5189edc4631c65e0cf07be5cfd6b6b18df3fc01631697f1f`.
- SHA-256 di `models/yolo26n-phenocam.pt`:
  `76ca4a80abd559c9d5df378052ad31480bf42f8af301372733c2b6ddb602fed0`.
- Valutazione successiva: [confronto del 15 settembre 2026](model-comparison-2026-09-15.md).

Nota successiva: il 14 settembre 2026 la copertura per deduplicazione tra viste
è stata portata dall'80% al 50% su richiesta dell'utente. Le misure di questo
rapporto restano riferite al runtime all'80%, conservato con i relativi hash
e snapshot; non sono state rimisurate con la nuova soglia.

Richiesta dell'11 settembre 2026: quattro training al massimo dal checkpoint
base `models/yolo26n.pt`, sull'intero dataset 0.1.6 (`dataset.yaml`).
I quattro training, la selezione ONNX e le repliche sono completati.
Il candidato è congelato come **sperimentale**: falliscono incremento medio
F1, stabilità minima F1 e limite di perdita mAP Open Images. Artefatti prodotti
e verificati il 14 settembre 2026: checkpoint PT, ONNX e metadati sono conservati
nella [revisione originale](https://github.com/e-tufarini-terrasystem/phenocam-vision-edge/tree/da14b7d6317d1d158859999947b819f26c332f11).
Il modello operativo resta invariato. Nessun push o pubblicazione.

## Avanzamento verificato

Tutti i training, le verifiche e i benchmark finali sono completati.
Le metriche del trainer nella tabella seguente non sono l'F1 della pipeline
di selezione.

| Ricetta | Seed | Epoche | Miglior epoca | mAP50–95 trainer | Durata | Picco MPS | Picco RSS |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| head, freeze 23 | 42 | 30 | 21 | 0,54911 | 4.062,61 s | 4,79 GB | 3,92 GB |
| neck-head, freeze 10 | 42 | 13 | 3 | 0,52146 | 2.362,67 s | 5,65 GB | 4,57 GB |
| neck-head, freeze 10 | 17 | 26 | 16 | 0,53512 | 4.120,98 s | 5,83 GB | 3,44 GB |
| neck-head, freeze 10 | 73 | 28 | 18 | 0,54959 | 4.931,81 s | 4,82 GB | 3,09 GB |

La ricevuta head/42 (output del training 0.1.6 (`receipts/head-seed42.json`), local evidence)
è `completed`. Verificati tutti gli otto checkpoint conservati, le 30 righe
finite del CSV e l'hash invariato del base. `best.pt` ha SHA-256
`f7d79cd121dbd832c2dfedcec8707d1c8e30be34334c9ee97bc6cf5436e5d460`;
`last.pt` ha SHA-256
`50beafaad4e21774434650b4d2f49d2fee190e0d08b3d73fbba3a923011509bb`.

La ricevuta neck-head/42 (output del training 0.1.6 (`receipts/neck-head-seed42.json`), local evidence)
è `completed`: early stopping dopo dieci epoche senza superare il massimo
dell'epoca 3. Verificati tutti i cinque checkpoint, le 13 righe finite del CSV
e la terminazione effettiva del processo. `best.pt` ha SHA-256
`76ca4a80abd559c9d5df378052ad31480bf42f8af301372733c2b6ddb602fed0`;
`last.pt` ha SHA-256
`8add3f47cf5fe90121c4356c15c2bc48d1d7d5b2eeb54f323436566b25a30408`.
Entrambi i run confermano epoca iniziale 0, optimizer senza stato, scheduler
da −1 e 2.020 immagini train. La valutazione del base è iniziata dopo la
conclusione del secondo run.

Riferimenti rimisurati sulle 206 immagini validation, con il runtime del
worktree e la stessa griglia di 51 soglie. La mAP è la misura standard PT;
precision, recall e F1 provengono dalla pipeline ONNX a 16 viste.

| Modello | Soglia | Precision | Recall | F1 ONNX | mAP50–95 | mAP Open Images |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| base | 0,45 | 0,40641 | 0,49284 | 0,44547 | 0,52545 | 0,53277 |
| 0.1.5 | 0,48 | 0,42500 | 0,48671 | 0,45377 | 0,53165 | 0,53890 |

Rapporti completi, con risultati per classe e sorgente:
base ONNX (output del training 0.1.6 (`evaluation/base-onnx/runtime/runtime-metrics.json`), local evidence),
0.1.5 ONNX (output del training 0.1.6 (metriche runtime ONNX del modello 0.1.5), local evidence),
base mAP (output del training 0.1.6 (`evaluation/base-pt/standard/standard-metrics.json`), local evidence),
0.1.5 mAP (output del training 0.1.6 (metriche standard PT del modello 0.1.5), local evidence).
I recall di riferimento 0.1.5 sono 1,0 su PhenoCam (2/2 box) e 0,35556 su PKLot
(16/45 box); i tre seed 0.1.6 devono rispettarli individualmente.

### Selezione iniziale

| Ricetta / checkpoint, seed 42 | Soglia | Precision | Recall | F1 ONNX | mAP Open Images |
| --- | ---: | ---: | ---: | ---: | ---: |
| head / best | 0,34 | 0,32316 | 0,47648 | 0,38512 | 0,54168 |
| head / last | 0,32 | 0,33562 | 0,50102 | 0,40197 | 0,53748 |
| neck-head / best | 0,45 | 0,47479 | 0,46217 | **0,46839** | 0,52107 |
| neck-head / last | 0,30 | 0,32237 | 0,50102 | 0,39231 | 0,51261 |

La scelta registrata (output del training 0.1.6 (`candidate.json`), local evidence) è
`neck-head/best`, soglia 0,45. Un controllo indipendente dei 204 abbinamenti
checkpoint/soglia conferma il massimo e le regole di parità; tutti i rapporti
contengono 206 immagini e gli stessi hash del codice effettivo.
La F1 del seed 42 supera 0.1.5 di 0,01463, ma la perdita mAP Open Images è
0,01783, oltre il massimo 0,01. Il gate complessivo non può quindi passare:
il risultato sarà sperimentale anche se le repliche fossero migliori.
Il recall PhenoCam resta 1,0 e quello PKLot sale a 0,66667 (30/45 box).

Il seed 17 è completato: 26 epoche, early stopping, massimo all'epoca 16.
La ricevuta (output del training 0.1.6 (`receipts/neck-head-seed17.json`), local evidence) e tutti
gli otto checkpoint sono stati verificati, insieme alle 26 righe finite del
CSV e alla terminazione del processo. `best.pt` ha SHA-256
`8850773e86c56b7f7afa4d51a0a7ec7d31fa42319b1b96b8e161736d33e903bd`.
È partito dal base con stato optimizer vuoto, scheduler −1 e tutti i 2.020
esempi; i parametri coincidono con quelli del seed 42, salvo il seed.
La sua valutazione ONNX (output del training 0.1.6 (`evaluation/neck-head-seed17-best-onnx/runtime/runtime-metrics.json`), local evidence)
è completa sulle 206 immagini, alla sola soglia 0,45: precision 0,35788,
recall 0,42740, F1 0,38956 (−0,06420 contro 0.1.5). Il recall PhenoCam resta
1,0; PKLot sale a 0,77778 (35/45 box). La mAP Open Images è 0,53061
(perdita 0,00830, entro il limite). Fallisce quindi anche il criterio di
assenza di repliche con F1 sotto 0.1.5: la migliore F1 del seed 42 non è stabile
sui primi due seed.

Il seed 73 è completato, quarto e ultimo slot: 28 epoche, early stopping,
massimo all'epoca 18. Verificati nuovamente base, optimizer senza stato,
scheduler −1, 2.020 immagini e parametri identici salvo il seed. La
ricevuta (output del training 0.1.6 (`receipts/neck-head-seed73.json`), local evidence), tutti gli
otto checkpoint e le 28 righe finite del CSV sono stati verificati.
`best.pt` ha SHA-256
`32cb8fc5f469cd5f1f7af2b38456b56985c1b551ecb08eaf04754e0ce5cbbff3`.
La valutazione usa `best` e soglia 0,45, senza una nuova ricerca delle soglie.

Il seed 73 ha attraversato lunghe pause del computer tra l'11 e il 14 settembre:
234.608,52 s di calendario (65,17 ore), contro 4.931,81 s (82,20 minuti)
registrati dal timer monotono del training. Alla ripresa sono stati verificati
lo stesso PID 72087 e l'avanzamento reale; nessun restart o ripristino di
optimizer/scheduler. Le durate in tabella sono quelle del timer monotono e
non costituiscono un benchmark di prestazioni confrontabile tra condizioni
termiche e sospensioni diverse.

### Stabilità e accettazione congelate

| Seed | Precision ONNX | Recall ONNX | F1 ONNX | Differenza F1 vs 0.1.5 | mAP Open Images | Recall PhenoCam | Recall PKLot |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 42 | 0,47479 | 0,46217 | 0,46839 | +0,01463 | 0,52107 | 1,0 | 0,66667 |
| 17 | 0,35788 | 0,42740 | 0,38956 | −0,06420 | 0,53061 | 1,0 | 0,77778 |
| 73 | 0,42582 | 0,45194 | 0,43849 | −0,01527 | 0,53837 | 1,0 | 0,80000 |

Tutti i seed usano soglia 0,45. F1 media 0,43215, differenza media contro
0.1.5 −0,02162 e deviazione standard campionaria 0,03980. Falliscono incremento
medio ≥0,005, nessun seed sotto 0.1.5 e perdita mAP Open Images ≤0,01 per ogni
seed. Passano i recall PhenoCam e PKLot per tutti e tre i seed.

La selezione congelata (output del training 0.1.6 (`frozen-selection.json`), local evidence)
mantiene il seed 42, checkpoint `neck-head/best`, soglia 0,45, stato
`experimental`, senza promozione. PT SHA-256
`76ca4a80abd559c9d5df378052ad31480bf42f8af301372733c2b6ddb602fed0`;
ONNX SHA-256
`72521182fa0c90fec60fb1cd9f3b2ac113d16de476aaeea3a328508b7b2d0b30`.
Il congelamento precede l'apertura dei benchmark storici. Training completato
non significa miglioramento dimostrato.

### Validation per classe e sorgente

Confronto del modello consegnato (seed 42, soglia 0,45) con 0.1.5 (soglia 0,48).
P/R/F1 sono della pipeline ONNX; mAP è la misura standard PT a IoU 0,50–0,95.
I rapporti JSON conservano anche tutti i conteggi e le metriche degli altri seed.

| Classe | F1 0.1.5 | P 0.1.6 | R 0.1.6 | F1 0.1.6 | mAP 0.1.5 | mAP 0.1.6 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| bus | 0,59574 | 0,60000 | 0,66667 | 0,63158 | 0,80217 | 0,70269 |
| car | 0,39024 | 0,30102 | 0,49580 | 0,37460 | 0,43116 | 0,49825 |
| motorcycle | 0,58333 | 0,70000 | 0,70000 | 0,70000 | 0,66163 | 0,54184 |
| person | 0,46130 | 0,62037 | 0,43791 | 0,51341 | 0,26200 | 0,24320 |
| truck | 0,41935 | 0,29167 | 0,26923 | 0,28000 | 0,50327 | 0,55177 |

Bicycle, esclusa dalla metrica operativa ma presente nella mAP e nelle 80
classi COCO: mAP50–95 0.1.5 0,52965, 0.1.6 0,56973. Le altre classi COCO non
hanno annotazioni nella validation.

| Sorgente | F1 0.1.5 | P 0.1.6 | R 0.1.6 | F1 0.1.6 | mAP 0.1.5 | mAP 0.1.6 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Open Images | 0,45036 | 0,43991 | 0,43891 | 0,43941 | 0,53890 | 0,52107 |
| PhenoCam | 0,50000 | 0,50000 | 1,00000 | 0,66667 | 0,32405 | 0,32384 |
| PKLot | 0,50000 | 0,96774 | 0,66667 | 0,78947 | 0,56495 | 0,57021 |

Il guadagno F1 del seed 42 dipende soprattutto da PKLot; su Open Images
scendono sia F1 sia mAP. Questo risultato non giustifica una promozione.

### Verifica del modello congelato

Export statico ONNX opset 20, input `[1,3,640,640]`, output `[1,300,6]`,
80 classi COCO identiche al base, ONNX checker superato e valori finiti.
Il rapporto di verifica (output del training 0.1.6 (`verification/contract.json`), local evidence)
confronta PT CPU e ONNX su tre immagini positive validation, una per sorgente:
48 viste e 101 rilevazioni abbinate sopra confidenza 0,20. IoU minimo
0,99262 (limite 0,99), differenza massima di confidenza 0,000004143
(limite 0,001). Prodotti annotated/privacy generati dal `process_image`
pubblico e dimensioni verificate; controllata visivamente l'annotazione PKLot.

Il confronto completo sulle 206 immagini a soglia 0,45 dà F1 PT/MPS
0,46791 e ONNX 0,46839: stessi 226 TP e 263 FN, 251 FP PT contro 250 ONNX.
La parità numerica non è identità bit per bit; la selezione e i gate usano
sempre le misure ONNX. Il test dell'API pubblica conserva la soglia operativa
0,47; il valutatore applica esplicitamente la soglia sperimentale 0,45.

### Benchmark storici dopo il congelamento

Questi insiemi erano già stati usati nello sviluppo: non sono nuovi test
incontaminati. Soglie fissate sulla validation: base 0,45, 0.1.5 0,48, 0.1.6 0,45.
Precision, recall e F1 sono ONNX a 16 viste; mAP50–95 è standard PT.

| Benchmark | Modello | Immagini | Precision | Recall | F1 | mAP50–95 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| PKLot holdout | base | 6 | 0,78827 | 0,18431 | 0,29877 | 0,08579 |
| PKLot holdout | 0.1.5 | 6 | 0,81437 | 0,20716 | 0,33030 | 0,09184 |
| PKLot holdout | 0.1.6 | 6 | 0,88966 | 0,68774 | 0,77577 | 0,12078 |
| TEST-ID | base | 200 | 0,39746 | 0,56433 | 0,46642 | 0,57809 |
| TEST-ID | 0.1.5 | 200 | 0,41525 | 0,55305 | 0,47435 | 0,58256 |
| TEST-ID | 0.1.6 | 200 | 0,48124 | 0,49210 | 0,48661 | 0,54575 |
| TEST-OOD | base | 240 | 0,93114 | 0,40901 | 0,56836 | 0,13260 |
| TEST-OOD | 0.1.5 | 240 | 0,93506 | 0,41105 | 0,57106 | 0,13421 |
| TEST-OOD | 0.1.6 | 240 | 0,88889 | 0,48530 | 0,62783 | 0,13795 |

Su TEST-OOD, separando lo stress estremo `raspberrypi2.local`, F1 0.1.5/0.1.6
è 0,83760/0,84164 nel resto del dominio e 0,51071/0,58272 nello stress.
I risultati storici favorevoli non annullano i gate falliti sulla validation
e sui seed. Su TEST-ID la F1 sale, ma recall e mAP scendono.
Rapporti completi per classe e sorgente in output del training 0.1.6 (`final/`) e
nei [metadati consegnati](https://github.com/e-tufarini-terrasystem/phenocam-vision-edge/tree/da14b7d6317d1d158859999947b819f26c332f11).

### Audit e consegna

Il workflow è terminato con codice 0. Tempo monotono totale 17.950,14 s
(4 h 59 min 10 s), di cui 15.478,07 s (4 h 17 min 58 s) nei quattro
training completati. Il tentativo iniziale fallito è conservato separatamente;
le pause del seed 73 rendono molto maggiore il tempo di calendario.

- Audit indipendente (output del training 0.1.6 (`audit-final.json`), local evidence): quattro
  inizializzazioni nuove dal base, 29 checkpoint verificati, 204 combinazioni
  di selezione e 35 valutazioni complete; gate ricalcolati e cronologia del
  congelamento verificata per tutti i benchmark.
- Integrità finale (output del training 0.1.6 (`integrity-final.json`), local evidence): fingerprint
  atteso invariato, 2.232 immagini decodificate, 7.119 label valide, 2.004
  immagini 0.1.5 ereditate identiche; modelli precedenti, codice e ambiente
  coincidono con gli hash del preflight.
- Test finali (output del training 0.1.6 (`logs/tests-final.log`), local evidence): 258 test,
  7 skip, esito positivo in 16,16 s, con autoinstall disabilitato.
- I sette cache preesistenti 0.1.3/0.1.5 sono identici al backup. I tre cache
  derivati generati da 0.1.6 sono archiviati fuori dal dataset in
  `provenance/dataset-caches-after-evaluation/`, con manifest e hash.
- I file PT e ONNX consegnati coincidono byte per byte con il modello
  congelato e verificato. Il JSON include provenienza, configurazione,
  ricevute, soglia, metriche, gate, verifica e limiti; `automatic_promotion`
  e `improvement_demonstrated` sono entrambi `false`.

## Protocollo fissato prima dell'esecuzione

Due ricette iniziali, seed 42: `head` congela i layer 0–22; `neck-head`
congela 0–9 (adatta anche l'ultimo blocco C2PSA del backbone). Entrambe usano AdamW, LR 0,0001, LR finale relativo 0,1,
30 epoche, patience 10, warmup 2, batch 16, 640×640, MPS disponibile,
workers 0 e tutti i 2.020 esempi train. AMP disattivato esplicitamente su MPS.
Il confronto riprende l'evidenza 0.1.5: LR 0,001 con adattamento ampio aveva
peggiorato le metriche; 0.1.6 verifica un adattamento più ampio con LR conservativo.

Augmentation fissate in implementazione del training 0.1.6 (`experiments.json`): mosaic 0,5,
close_mosaic 5, HSV 0,015/0,7/0,4, translate 0,1, scale 0,5, fliplr 0,5;
mixup, copy-paste, cutmix, rotazione, shear, perspective e flip verticale nulli.
Optimizer e scheduler sono nuovi: nessun resume da run precedenti.
Si conserva la non-deterministicità residua MPS già osservata in 0.1.5.

Si valutano i checkpoint `best` (scelto dal trainer sulla mAP validation) e
`last`. La metrica primaria per scegliere ricetta, checkpoint e soglia è
l'F1 ONNX della pipeline completa: intera immagine più 15 crop, deduplicatore
del worktree attuale, matching IoU 0,5 e classi operative del repository.
La bicycle rimane nelle 80 classi COCO e nella mAP standard, ma non nelle
cinque classi operative della metrica runtime storica.

Griglia identica per base, 0.1.5 e candidati: 0,20–0,70 con passo 0,01.
Parità: recall maggiore, soglia minore, ordine ricetta, `best` prima di `last`.
Si ripete soltanto la ricetta vincente con seed 17 e 73 e si mantiene sia la
regola di checkpoint sia la soglia del seed 42. L'artefatto consegnato resta
il seed 42: nessuna selezione del seed sulla base dei risultati.
Non sono previsti soup, interpolazione o ampliamenti della griglia.

Accettazione: incremento medio F1 ONNX sui tre seed ≥0,005 contro 0.1.5
rimisurato; nessun seed sotto 0.1.5; per ciascun seed perdita mAP50–95 Open
Images ≤0,01 e recall PhenoCam/PKLot validation non inferiore a 0.1.5.
Anche un candidato respinto viene esportato e consegnato come sperimentale.
La scelta viene congelata prima dei benchmark storici PKLot holdout,
TEST-ID e TEST-OOD; quei risultati non possono cambiare modello o soglia.

Validation 0.1.6 ereditata da 0.1.5: 128 Open Images, 75 PhenoCam (solo due box
positive) e tre PKLot. Mancano giorni annotati indipendenti PhenoZero/TS02:
questo esperimento non dimostra il miglioramento sulle nuove camere.

## Struttura e invarianti

Nel dominio esistente implementazione del training 0.1.6: `preflight.py` (~190 linee),
`train.py` (~170), `evaluation.py` (~180), `selection.py` (~160),
`artifacts.py` (~180); stime produttive senza test. `experiments.json`
registra la griglia. Il builder 0.1.6 già presente resta invariato.
`workflow.py` (~140 linee) possiede la sequenza e registra i PID dei processi;
non riavvia automaticamente un training incompleto.
Ogni modulo di orchestrazione resta entro 200 linee produttive.

L’output del training 0.1.6 conserva snapshot del codice effettivo, diff iniziale,
hash di tutti i modelli precedenti, ambiente fissato, ricevute, log,
checkpoint e rapporti. Nessun push o promozione automatica.
Prima delle valutazioni sono stati copiati e verificati anche i cache derivati
preesistenti 0.1.3/0.1.5 in `provenance/dataset-caches-before-evaluation/`, insieme
all'inventario dei cache generati dal training 0.1.6. Servono a preservare anche
questi file durante il controllo finale, senza toccare immagini o annotazioni.
Il rischio principale è il forgetting; il gate include la sorgente generale.
Le nuove responsabilità separano integrità, training, misure e consegna
senza modificare la pipeline operativa.

## Ambiente e comandi

Python 3.13, torch 2.12.1, torchvision 0.27.1, Ultralytics 8.4.48,
ONNX 1.19.0, onnxslim 0.1.71, runtime e builder nelle versioni del progetto.
Tutte le dipendenze transitive effettive sono registrate nel lock locale
output del training 0.1.6 (`provenance/environment.txt`) prima del training.

Per i comandi storici, consultare il documento originale indicato sopra.

Per ricreare esattamente l'ambiente usare il lock delle dipendenze transitive:
il comando storico documentato nella fonte originale.
L'avvio iniziale presuppone output non ancora popolato. Il preflight rifiuta
di sostituire la provenienza esistente; il workflow riusa soltanto run e
valutazioni complete con hash corrispondenti. Conservare gli artefatti del
presente esperimento quando si prepara una nuova riproduzione.
Riferimenti verificati: [training e resume Ultralytics](https://docs.ultralytics.com/modes/train/),
[export statico](https://docs.ultralytics.com/modes/export/),
[abbinamenti torch/torchvision](https://pytorch.org/get-started/previous-versions/).

## Ripristino della compatibilità MPS

Il primo tentativo `head/42` con torch 2.8.0 si è arrestato dopo 149,44 s,
al batch 122/127 della prima epoca: indicizzazione booleana TAL con dimensioni
incoerenti (29226 contro 15882). Nessuna epoca completata, nessun checkpoint
prodotto e nessun risultato usato per selezionare la griglia. Processo verificato
terminato con codice 1 prima di qualsiasi recupero.

Il guasto è coerente con [PyTorch #178079](https://github.com/pytorch/pytorch/issues/178079)
e [#181867](https://github.com/pytorch/pytorch/issues/181867), bug intermittenti
MPS dell'indicizzazione corretti nella serie 2.12. Il piccolo stress test locale
di 300 iterazioni non ha riprodotto il guasto in 2.8.0: non si dichiara quindi
una riproduzione isolata del difetto. Si fissa la coppia ufficiale torch 2.12.1 /
torchvision 0.27.1 e si ripete il controllo contro CPU prima del training.

Il tentativo fallito, il primo preflight e il vecchio lock restano in
output del training 0.1.6 (`attempts/head-seed42-torch2.8/`). Il retry riguarda lo stesso
slot ricetta/seed, da base e con optimizer nuovo; non aggiunge una ricetta o
una replica alla griglia di quattro esperimenti. Tutte le misure confrontabili
saranno effettuate con l'ambiente corretto, prima dell'apertura dei benchmark.

## Verifiche prima della sequenza definitiva

- Dataset ricontrollato integralmente dopo il ripristino dell'ambiente:
  fingerprint `e1491ab0e3bc073c167026f7e0dd0057c86bede700baa2122a8d3a4126dbfddc`,
  2.232 immagini decodificate uniche, 7.119 label geometricamente valide,
  2.004 immagini 0.1.5 e relative annotazioni identiche, test storici esclusi.
- Base PT SHA-256:
  `9b09cc8bf347f0fc8a5f7657480587f25db09b34bf33b0652110fb03a8ad4fef`.
- PyTorch 2.12.1: 300 iterazioni dell'indicizzazione TAL contro CPU passate
  in 6,31 s; nessun training o aggiornamento dei pesi durante questa diagnosi.
- Export preliminare della copia del base verificato con ONNX checker e
  runtime del repository: input `[1,3,640,640]`, output `[1,300,6]`, 80 classi,
  valori finiti. Artefatto diagnostico in `exports/environment-base/`.
- `YOLO_AUTOINSTALL=false .venv-export/bin/python -m unittest discover -s tests`:
  258 test, 7 skip, esito positivo in 14,29 s. La prima esecuzione nell'ambiente
  runtime minimale aveva segnalato l'assenza di Ultralytics per i test 0.1.5;
  la suite completa richiede l'ambiente training/export.
- Il primo giro di test aveva installato automaticamente `pi-heif`; rimosso
  perché non richiesto dall'esperimento. La suite passa anche senza di esso,
  con autoinstall disabilitato. Il lock finale conserva solo le dipendenze
  effettivamente necessarie; tutti i processi del workflow disabilitano autoinstall.
- Tutti i nuovi moduli restano sotto le stime dichiarate e sotto 200 linee
  produttive. Le modifiche iniziali dell'utente restano preservate.

Monitoraggio riproducibile: output del training 0.1.6 (`workflow.json`) identifica il
processo corrente e il figlio; `monitor/<ricetta>-seed<seed>.json` registra
l'ultima epoca, loss, metriche, memoria MPS e RSS. I CSV nei run conservano
l'intera serie temporale. Prima di un recupero verificare il PID reale e il
codice d'uscita del processo; la sola presenza di una ricevuta non prova attività.
