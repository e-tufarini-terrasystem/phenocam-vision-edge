# Stato training v3

## 2026-08-31 — Fase 0 completata

- Obiettivo: stabilizzare il modello v2 e il viewer del dataset.
- Commit modello/package: `0bacc0f125b243f596cd0b7a4418f03d1523f9e5`.
- Commit viewer successivi: `dcca648566ab4073952eb2e823851bc15e26ebb5`,
  `3b528c8a8d3f7a11607c6322969e61b7a5b4c978`.
- Checkpoint PT SHA-256:
  `49f3d38cb891c95069590edf76ec8b59ce399163f27f1e297831d042baffb912`.
- Modello ONNX SHA-256:
  `f62fd573fd6315fc34ddc3b0f22f4351be78f05db768babf6bca35b0c69d8962`.
- Contratto verificato: 80 classi COCO, input `1x3x640x640`, output
  `1x300x6`.
- Suite runtime: 237 test passati, 7 reference-image test saltati perché le
  fixture nominate non erano nella worktree temporanea.
- Suite dataset: 27 test passati.
- Shell syntax, `git diff --check` e package da commit temporaneo: passati.
- Package verificato con il solo modello runtime `yolo26n-v2.onnx`.
- Viewer verificato da Emanuele su 2.000 immagini: 1.244 positive e 756
  negative; campione positivo con quattro box e campione negativo senza box
  corretti.
- La revisione approfondita ha segnalato 22 probabili label issue nel dataset
  pubblico (depiction, statue/manichini, miniature e interni di veicoli). Non
  sono state corrette automaticamente: entrano nella successiva coda di label
  audit attribuita in CVAT.
- Decisione: v2 mantenuta come baseline di sviluppo; validation operativa
  ancora pending.

## Fase 1 — Contratto operativo

- Bicicletta isolata esclusa dal filtro privacy; il ciclista è protetto tramite
  `person`.
- Famiglia veicoli: `car`, `motorcycle`, `bus`, `truck`.
- Configurazione unica: `dataset/config/training-v3.json`; nessun path locale è
  incluso nell'artifact committabile.
- Revisione: `single_reviewer_waiver=true`.
- Test operativo: `sealed`.

## 2026-08-31 — Fase 2 completata

- Sorgente operativa letta senza scritture: 1.029 immagini accettate, zero
  rifiuti, 1.029 AppleDouble ignorati e zero symlink seguiti.
- Inventario SHA-256:
  `6f7d12f075664497fc5bee464893bd2ed9258886026ad8940129dd9ccdfa3810`.
- Split SHA-256:
  `f29cba3d89637be4988d3c82593e6f0585f3c641830fb72fa6a7cdb40889e777`.
- `operational_dev`: 403 immagini, 16 gruppi sito-giorno.
- `operational_mining`: 385 immagini, 14 gruppi sito-giorno.
- `sealed_test`: 241 immagini, 8 gruppi sito-giorno; stato `sealed`.
- Raspberry Pi camera: 3 giorni test, 6 dev, 5 mining.
- Site TS02: 5 giorni test, 10 dev, 9 mining.
- Leakage fra gruppi: zero.
- SSCD: 788 immagini dev/mining, 512 dimensioni; immagini sealed lette: zero.
- Artifact completi sotto `dataset/workspace/training-v3/`, ignorato da Git.
- Limite: due soli siti; il test non rappresenta l'intera rete PhenoCam.

## Coda di correzione label pubbliche

- Le 22 segnalazioni del controllo manuale sono state trascritte in un artifact
  locale con decisione, motivazione e revisore.
- Bundle CVAT dedicato: 22 immagini e 37 annotazioni correnti precaricate.
- La coda è separata dal pilot pubblico da 200 immagini e non ne consuma le
  quote.
- Venti casi sono correzioni probabili; due restano esplicitamente ambigui fino
  alla revisione in CVAT.

## Variazione strutturale locale

Durante la selezione interna è emersa la necessità di bilanciare i frame per
sito-giorno senza duplicare giorni con pochi campioni. Per mantenere ogni file
di orchestrazione entro 200 linee produttive, la responsabilità dello split è
separata prima della correzione:

- `dataset/builder/mining/split.py` (~55 linee produttive): split sealed e
  manifest degli embedding;
- `dataset/builder/mining/selection.py` (~160 linee produttive): sole selezioni
  pubbliche e interne.

## 2026-08-31 — Fase 3 completata

- Screening pubblico baseline e v2: 5.675 + 5.675 record completati, zero
  fallimenti.
- Screening interno baseline e v2: 788 + 788 record completati; 403 dev e 385
  mining per indice, zero record sealed.
- SHA-256 indici: pubblico baseline `ac60714e4c721da8aef63bc5ab3244c1ca68436507673ee21f1f16e963b007c4`,
  pubblico v2 `2f5423753d27658dd3cd6dff1f3f8a7e11133fce4c4630b7b287e4e75534fcdd`,
  interno baseline `02e31eb7fffef10260575f0da3804335dd54581b1f190d0646918650b3290eaa`,
  interno v2 `71b0804272b20b0e21d7a775c91d118af1375f777d1416d77d0ec33e37e76aad`.
- Pilot pubblico: 200 identità e hash unici, nessun riuso dal dataset esistente,
  massimo due immagini per gruppo, similarità SSCD massima `0,942577` rispetto
  al limite `0,95`. Contiene tutti i 20 confusori disponibili; undici hard
  negative completano la quota indicativa non soddisfatta.
- Selezione rappresentativa: 120 immagini uniche, 60 per sito, tutti i 16
  gruppi dev coperti, massimo 12 immagini per gruppo.
- Selezione informativa: 120 immagini uniche, 60 per sito, massimo 15 immagini
  per gruppo.
- SHA-256 selezioni: pubblico `d1c4e1405167cf40d37892f8078fba6ed839a4b02feca4e100fc12a68fce3786`,
  rappresentativo `3c4448c9239f57a2c9713ac44ccc881ca9bb9d78bb95bf2649f154df0bb4e268`,
  informativo `0ca9122e4ad8c9aecee9b57b81f5f95173bbd221e8d7ba79511ca00c89920ce1`.

## 2026-08-31 — Fase 4 al gate umano obbligatorio

Progetto CVAT `Phenocam privacy detector v3`, ID 1. Readback autenticato al
momento della creazione: quattro task, label `person`, `car`, `motorcycle`,
`bus`, `truck`, `ambiguous`,
attributi `occluded`, `truncated` e `vehicle_subtype`. Tutte le rotte task/job
seguenti rispondono HTTP 200; la superficie browser in-app non era disponibile
per una seconda verifica visuale.

| Task | ID | Immagini | Job | Box iniziali |
|---|---:|---:|---|---:|
| V3 public PhenoCam mining - pilot 200 | 5 | 200 | 19-22 | 1.724 |
| V3 public dataset label audit - reported 22 | 6 | 22 | 23 | 37 |
| V3 internal operational dev - representative 120 | 7 | 120 | 24-26 | 8.673 |
| V3 internal operational mining - informative 120 | 8 | 120 | 27-29 | 8.856 |

Hash SHA-256 dei receipt `bundle.json`, nello stesso ordine: pubblico
`35862ed36c09a9f9f1a383ee582ab243d6df3efd218bd287f8739f885dae7c79`,
label audit `1f8f6e8a0c6331c24a93f2c4edfebe61e9f8d0322463726603391930b17a9036`,
rappresentativo `756d87a2a1a46f3469383f4e11ae4e7a4998c810a7b447168f4450dc023926c3`,
informativo `8b38a5dbfd6c2fe22375b4ccdbfbdb5a31fdabcd79049a2f7c412fe8381ec385`.

La revisione live è iniziata subito dopo la creazione dei task; i conteggi dei
box correnti sono quindi mutabili e non sostituiscono quelli iniziali attestati
dai bundle. Nessun export, import nel dataset o training è stato avviato.

## 2026-08-31 — Esito pilot pubblico

- Task 5 completato: quattro job completati, 200 immagini, zero box e zero tag.
- Yield positivo: `0/200` (`0%`). Come previsto dal piano, non viene creata una
  seconda coda pubblica prima di correggere il ranking.
- Export COCO verificato: 200 immagini, zero annotazioni, SHA-256
  `66ce803c6f8615d8fb9aa8b47784729899afaae47dc16399ed0f4b6f2841d1c5`.
- Backup completo SHA-256:
  `ec601fc68e999777d450c08399965cffc82e3b835fc53c057c13c6e0af9428a0`.
- Diagnosi: soltanto 7 dei 200 frame avevano V2 `>=0,30`; 161 erano
  `crop_only` e la confidenza massima V2 mediana era `0,0193`. Il selettore
  considerava condivise anche predizioni sovrapposte a confidence floor `0,01`
  e applicava la diversità prima della confidenza.
- Controllo diagnostico non indipendente sui 15 positivi PhenoCam v2 noti:
  14/15 hanno V2 `>=0,30`, con mediana `0,7574`. Il campione include immagini
  di training e serve soltanto come sanity check del ranking.

## Feedback sulle preannotazioni interne

I bundle interni hanno importato tutte le predizioni di screening a soglia
`0,01`, producendo troppi falsi box. Il task 8 non era ancora iniziato; nel task
7 era iniziato soltanto il job 26. Sul primo piccolo campione modificato, il
gate `stessa classe + IoU >=0,5 + entrambi i modelli >=0,30` conserva 10 box
umani su 11 proposti, ma copre soltanto 10 dei 37 box correnti. Il risultato è
preliminare perché il job non era completato.

### Sostituti clean

L'utente ha autorizzato due task sostitutivi e la rimozione dei task superati.
Il filtro conserva soltanto box con stessa classe, IoU `>=0,5` e confidenza
`>=0,30` in entrambi i modelli. Prima dell'implementazione è stato separato per
mantenere l'orchestrazione CVAT sotto 200 linee produttive:

- `dataset/builder/mining/suggestions.py` (~50 linee produttive): unione
  diagnostica e gate di accordo ad alta precisione;
- `dataset/builder/mining/cvat.py` (~185 linee produttive): materializzazione
  atomica dei bundle.

Bundle e task verificati:

| Task attivo | ID | Immagini | Job | Box iniziali |
|---|---:|---:|---|---:|
| V3 internal operational dev - representative 120 - clean | 9 | 120 | 30-32 | 1.019 |
| V3 internal operational mining - informative 120 - clean | 10 | 120 | 33-35 | 1.115 |

- Receipt SHA-256: rappresentativo
  `8f5a20d585b5e7bc6f2846c5065e45a2963718190f9d5ead630239b97e12e324`;
  informativo
  `b69421f659ac7f00e83fbbc04077c53c121fc431d4a170370a3f40fc7d846ea9`.
- COCO iniziale SHA-256: rappresentativo
  `2d034684876f1b66044c4c31ae54eed8d5c1c1b81db24cb43170232dc7363ba5`;
  informativo
  `1af9434a0c5639ffae48e63d4ca67645ced48ee9b5fee3996877d2dbd0921a59`.
- Le 1.019 e 1.115 box sono proposte, non ground truth: vanno eliminate quando
  errate e integrate quando manca un target reale.

## Pulizia dei task CVAT

La lista CVAT contiene ora soltanto i task attivi 9 e 10. Sono stati rimossi:

- task 7 e 8, sostituiti dalle versioni clean;
- task 5 e 6, già completati ed esportati;
- task storici 2-4, già esportati e importati nel dataset.

Tutte le rimozioni sono recuperabili. I backup completi sono sotto
`dataset/workspace/annotation/exports/` e sono stati verificati con il test
dell'archivio. Per i task superati 7 e 8, gli export COCO conservano
rispettivamente 120 immagini/8.540 box e 120 immagini/8.856 box; SHA-256 backup
`400363c61826e70101d97938af764f1015558b896da090dec95a1cdf776d771b` e
`f0beacba6d927a2561c6225197e71885be705804ac16b7c46b56d56fd48f6fd6`.
L'audit pubblico completato conserva 22 immagini e 4 box; SHA-256 COCO
`36f054124c522d1cf42efab0a6dddac01f3d0f857663fce45e1143dd3aedaff5`
e backup
`60c978240f107ebe58d5759e140a8211aae14c4d94e2bce41979a78d09574dc1`.

## Valutazione di un nuovo batch PhenoCam

Dopo le esclusioni originarie restano 3.361 candidati. Il gate stretto usato
per i task clean trova soltanto 58 immagini con accordo baseline/v2; 2 erano
già nel pilot e sono state marcate negative, quindi ne restano 56 distribuite
su 10 siti e 49 gruppi. Dieci hanno confidenza congiunta `>=0,70`.

Il controllo visuale dei dieci candidati più forti mostra soprattutto veicoli
molto lontani, spesso alti 7-15 pixel. Anche uno dei due candidati già respinti
dal pilot contiene lo stesso tipo di veicoli lontani. Poiché la guida corrente
richiede di annotare ogni target reale visibile ma il feedback umano li ha
trattati come negativi, non viene creato ora un nuovo task pubblico: prima va
resa esplicita la regola sui target piccoli/lontani. Un nuovo pilot avrebbe
altrimenti ground truth incoerente e aggiungerebbe lavoro manuale senza una
decisione utilizzabile.

## Prossima azione

Completare i task clean 9 e 10. Prima di riaprire il mining pubblico, decidere
se i veicoli riconoscibili ma molto lontani devono essere target privacy oppure
negativi operativi; solo dopo creare un eventuale pilot stretto da 20 immagini.
