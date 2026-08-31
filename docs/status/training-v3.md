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

Progetto CVAT `Phenocam privacy detector v3`, ID 1. Readback autenticato:
quattro task, label `person`, `car`, `motorcycle`, `bus`, `truck`, `ambiguous`,
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

## Prossima azione

Completare manualmente i task CVAT e comunicare l'ID completato. Il workflow si
ferma qui fino alla dichiarazione esplicita dell'utente.
