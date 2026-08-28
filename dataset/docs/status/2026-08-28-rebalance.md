# Ribilanciamento dopo l’audit CVAT — 28 agosto 2026

## Evidenza osservata

I due task positivi sono stati esportati, sottoposti ad audit e importati senza
track:

- Open Images: 202 frame revisionati, 675 box, nessun frame vuoto;
- PhenoCam: 350 frame revisionati, 15 positivi reali, 335 frame vuoti, 59 box.

Il tasso di falsi positivi PhenoCam rendeva impossibile conservare il contratto
provvisorio 350 positivi / 750 negativi senza inventare ground truth o acquisire
un nuovo pool non ancora verificato.

## Decisione approvata

Il dataset resta di 2.000 frame sorgente unici:

| Fonte | Positivi | Negativi | Totale |
|---|---:|---:|---:|
| Open Images V7 | 1.229 | 50 | 1.279 |
| PhenoCam v3 | 15 | 706 | 721 |
| **Totale** | **1.244** | **756** | **2.000** |

Le soglie restano 900 persone, 750 car, 270 truck, 200 biciclette, 200 moto e
180 bus.

## Selezione supplementare riproducibile

Un problema binario intero seleziona esattamente 379 positivi dal pool Open
Images già scaricato e deduplicato. I vincoli includono le soglie congiunte,
l’esclusione dei 900 frame della selezione base e un massimo globale di cinque
frame per gruppo di provenienza.

Il risultato verificato contiene 1.251 box supplementari e porta i conteggi
congiunti a: person 3.019, car 768, truck 270, bicycle 200, motorcycle 205 e bus
180. Tutti gli scostamenti dalle soglie sono zero; il gruppo più numeroso ha
cinque frame.

Lo screening baseline non ha trovato conflitti. La politica di audit conserva
le annotazioni umane Open Images come sorgente e invia a CVAT un campione
deterministico che copre almeno il 10% dei box: 38 immagini e 126 box nel task 4.
Il modello serve solo a stabilire priorità e non è ground truth.
