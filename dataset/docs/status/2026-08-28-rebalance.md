# Ribilanciamento dopo l’audit CVAT — 28 agosto 2026

> Historical record for this iteration; use the [current guide](https://github.com/e-tufarini-terrasystem/phenocam-vision-edge/blob/main/dataset/README.md).
> Commands refer to the original workflow; ignored artifacts require local evidence.

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

## Esito del task supplementare e riconciliazione

Il task 4 è stato completato con 38 immagini, 162 box finali e zero track. Le
correzioni hanno portato temporaneamente `bicycle` a 199. La riconciliazione ha
bloccato tutti i frame revisionati e ha sostituito un solo frame non revisionato:

- ingresso: `open_images:test:9104bcc6457fe170`, una bicicletta;
- uscita: `open_images:test:e2483da930cb51ed`;
- screening del nuovo frame: completato, nessun rifiuto.

I conteggi congiunti risultanti sono person 3.051, car 772, truck 270, bicycle
200, motorcycle 204 e bus 180. Il massimo per gruppo di provenienza resta cinque.

## Code negative finali

Il contratto richiede 50 negativi Open Images e 706 PhenoCam. Dei 706 PhenoCam,
335 riutilizzano la prima verifica già svolta in CVAT; 371 richiedono una nuova
prima verifica. Tutti i 706 richiedono una seconda verifica cieca da parte di
una persona diversa. La selezione dei 371 usa gli embedding SSCD fissati per
massimizzare la diversità rispetto ai 335 già confermati.

Il primo ciclo manuale ha classificato 17 dei 50 Open Images e 5 dei 706
PhenoCam come `target_present`. Inoltre entrambi i round PhenoCam risultavano
firmati da Emanuele, quindi non erano indipendenti. Le 22 immagini sono state
escluse e sostituite deterministicamente. Le pagine brevi di risoluzione
contengono 17 Open Images e 5 PhenoCam; soltanto dopo la loro conferma viene
generato il round B definitivo per il secondo revisore.

Nella prima coda di sostituzione, tutti i 5 PhenoCam sono stati confermati
negativi; 16 dei 17 Open Images sono stati confermati e uno conteneva ancora un
target. Il builder ha quindi prodotto una sola ultima sostituzione Open Images,
da confermare prima di creare il round B definitivo.
