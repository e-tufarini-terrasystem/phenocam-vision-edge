<!--
Scopo: indicizzare la documentazione tecnica del progetto.
Responsabilita: orientare il lettore senza duplicare i contenuti dei documenti collegati.
Contesto: e il punto di ingresso per comprendere, usare e verificare il software.
-->

# Documentazione tecnica

Questo progetto annota una singola immagine locale usando un modello YOLO26n
end-to-end in formato ONNX. Il percorso di deploy e pensato per Raspberry Pi 3
a 64 bit e usa soltanto NumPy, ONNX Runtime e Pillow.

## Percorso di lettura

1. [Architettura](architettura.md) descrive componenti, confini e flusso
   complessivo del comando.
2. [Inferenza](inferenza.md) documenta nel dettaglio preparazione delle viste,
   contratto ONNX, coordinate, fusione delle detection e rendering.
3. [Operativita](operativita.md) spiega installazione, configurazione, uso
   singolo, batch, errori ed export opzionale.
4. [Verifica](verifica.md) collega invarianti e comportamenti alla suite di test
   e distingue le garanzie locali dalle misure ancora esterne.

## Documenti specifici Raspberry Pi

I due rapporti affiancano i documenti tecnici e conservano informazioni legate
alla piattaforma e alla storia dell'adattamento:

- [Modifiche per Raspberry Pi](modifiche-raspberry-pi.md): interventi,
  dipendenze e risultati osservati.
- [Limitazioni Raspberry Pi](limitazioni-raspberry-pi.md): compatibilita,
  risorse, prestazioni e funzioni non supportate.

Il [README principale](../README.md) resta la guida rapida. In caso di
divergenza, il codice e i test descrivono il contratto eseguibile corrente;
questa documentazione deve essere aggiornata insieme alle modifiche che
interessano quei contratti.
