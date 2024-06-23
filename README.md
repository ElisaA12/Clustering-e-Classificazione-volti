# Clustering e Classificazione volti
Il progetto si compone di uno script jupyter principale "Clustering" il quale contiene l'elaborazione della traccia n. 8.
Inoltre contiene uno script jupyter aggiuntivo "Classificazione" per osservare le differenze con, appunto, la classificazione.

La cartella "keypoints" ha al suo interno il file "keypoints_loc" che verrà utilizzato per prendere i punti di ritaglio dell'immagine intorno al viso.

La cartella zippata "images" contiene le immagini da trattare.

Si hanno infine gli script python: 
1. Kmeans
2. Knn
3. eigen_training, per il calcolo degli autovalori e autovettori

Purtorppo non tutte le immagini sono state ritagliate in modo adeguato poichè, anche andando a selezionare i keypoints giusti, risultavano con il viso inclinato o comunque il soggetto era in una posa non ottima per l'elaborazione.
