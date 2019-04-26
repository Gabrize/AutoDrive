# AutoDrive

Algorithme génétique et réseau de neurones - Projet AutoDrive
=================================================

Présentation
---------------

Le but va être d'apprendre à de petites voitures à se déplacer de manière autonome sur un circuit.
<img src="https://amp.businessinsider.com/images/57768fc1dd0895585b8b4d01-1920-1439.png" width="300" alt="Tesla">

Comme ça en fait : 

<img src="https://drive.google.com/uc?id=1ums42Jb9u-cz9fxdZjznur8HnkJfmU3u" width="300" alt="Original">

Structure des voitures
---------------
Chaque voiture est modélisée par un rectangle de taille fixée (réglable dans la classe Car de Components). Elle se dirige en prenant en compte sa distance aux murs dans les directions définies par l'attribut sensor_angles. 


<img src="https://drive.google.com/uc?id=14w72mTyjQM4kzG4PbpxYrDyTaA5BhzEN" width="300" alt="Sensors">

La fonction que l'on va apprendre est précisément celle qui prend en entrée ces distances et qui renvoie une variation angulaire et une vitesse (voir la fonction move de Car pour plus de précision sur la physique), cette fonction est approximée avec un réseau de neurone maison défini dans la classe NeuralNetwork, chaque voiture dispose de son propre réseau de neurones (même structure mais poids éventuellement différents).

Structure du circuit
---------------
La principale fonction du circuit est de permettre le calcul des distances aux murs dans les directions souhaitées et le calcul d'un score pour chaque voiture en vue de les classer.

<img src="https://drive.google.com/uc?id=1xUoccf_bwQFBRr7rUuQ3__JW5wjpYGUK" width="300" alt="Score">

Pour cela, on code le circuit sous la forme d'une image RGB avec les conventions suivantes :
* Les pixels de l'intérieur du circuit (avec lesquels il n'y a pas de collisions et qui sont "transparents" pour les capteurs) sont en blanc (255, 255, 255)
* Les pixels qui sont "transparents" pour les capteurs mais avec lesquels il y a collision sont en gris (ie les trois composantes RGB sont égales comme (120, 120, 120)). Ces pixels sont utiles lorsqu'on est sur un circuit en boucle pour calculer le score de manière réaliste ou alors pour forcer une voiture à s'arrêter sur la ligne d'arrivée et éviter qu'elle ait à se prendre un mur en connaissance de cause pour finir le circuit.
* Les pixels qui sont en jeu lors des collisions ont toutes les autres couleurs permises.

L'algorithme a aussi besoin de la donnée des coordonnées d'un point de départ pour utiliser une nouvelle carte.

Structure de l'algorithme
---------------
L'entrainement des réseaux de neurones des voitures est un peu original par rapport à ce qu'on a fait pendant les formations. Il s'agit d'un apprentissage non-supervisé qui utilise un concept d'algorithme génétique, c'est à dire que beaucoup de solutions sont testées puis les meilleures sont sélectionnées, de nouvelles solution sont construites à partir des meilleures. Puis on recommence, le processus se rapproche de l'évolution de Darwin par beaucoup d'aspects. 

Mais alors quel est l'ADN d'une voiture ? En fait, ce sont les poids de son réseau de neurones qui sont placés dans une longue liste.

<img src="https://drive.google.com/uc?id=1pjAcGKTX91a_uPPy4wkSNz_frjeNbIbp" width="400" alt="Etapes">




Un petit peu plus proprement, l'algorithme présente 4 étapes principales :
* L'évaluation : c'est la simulation physique des voitures dirigées par leur réseau de neurones, à chaque voiture est affecté un score qui reflète son succès sur le circuit. Pour calculer ce score dans notre cas, on utilise la distance au point de départ. On peut bien sûr rajouter un bonus lorsque la voiture fini le circuit, lorsqu'elle le finit rapidement, on pourrait même récompenser sa faible consommation de carburant !

* La sélection : il s'agit de sélectionner les voitures qui serviront de base à la population suivante, elles sont sélectionnées avec une probabilité proportionnelle à une fonction croissante de leur score sur le circuit. Dans la version actuelle, c'est une fonction puissance qui est employée. L'exposant est réglable via la valeur de elitism. Une valeur nulle entrainera une sélection aléatoire et uniforme, sans prise en compte du score, une valeur de 1 entraînera une selection avec une probabilité directement proportionnelle au score (un score deux fois plus élevé donne une probabilité deux fois plus élevée), des valeurs supérieures favorisent très fortement les meilleurs individus. Par exemple, pour un elitism de 2,  un score deux fois plus élevé donne une probabilité quatre fois plus élevée

* Le croisement : pour construire un individu de la nouvelle génération, on choisit deux parents selon notre algorithme de sélection puis on construit deux enfants à partir d'eux en choisissant aléatoirement certains poids de l'un et en complétant avec les poids de l'autre. Cette étape est désactivée par défaut dans le programme avec la valeur par défaut crossing_rate = 0. En effet, dans notre cas précis, l'intérêt du croisement n'est pas du tout parce que deux neurones au même emplacement dans deux réseaux différents peuvent avoir des fonctions qui n'ont aucun rapport. Donc remplacer un neurone d'un individu par le neurone de même emplacement d'un autre individu peut se révéler contre-productif.

* La mutation : pour permettre à l'algorithme de découvrir de nouvelles solutions, on introduit une dernière étape encore inspirée de la théorie de l'évolution. Il s'agit d'ajouter aux poids des nouveaux individus un bruit aléatoire en vue de faire émerger de nouveau comportements. La matrice de bruit est paramètrée par mutating_probability (la proportion de valeurs non-nulles dans la matrice de bruit) et noise_amplitude (le bruit prend ses valeurs dans un intervalle de cette longueur centré en 0).

En plus de ces étapes, on ajoute un mécanisme de préservation des meilleurs individus. C'est à dire qu'un certain nombre d'individus seront directement copiés dans la génération suivante sans subir ni croisement, ni mutation. La taille de cette population copiée est réglable via elite_size.


Lancer un apprentissage
-----------------------------

Pour lancer un apprentissage, il suffit d'appeler la fonction main de Main avec les paramètres souhaités. Pour lancer un apprentissage sur un nouveau circuit, il faut en plus modifier les lignes 

path = "circuit_rond_trou.png"

start = (74, 300)

Petites précisions sur l'affichage
-----------------------------

<img src="https://drive.google.com/uc?id=1mBGOemDd6N3kKrxImKW-P25RLnF_WZ6z" width="400" alt="Etapes">


Il y a alors une fenêtre toute jolie qui s'affiche (si Pygame).
À partir de la deuxième génération, les voitures ont des couleurs oranges (au hasard). Ces couleurs correspondent à la performance des parents de la voiture en question (plus c'est jaune, meilleurs sont les parents).
En bas à gauche, tu vois un petit réseau de neurones, c'est celui de la meilleure voiture sur le circuit. L'épaisseur des liens est proportionnelle à la valeur des poids et la couleur indique son signe : vert pour les positifs et rouge pour les négatifs.



Créer un nouveau circuit
-----------------------------

<img src="https://drive.google.com/uc?id=1zmfVduTSC2470g6uQj2rKBMy5RK1WH2-" width="400" alt="Edition">

Les circuits sont de simples images qui doivent respecter les conventions ci-dessus. Il y a quelques précautions à prendre pour éviter des erreurs embarrassantes :
* Bien enregistrer les images sans canal alpha (transparence) si elles sont faites avec un logiciel de dessin
* Pour les circuits, le couper avec une ligne d'arrivée grise sinon le calcul des distances n'aura aucun sens : l'arrivée serait l'un des points les plus proches du départ !
* Il est possible qu'il faille changer manuellement l'angle initial des véhicules sous peine de les voir dans l'impossibilité pratique d'avancer dans le circuit.


Adapter la fonction de score
-----------------------------
Il y a plein de moyens rendre plus pertinente la fonction de score, comme donner un bonus si la voiture arrive vite. C'est à toi de jouer !
La fonction qui donne les scores est localisée dans la classe Map du module Components. Pour l'instant, elle n'affecte que la distance, on peut y rajouter beaucoup d'autres choses.
