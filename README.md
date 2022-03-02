# The Herbarium 2022 - FGVC9
## Identifier les espèces de plantes des Amériques à partir de spécimens d'herbiers.
"The Herbarium 2022: Flora of North America" fait partie d'un projet du " New York Botanical Garden" financé par la "National Science Foundation" pour créer des outils permettant d'identifier de nouvelles espèces végétales dans le monde.
Le jeu de données se trouve ici : https://www.kaggle.com/c/herbarium-2022-fgvc9/data. Celui-ci est constitué de 1,05 million d'images de 15 500 plantes vasculaires, qui constituent plus de 90 % des taxons documentés en Amérique du Nord.Les ensembles d'entraînement et de test contiennent des images de spécimens d'herbiers de 15 501 espèces de plantes vasculaires. Chaque image contient exactement un spécimen. L'ensemble d'entraînement comporte un nombre d'eexemples d'images représentant des espèces plafonné à 80 images maximum. 
L'objectif est de **catégoriser automatiquement les 15 501 espèces de plantes vasculaires.**

 ## Evaluation :
 Les soumissions au concours sont évaluées à l'aide du score macro F1.
 Le score F1 est calculé comme suit : 
 
![alt text](https://render.githubusercontent.com/render/math?math=F_1%20=%202%20*%20\frac{precision%20*%20recall}{precision%20%2B%20recall})

Où 

![alt text](https://render.githubusercontent.com/render/math?math=precision%20=%20\frac{TP}{TP%20%2B%20FP})

![alt text](https://render.githubusercontent.com/render/math?math=recall%20=%20\frac{TP}{TP%20%2B%20FN})
