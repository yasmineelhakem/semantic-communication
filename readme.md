# 3. Conception et implémentation du système de communication sémantique des images

## 3.1. Présentation générale

Dans cette partie du projet, nous avons développé un système de communication sémantique basé sur le Deep Learning pour la transmission d'images à travers un canal de communication bruité.

Contrairement aux systèmes de transmission classiques qui cherchent à reconstruire les données transmises bit par bit, la communication sémantique vise à préserver les caractéristiques importantes de l'information transmise.

Le système proposé repose sur une architecture de type encodeur-canal-décodeur (Encoder-Channel-Decoder). Il comprend trois composants principaux :

- Un encodeur neuronal chargé d'extraire une représentation compacte de l'image.
- Un canal AWGN simulant les perturbations dues au bruit.
- Un décodeur neuronal chargé de reconstruire l'image à partir de la représentation reçue.

L'objectif est d'étudier la capacité du système à reconstruire des images de bonne qualité pour différents niveaux de bruit.

## 3.2. Architecture du système proposé

### 3.2.1. Encodeur neuronal

L'encodeur est un réseau de neurones convolutif (CNN) qui transforme l'image originale en une représentation latente de dimension réduite.

Il est constitué de trois couches convolutives :

- Première couche : convolution avec 32 filtres, suivie d'une fonction d'activation ReLU et d'un sous-échantillonnage avec un stride de 2.
- Deuxième couche : convolution avec 64 filtres, suivie d'une fonction ReLU et d'un sous-échantillonnage avec un stride de 2.
- Troisième couche : convolution avec 2 filtres, produisant la représentation destinée à la transmission.

L'image d'entrée possède trois canaux de couleur RGB et une résolution de 128 × 128 pixels. Après encodage, elle est représentée par un tenseur de dimension 2 × 32 × 32.

Cette transformation réduit considérablement la dimension de la représentation transmise.

### 3.2.2. Canal de communication AWGN

Afin de simuler les conditions d'un canal de communication réel, nous avons intégré un canal AWGN (Additive White Gaussian Noise).

Ce canal ajoute un bruit gaussien blanc additif au signal transmis.

Le signal reçu est donné par :

$$
y = x + n
$$

où :

- $x$ représente le signal transmis ;
- $n$ représente le bruit gaussien ;
- $y$ représente le signal reçu.

Le niveau de bruit est contrôlé par le rapport signal sur bruit (SNR), exprimé en décibels :

$$
\mathrm{SNR}_{\mathrm{dB}} = 10 \log_{10}\left(\frac{P_{\mathrm{signal}}}{P_{\mathrm{noise}}}\right)
$$

Dans notre implémentation, la représentation latente est normalisée afin que sa puissance moyenne complexe soit égale à 1 pour chaque échantillon.

Le bruit est ensuite généré indépendamment sur les deux composantes du signal (I et Q). Sa variance par composante réelle est définie par :

$$
\sigma^2 = \frac{1}{2 \times \mathrm{SNR}_{\mathrm{linear}}}
$$

avec :

$$
\mathrm{SNR}_{\mathrm{linear}} = 10^{\mathrm{SNR}_{\mathrm{dB}}/10}
$$

Cette approche permet de contrôler précisément les conditions de transmission et de tester la robustesse du système face à différents niveaux de bruit.

### 3.2.3. Décodeur neuronal

Le décodeur est également basé sur un réseau convolutif.

Il reçoit la représentation bruitée provenant du canal AWGN et tente de reconstruire l'image originale.

Son architecture comprend :

- Une couche convolutive transformant les deux canaux reçus en 64 caractéristiques.
- Une première couche de convolution transposée permettant d'augmenter la résolution.
- Une deuxième couche de convolution transposée permettant de retrouver les trois canaux RGB.
- Une fonction d'activation Sigmoid garantissant des valeurs de pixels comprises entre 0 et 1.

Le résultat final est une image reconstruite de dimension 3 × 128 × 128.

## 3.3. Fonctions de perte utilisées

Afin d'étudier l'influence de la fonction objectif sur la qualité des images reconstruites, deux stratégies d'entraînement ont été comparées.

### 3.3.1. Fonction de perte MSE

La première stratégie repose sur l'erreur quadratique moyenne (Mean Squared Error).

Elle mesure la différence pixel par pixel entre l'image originale et l'image reconstruite.

La fonction est définie par :

$$
\mathcal{L}_{\mathrm{MSE}} = \frac{1}{N} \sum_{i=1}^{N} (x_i - \hat{x}_i)^2
$$

où :

- x_i représente un pixel de l'image originale ;
- x̂_i représente le pixel correspondant de l'image reconstruite ;
- N représente le nombre total de valeurs de pixels.

Cette fonction de perte constitue une première référence pour entraîner le système de communication.

### 3.3.2. Fonction de perte perceptuelle

Dans la deuxième stratégie, nous avons ajouté une perte perceptuelle basée sur un réseau VGG16 pré-entraîné sur ImageNet.

Contrairement à la MSE, qui compare directement les pixels, la perte perceptuelle compare les caractéristiques extraites des images par le réseau VGG16.

Les couches utilisées sont gelées afin de conserver les caractéristiques apprises pendant le pré-entraînement.

La perte perceptuelle est calculée à partir de la distance L1 entre les représentations extraites de l'image originale et de l'image reconstruite :

$$
\mathcal{L}_{\mathrm{perceptual}} = \left\lVert \phi(x) - \phi(\hat{x}) \right\rVert_1
$$

où $\phi$ représente l'extracteur de caractéristiques VGG16.

La fonction de perte totale utilisée pendant le deuxième entraînement est :

$$
\mathcal{L}_{\mathrm{total}} = \mathcal{L}_{\mathrm{MSE}} + \lambda \times \mathcal{L}_{\mathrm{perceptual}}
$$

Le coefficient $\lambda$ a été fixé à 0,01.

Ce choix permet de combiner la précision de reconstruction pixel par pixel avec la préservation des caractéristiques visuelles.

Les deux stratégies sont comparées afin d'étudier l'apport de la perte perceptuelle dans le contexte de la communication sémantique.

## 3.4. Protocole expérimental

### 3.4.1. Jeu de données

Les expériences ont été réalisées sur le jeu de données CIFAR-10.

Ce jeu de données contient des images appartenant à dix classes différentes. Les images originales possèdent une résolution de 32 × 32 pixels.

Pour les besoins de notre architecture, les images ont été redimensionnées à 128 × 128 pixels et converties en tenseurs.

Un sous-ensemble de 5 000 images a été utilisé pour l'entraînement et 1 000 images pour l'évaluation.

### 3.4.2. Paramètres d'entraînement

Les paramètres utilisés sont les suivants :

| Paramètre | Valeur |
|---|---|
| Architecture | CNN Encoder-Decoder |
| Jeu de données | CIFAR-10 |
| Résolution d'entrée | 128 × 128 |
| Taille du batch | 32 |
| Nombre d'époques | 3 |
| Optimiseur | Adam |
| Taux d'apprentissage | 0,001 |
| Fonction de perte 1 | MSE |
| Fonction de perte 2 | MSE + 0,01 × Perceptual Loss |
| Canal | AWGN |
| SNR pendant l'entraînement | Tiré aléatoirement entre 0 et 20 dB |

Le SNR est choisi aléatoirement pendant l'entraînement afin d'exposer le modèle à différentes conditions de transmission.

### 3.4.3. Conditions d'évaluation

Les deux modèles entraînés ont été évalués sur les mêmes conditions de transmission.

Les valeurs de SNR considérées sont :

- -5 dB
- 0 dB
- 5 dB
- 10 dB
- 15 dB
- 20 dB

Cette évaluation permet d'observer le comportement du système lorsque le niveau de bruit varie.

## 3.5. Métriques d'évaluation

Trois métriques ont été utilisées pour comparer les performances des deux modèles.

### 3.5.1. Peak Signal-to-Noise Ratio (PSNR)

Le PSNR mesure la qualité de reconstruction en se basant sur l'erreur quadratique moyenne entre l'image originale et l'image reconstruite.

Une valeur élevée indique une meilleure qualité de reconstruction.

### 3.5.2. Structural Similarity Index (SSIM)

Le SSIM mesure la similarité structurelle entre deux images en tenant compte de leur luminance, de leur contraste et de leur structure.

Sa valeur est généralement comprise entre 0 et 1. Une valeur proche de 1 indique une forte similarité.

### 3.5.3. Perceptual Loss

La Perceptual Loss mesure la différence entre les caractéristiques extraites des images originales et reconstruites à l'aide de VGG16.

Une valeur faible indique une meilleure similarité dans l'espace des caractéristiques utilisé.

## 3.6. Résultats expérimentaux et discussion

Les résultats obtenus montrent que les performances des deux modèles s'améliorent lorsque le SNR augmente.

Cette évolution est attendue, car un SNR élevé correspond à un niveau de bruit plus faible dans le canal de communication.

### 3.6.1. Analyse du PSNR

Le modèle entraîné avec MSE + Perceptual Loss présente un PSNR supérieur à celui du modèle entraîné uniquement avec MSE pour toutes les valeurs de SNR évaluées.

À 20 dB, le modèle perceptuel atteint 27,68 dB, contre 26,30 dB pour le modèle MSE.

Ces résultats indiquent une amélioration de la qualité de reconstruction mesurée par le PSNR.

### 3.6.2. Analyse du SSIM

Les résultats du SSIM montrent également une amélioration en faveur du modèle intégrant la perte perceptuelle.

À 20 dB, le SSIM atteint environ 0,909 pour le modèle perceptuel, contre 0,870 pour le modèle MSE.

Cela suggère une meilleure préservation des structures visuelles des images reconstruites.

### 3.6.3. Analyse de la Perceptual Loss

La Perceptual Loss diminue lorsque le SNR augmente pour les deux modèles.

Le modèle entraîné avec la combinaison MSE + Perceptual Loss obtient des valeurs plus faibles sur l'ensemble des conditions évaluées.

À 20 dB, cette valeur est de 0,377 pour le modèle perceptuel, contre 0,513 pour le modèle MSE.

Cette observation confirme que l'ajout de la perte perceptuelle pendant l'entraînement améliore la similarité des caractéristiques extraites par VGG16.

### 3.6.4. Comparaison globale

Les résultats expérimentaux montrent que l'intégration d'une perte perceptuelle améliore les trois métriques considérées dans cette expérience.

L'amélioration est particulièrement visible lorsque les conditions de transmission sont moins bruitées.

Ces observations mettent en évidence l'intérêt de combiner une perte pixel par pixel et une perte basée sur les caractéristiques visuelles pour entraîner un système de communication sémantique.

## 3.7. Conclusion

Dans cette partie, nous avons conçu et évalué un système de communication sémantique d'images reposant sur une architecture CNN encodeur-décodeur et un canal AWGN.

Deux stratégies d'entraînement ont été étudiées : une première basée uniquement sur la MSE et une deuxième combinant la MSE avec une perte perceptuelle extraite d'un réseau VGG16 pré-entraîné.

Les résultats obtenus montrent que la deuxième stratégie fournit de meilleures performances selon les métriques PSNR, SSIM et Perceptual Loss pour les conditions de SNR étudiées.

Cette expérimentation constitue une première validation du système proposé. Des entraînements supplémentaires et des évaluations sur des jeux de données plus diversifiés permettraient de renforcer la généralisation et la robustesse des résultats.
