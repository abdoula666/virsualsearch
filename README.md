# Visual Search Application

Une application de recherche visuelle qui permet aux utilisateurs de trouver des produits similaires en téléchargeant ou en prenant une photo. L'application utilise l'apprentissage profond pour trouver des produits visuellement similaires dans votre boutique WooCommerce.

## Fonctionnalités

- Interface mobile-first responsive
- Intégration de la caméra pour la capture directe de photos
- Support du téléchargement d'images
- Recherche visuelle en temps réel avec ResNet50
- Intégration avec l'API WooCommerce
- Affichage des noms et prix des produits
- Liens directs vers les pages produits
- Filtrage par catégorie
- Mise à jour automatique des produits
- Affichage de 30 résultats les plus pertinents
- Interface utilisateur moderne et réactive

## Technologies Utilisées

- Backend:
  - Flask (Serveur web)
  - TensorFlow/Keras (Deep Learning)
  - ResNet50 (Extraction de caractéristiques)
  - APScheduler (Tâches en arrière-plan)
  - NumPy (Calculs numériques)
  - PIL (Traitement d'images)

- Frontend:
  - HTML5/CSS3 (Interface utilisateur)
  - JavaScript (Interactivité)
  - Responsive Design
  - Font Awesome (Icônes)

## Installation

1. Clonez le dépôt:
```bash
git clone https://github.com/votre-username/visual-search.git
cd visual-search
```

2. Installez les dépendances:
```bash
pip install -r requirements.txt
```

3. Configurez les variables d'environnement:
Créez un fichier `.env` avec:
```
WOOCOMMERCE_URL=votre_url_woocommerce
CONSUMER_KEY=votre_cle_consommateur
CONSUMER_SECRET=votre_secret_consommateur
```

4. Lancez l'application:
```bash
python app.py
```

5. Ouvrez votre navigateur et accédez à `http://localhost:5000`

## Utilisation

1. Cliquez sur "Prendre une photo" pour utiliser la caméra
2. Ou cliquez sur "Choisir une image" pour télécharger une image
3. Utilisez le filtre de catégorie pour affiner les résultats
4. Cliquez sur un produit pour voir les détails sur la boutique

## Structure du Projet

```
visual-search/
├── app.py                 # Application Flask principale
├── requirements.txt       # Dépendances Python
├── templates/
│   └── index.html        # Interface utilisateur
├── static/
│   └── styles/           # Fichiers CSS
└── README.md             # Documentation
```

## Déploiement

1. Créez un compte sur GitHub si ce n'est pas déjà fait
2. Créez un nouveau dépôt
3. Initialisez Git et poussez le code:
```bash
git init
git add .
git commit -m "Initial commit"
git remote add origin https://github.com/votre-username/visual-search.git
git push -u origin main
```

## Contribution

Les contributions sont les bienvenues! N'hésitez pas à:
1. Fork le projet
2. Créer une branche (`git checkout -b feature/amelioration`)
3. Commit vos changements (`git commit -m 'Ajout de fonctionnalité'`)
4. Push à la branche (`git push origin feature/amelioration`)
5. Ouvrir une Pull Request

## Licence

Ce projet est sous licence MIT. Voir le fichier `LICENSE` pour plus de détails.

## Contact

Pour toute question ou suggestion, n'hésitez pas à ouvrir une issue sur GitHub.
