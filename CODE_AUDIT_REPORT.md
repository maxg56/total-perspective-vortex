# Code Audit Report - Total Perspective Vortex

| **Champ** | **Valeur** |
|-----------|-----------|
| **Date** | 2025-12-11 |
| **Auditeur** | Claude Code (Opus 4) |
| **Dépôt** | total-perspective-vortex |
| **Technologie** | Python 3.x - EEG/BCI avec MNE, scikit-learn |
| **Fichiers Python** | 43 fichiers |
| **LOC estimées** | ~5,102 lignes |

---

## 1. Executive Summary

Le projet **Total Perspective Vortex** est une application BCI (Brain-Computer Interface) pour la classification de signaux EEG basée sur l'imagerie motrice. Le code présente une **architecture bien structurée** avec une séparation claire des responsabilités (preprocessing, training, prediction, visualization). La qualité du code est **globalement bonne** avec utilisation de type hints et docstrings. Cependant, des **vulnérabilités de sécurité** ont été identifiées, notamment l'absence de validation des fichiers modèle chargés via `joblib.load()`. La gestion des erreurs est **partielle** avec un mélange de `print()` et `logging`. La couverture de tests est **satisfaisante** avec 15 fichiers de tests mais manque certains cas critiques.

### Notes Globales

| Critère | Score | Commentaire |
|---------|-------|-------------|
| **Code Quality** | 7.5/10 | Bonne structure, type hints, mais refactoring possible |
| **Security** | 5/10 | Absence de validation SHA256 pour les modèles, pas de path traversal protection |
| **Error Handling** | 6/10 | Mélange print/logging, certains except trop larges |
| **Testing** | 7/10 | Bonne couverture de base, manque tests edge cases |
| **Documentation** | 7.5/10 | Docstrings présentes, README correct |

**Niveau de risque global : MEDIUM**

---

## 2. Architecture & Structure

### Arborescence Détectée

```
total-perspective-vortex/
├── src/
│   ├── __init__.py
│   ├── constants.py          # Constantes centralisées
│   ├── preprocess.py         # Chargement et prétraitement EEG
│   ├── features.py           # Extracteurs de features (PSD, BandPower)
│   ├── pipeline.py           # Factory de pipelines sklearn
│   ├── mybci.py              # Point d'entrée CLI principal
│   ├── mycsp.py              # Re-export CSP/PCA (backward compat)
│   ├── predict.py            # Prédiction et simulation temps réel
│   ├── train.py              # Re-export training (backward compat)
│   ├── training/
│   │   ├── __init__.py
│   │   ├── core.py           # train_and_evaluate, train_with_holdout
│   │   ├── comparison.py     # compare_pipelines
│   │   ├── persistence.py    # save_model, load_model
│   │   └── subject.py        # train_subject
│   ├── transforms/
│   │   ├── __init__.py
│   │   ├── csp.py            # Implémentation CSP custom
│   │   └── pca.py            # Implémentation PCA custom
│   └── visualization/
│       ├── __init__.py
│       ├── _base.py          # Utilitaires de plot
│       ├── cv_plots.py       # Plots cross-validation
│       ├── comparison_plots.py
│       └── metrics_plots.py
├── tests/
│   ├── conftest.py           # Fixtures pytest
│   ├── test_*.py             # Tests unitaires
│   ├── features/             # Tests extracteurs
│   ├── pipeline/             # Tests pipelines
│   └── transforms/           # Tests CSP/PCA
├── models/                   # Modèles sauvegardés (.pkl)
├── plots/                    # Graphiques générés
├── requirements.txt
├── requirements-lock.txt
└── setup.cfg
```

### Points Forts d'Architecture

1. **Séparation des responsabilités** : Modules distincts pour preprocessing, training, prediction
2. **Pattern Factory** : `get_pipeline()` pour instancier les pipelines ML
3. **Constantes centralisées** : `constants.py` évite les magic numbers
4. **Backward compatibility** : Re-exports dans `train.py` et `mycsp.py`
5. **Fixtures pytest** : Données synthétiques pour tests sans téléchargement Physionet

---

## 3. Code Quality Assessment

### Synthèse des Issues

| Sévérité | Nombre | Description |
|----------|--------|-------------|
| **Critical** | 1 | Chargement modèle non sécurisé (joblib.load sans validation) |
| **High** | 2 | Path traversal possible, except Exception trop large |
| **Medium** | 4 | Mélange print/logging, pas de rotation logs, magic strings |
| **Low** | 6 | Code duplication, imports non optimaux, type hints manquants |

### Issue Critical #1 : Chargement de modèle non sécurisé

**Fichier** : `src/training/persistence.py:63`

**Description** : La fonction `load_model()` utilise `joblib.load()` sans aucune validation d'intégrité. Un fichier `.pkl` malveillant peut exécuter du code arbitraire lors du chargement.

**Reproduction** :
```python
# Un attaquant pourrait créer un fichier malveillant
# qui s'exécute lors de joblib.load()
```

**Preuve/Explication** :
```python
# Code actuel - VULNÉRABLE
def load_model(path: str) -> Tuple[Pipeline, Dict[str, Any]]:
    model_data = joblib.load(path)  # <-- Exécution de code arbitraire possible
    return model_data['pipeline'], model_data['metadata']
```

**Correctif proposé** : Voir Section 5 - Security Audit Findings

---

### Issue High #1 : Path Traversal Possible

**Fichier** : `src/training/persistence.py:36-40`

**Description** : La fonction `save_model()` accepte n'importe quel chemin sans validation, permettant potentiellement l'écriture dans des répertoires non autorisés.

**Code problématique** :
```python
def save_model(pipeline: Pipeline, path: str, ...) -> None:
    dir_path = os.path.dirname(path) or '.'
    os.makedirs(dir_path, exist_ok=True)  # Pas de validation du chemin
    joblib.dump(model_data, path)
```

---

### Issue High #2 : Except Exception trop large

**Fichier** : `src/preprocess.py:268-270`

**Description** : L'exception `except Exception` capture toutes les erreurs y compris `KeyboardInterrupt`, `SystemExit`, masquant potentiellement des bugs.

**Code problématique** :
```python
except Exception as e:
    print(f"Error loading subject {subject}: {e}")
    continue  # Continue silencieusement
```

---

### Issue Medium #1 : Mélange print/logging

**Statistiques** :
- 15 fichiers utilisent `print()`
- 6 fichiers utilisent `logging`
- Incohérence dans la stratégie de logging

---

### Recommandations de Refactoring

1. **Unifier le logging** : Remplacer tous les `print()` par `logging`
2. **Ajouter validation des entrées** : Valider paths, subject IDs, run numbers
3. **Extraire les constantes restantes** : Quelques magic strings restent (ex: format de fichier modèle)
4. **Type hints complets** : Ajouter les types de retour manquants dans certaines méthodes
5. **Réduire la duplication** : Les séparateurs `'=' * 60` sont répétés

---

## 4. Security Audit Findings

### Issues de Sécurité Classées

| ID | Sévérité | Description | Fichier |
|----|----------|-------------|---------|
| SEC-01 | **Critical** | Chargement modèle sans validation intégrité | persistence.py:63 |
| SEC-02 | **High** | Path traversal dans save_model | persistence.py:36 |
| SEC-03 | **High** | Path traversal dans save plot | _base.py:41 |
| SEC-04 | **Medium** | Pas de validation du format fichier | predict.py:273 |
| SEC-05 | **Low** | Secrets potentiels non protégés si ajoutés | N/A |

---

### Correctif SEC-01 : load_model sécurisé avec SHA256

```python
"""
Corrige le chargement non sécurisé en ajoutant une validation SHA256.
Le hash doit être stocké lors de la sauvegarde et vérifié au chargement.
"""
import hashlib
import os
import logging
from typing import Tuple, Dict, Optional, Any
import joblib
from sklearn.pipeline import Pipeline

logger = logging.getLogger(__name__)

HASH_SUFFIX = ".sha256"


def compute_file_hash(path: str) -> str:
    """Calcule le hash SHA256 d'un fichier."""
    sha256 = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(8192), b''):
            sha256.update(chunk)
    return sha256.hexdigest()


def save_model_secure(
    pipeline: Pipeline,
    path: str,
    metadata: Optional[Dict[str, Any]] = None
) -> str:
    """
    Sauvegarde un modèle avec hash d'intégrité SHA256.

    Returns
    -------
    hash_value : str
        Le hash SHA256 du fichier sauvegardé
    """
    # Validation du chemin (voir correctif SEC-02)
    path = validate_and_resolve_path(path, allowed_dirs=['models'])

    model_data = {'pipeline': pipeline, 'metadata': metadata or {}}

    dir_path = os.path.dirname(path) or '.'
    os.makedirs(dir_path, exist_ok=True)

    joblib.dump(model_data, path)

    # Calculer et sauvegarder le hash
    file_hash = compute_file_hash(path)
    hash_path = path + HASH_SUFFIX
    with open(hash_path, 'w') as f:
        f.write(file_hash)

    logger.info(f"Model saved to: {path} (SHA256: {file_hash[:16]}...)")
    return file_hash


def load_model_secure(
    path: str,
    expected_hash: Optional[str] = None
) -> Tuple[Pipeline, Dict[str, Any]]:
    """
    Charge un modèle avec vérification d'intégrité SHA256.

    Parameters
    ----------
    path : str
        Chemin vers le fichier modèle
    expected_hash : str, optional
        Hash SHA256 attendu. Si None, lit depuis le fichier .sha256

    Raises
    ------
    FileNotFoundError
        Si le fichier modèle n'existe pas
    ValueError
        Si le hash ne correspond pas (fichier corrompu/modifié)
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model not found: {path}")

    # Récupérer le hash attendu
    if expected_hash is None:
        hash_path = path + HASH_SUFFIX
        if os.path.exists(hash_path):
            with open(hash_path, 'r') as f:
                expected_hash = f.read().strip()
        else:
            logger.warning(f"No hash file found for {path}, loading without verification")

    # Vérifier l'intégrité
    if expected_hash:
        actual_hash = compute_file_hash(path)
        if actual_hash != expected_hash:
            raise ValueError(
                f"Model integrity check failed!\n"
                f"Expected: {expected_hash}\n"
                f"Got: {actual_hash}\n"
                f"The file may have been corrupted or tampered with."
            )
        logger.info(f"Model integrity verified: {path}")

    model_data = joblib.load(path)
    return model_data['pipeline'], model_data['metadata']
```

---

### Correctif SEC-02 : Validation de chemin (Path Traversal)

```python
"""
Valide les chemins pour empêcher les attaques path traversal.
"""
import os
from pathlib import Path
from typing import List, Optional

ALLOWED_MODEL_DIRS = ['models', 'tmp', '/tmp']
ALLOWED_PLOT_DIRS = ['plots', 'tmp', '/tmp']


def validate_and_resolve_path(
    path: str,
    allowed_dirs: Optional[List[str]] = None,
    base_dir: Optional[str] = None
) -> str:
    """
    Valide et résout un chemin de fichier de manière sécurisée.

    Parameters
    ----------
    path : str
        Chemin à valider
    allowed_dirs : list, optional
        Liste des répertoires autorisés
    base_dir : str, optional
        Répertoire de base (défaut: cwd)

    Returns
    -------
    resolved_path : str
        Chemin absolu résolu et validé

    Raises
    ------
    ValueError
        Si le chemin contient des séquences dangereuses
        ou sort des répertoires autorisés
    """
    if base_dir is None:
        base_dir = os.getcwd()

    # Détecter les tentatives de path traversal
    if '..' in path or path.startswith('/') and allowed_dirs:
        # Vérifier si le chemin absolu est dans les répertoires autorisés
        abs_path = os.path.abspath(path)
        allowed = False
        for allowed_dir in allowed_dirs:
            allowed_abs = os.path.abspath(allowed_dir)
            if abs_path.startswith(allowed_abs):
                allowed = True
                break
        if not allowed:
            raise ValueError(
                f"Path '{path}' is outside allowed directories: {allowed_dirs}"
            )

    # Résoudre le chemin
    resolved = os.path.normpath(os.path.join(base_dir, path))

    # Vérifier que le chemin résolu reste dans base_dir ou allowed_dirs
    if allowed_dirs:
        in_allowed = False
        for allowed_dir in allowed_dirs:
            allowed_abs = os.path.abspath(os.path.join(base_dir, allowed_dir))
            if resolved.startswith(allowed_abs):
                in_allowed = True
                break
        if not in_allowed and not resolved.startswith(os.path.abspath(base_dir)):
            raise ValueError(f"Resolved path '{resolved}' is outside allowed scope")

    return resolved
```

---

### Correctif SEC-03 : Sauvegarde de plots sécurisée

```python
"""
Sauvegarde sécurisée des plots avec validation du chemin.
"""
import os
import matplotlib.pyplot as plt
from typing import Optional

from constants import PLOT_DPI

ALLOWED_PLOT_EXTENSIONS = {'.png', '.pdf', '.svg', '.jpg', '.jpeg'}


def save_plot_secure(
    fig,
    save_path: str,
    allowed_dirs: Optional[list] = None
) -> str:
    """
    Sauvegarde un plot de manière sécurisée.

    Raises
    ------
    ValueError
        Si l'extension n'est pas autorisée ou chemin invalide
    """
    if allowed_dirs is None:
        allowed_dirs = ['plots', 'tmp']

    # Valider l'extension
    ext = os.path.splitext(save_path)[1].lower()
    if ext not in ALLOWED_PLOT_EXTENSIONS:
        raise ValueError(
            f"Extension '{ext}' not allowed. "
            f"Allowed: {ALLOWED_PLOT_EXTENSIONS}"
        )

    # Valider le chemin
    safe_path = validate_and_resolve_path(save_path, allowed_dirs=allowed_dirs)

    # Créer le répertoire
    save_dir = os.path.dirname(safe_path)
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    # Sauvegarder
    fig.savefig(safe_path, dpi=PLOT_DPI, bbox_inches='tight')

    return safe_path
```

---

## 5. Error Handling & Logging

### État Actuel

| Métrique | Valeur |
|----------|--------|
| Fichiers utilisant `print()` | 15 |
| Fichiers utilisant `logging` | 6 |
| Handlers `except Exception` larges | 2 |
| Rotation des logs | Non configurée |

### Problèmes Identifiés

1. **Incohérence** : Mix de `print()` et `logging` dans le même module
2. **Pas de niveaux de log** : Absence de DEBUG/INFO/WARNING différenciés
3. **Pas de rotation** : Logs peuvent grandir indéfiniment
4. **Pas de flag --debug** : Impossible d'activer le debug à l'exécution

---

### Snippet : setup_logging() centralisé avec rotation

```python
"""
Configuration centralisée du logging avec rotation.
À placer dans src/logging_config.py
"""
import logging
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Optional


def setup_logging(
    level: int = logging.INFO,
    log_file: Optional[str] = None,
    max_bytes: int = 10 * 1024 * 1024,  # 10 MB
    backup_count: int = 5,
    debug: bool = False
) -> logging.Logger:
    """
    Configure le logging centralisé avec rotation optionnelle.

    Parameters
    ----------
    level : int
        Niveau de log par défaut (INFO)
    log_file : str, optional
        Chemin vers le fichier de log
    max_bytes : int
        Taille max du fichier avant rotation (10 MB)
    backup_count : int
        Nombre de fichiers de backup à garder
    debug : bool
        Si True, force le niveau DEBUG

    Returns
    -------
    logger : logging.Logger
        Logger racine configuré
    """
    if debug:
        level = logging.DEBUG

    # Format
    fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    date_fmt = '%Y-%m-%d %H:%M:%S'
    formatter = logging.Formatter(fmt, datefmt=date_fmt)

    # Logger racine
    root_logger = logging.getLogger()
    root_logger.setLevel(level)

    # Supprimer les handlers existants
    root_logger.handlers.clear()

    # Handler console
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)

    # Handler fichier avec rotation
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        file_handler = RotatingFileHandler(
            log_file,
            maxBytes=max_bytes,
            backupCount=backup_count,
            encoding='utf-8'
        )
        file_handler.setLevel(logging.DEBUG)  # Fichier capture tout
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)

    # Réduire le bruit des loggers externes
    logging.getLogger('matplotlib').setLevel(logging.WARNING)
    logging.getLogger('mne').setLevel(logging.WARNING)

    return root_logger


# Usage dans mybci.py:
# from logging_config import setup_logging
# setup_logging(debug=args.debug, log_file='logs/bci.log')
```

---

### Snippet : Remplacement de except Exception

```python
"""
Remplace le pattern 'except Exception' trop large par une gestion ciblée.
Fichier: src/preprocess.py lignes 263-270
"""
# AVANT - Problématique
# except Exception as e:
#     print(f"Error loading subject {subject}: {e}")
#     continue

# APRÈS - Amélioré
import logging
from mne.io import RawEdf  # Pour les erreurs MNE spécifiques

logger = logging.getLogger(__name__)


def load_multiple_subjects_safe(subjects, runs, **kwargs):
    """Version améliorée avec gestion d'erreurs ciblée."""
    X_all, y_all = [], []
    failed_subjects = []

    for subject in subjects:
        try:
            X, y, _ = preprocess_subject(subject, runs, **kwargs)
            X_all.append(X)
            y_all.append(y)
            logger.info(f"Subject {subject}: {len(y)} epochs loaded")

        except FileNotFoundError as e:
            logger.warning(f"Subject {subject}: Data file not found - {e}")
            failed_subjects.append((subject, 'file_not_found'))

        except ValueError as e:
            logger.warning(f"Subject {subject}: Invalid data - {e}")
            failed_subjects.append((subject, 'invalid_data'))

        except OSError as e:
            logger.error(f"Subject {subject}: I/O error - {e}")
            failed_subjects.append((subject, 'io_error'))

        except MemoryError:
            logger.critical(f"Subject {subject}: Out of memory!")
            raise  # Re-raise les erreurs critiques

    if failed_subjects:
        logger.warning(f"Failed to load {len(failed_subjects)} subjects: {failed_subjects}")

    if not X_all:
        raise RuntimeError("No subjects could be loaded successfully")

    return np.concatenate(X_all), np.concatenate(y_all)
```

---

### Plan pour ajouter --debug flag

```python
"""
Modification de mybci.py pour ajouter le flag --debug
"""
# Dans parse_args(), ajouter:
parser.add_argument(
    '--debug', '-d',
    action='store_true',
    help='Enable debug logging (verbose output)'
)

# Dans main(), avant les autres opérations:
def main() -> int:
    args = parse_args()

    # Configurer le logging AVANT tout
    from logging_config import setup_logging
    log_file = 'logs/bci.log' if not args.quiet else None
    setup_logging(debug=args.debug, log_file=log_file)

    # ... reste du code
```

---

## 6. Testing Coverage Analysis

### Synthèse de la Couverture

| Métrique | Valeur |
|----------|--------|
| Fichiers de test | 15 |
| Modules couverts | ~70% |
| Cas edge couverts | ~50% |

### Tests Manquants Critiques

| Priorité | Test Manquant | Module |
|----------|---------------|--------|
| **High** | Test load_model avec fichier corrompu | persistence.py |
| **High** | Test path traversal | persistence.py, _base.py |
| **High** | Test avec données NaN/Inf | preprocess.py, features.py |
| **Medium** | Test timeout prédiction | predict.py |
| **Medium** | Test robustesse CSP (matrice singulière) | csp.py |
| **Low** | Test comparaison pipelines avec failures | comparison.py |

---

### Exemples de Tests à Ajouter

#### Test de Persistence/Intégrité

```python
"""
Tests pour la persistence sécurisée des modèles.
Fichier: tests/test_persistence_security.py
"""
import pytest
import os
import tempfile
import hashlib

from training.persistence import (
    save_model_secure,
    load_model_secure,
    compute_file_hash
)


class TestSecureModelPersistence:
    """Tests pour la sauvegarde/chargement sécurisé."""

    def test_save_creates_hash_file(self, trained_csp_pipeline, tmp_path):
        """Vérifie que save_model_secure crée un fichier .sha256."""
        pipeline, X, y = trained_csp_pipeline
        model_path = str(tmp_path / "models" / "test.pkl")

        save_model_secure(pipeline, model_path)

        assert os.path.exists(model_path)
        assert os.path.exists(model_path + ".sha256")

    def test_load_verifies_hash(self, trained_csp_pipeline, tmp_path):
        """Vérifie que load_model_secure valide le hash."""
        pipeline, X, y = trained_csp_pipeline
        model_path = str(tmp_path / "models" / "test.pkl")

        expected_hash = save_model_secure(pipeline, model_path)
        loaded_pipeline, metadata = load_model_secure(model_path, expected_hash)

        # Doit fonctionner sans erreur
        predictions = loaded_pipeline.predict(X)
        assert len(predictions) == len(y)

    def test_load_fails_on_corrupted_file(self, trained_csp_pipeline, tmp_path):
        """Vérifie que load échoue si le fichier est modifié."""
        pipeline, X, y = trained_csp_pipeline
        model_path = str(tmp_path / "models" / "test.pkl")

        save_model_secure(pipeline, model_path)

        # Corrompre le fichier
        with open(model_path, 'ab') as f:
            f.write(b'CORRUPTED')

        with pytest.raises(ValueError, match="integrity check failed"):
            load_model_secure(model_path)

    def test_path_traversal_blocked(self, trained_csp_pipeline, tmp_path):
        """Vérifie que les path traversal sont bloqués."""
        pipeline, X, y = trained_csp_pipeline

        # Tentative de path traversal
        malicious_path = "../../../etc/passwd"

        with pytest.raises(ValueError, match="outside allowed"):
            save_model_secure(pipeline, malicious_path)


class TestModelLoadErrors:
    """Tests de gestion d'erreurs au chargement."""

    def test_load_nonexistent_raises_error(self, tmp_path):
        """Vérifie l'erreur pour fichier inexistant."""
        with pytest.raises(FileNotFoundError):
            load_model_secure(str(tmp_path / "nonexistent.pkl"))

    def test_load_without_hash_warns(self, trained_csp_pipeline, tmp_path, caplog):
        """Vérifie le warning si pas de fichier hash."""
        import logging

        pipeline, X, y = trained_csp_pipeline
        model_path = str(tmp_path / "test.pkl")

        # Sauvegarder sans hash (ancien format)
        import joblib
        joblib.dump({'pipeline': pipeline, 'metadata': {}}, model_path)

        with caplog.at_level(logging.WARNING):
            load_model_secure(model_path)

        assert "No hash file found" in caplog.text
```

---

#### Test de Training avec Edge Cases

```python
"""
Tests pour les cas limites du training.
Fichier: tests/test_training_edge_cases.py
"""
import pytest
import numpy as np


class TestTrainingEdgeCases:
    """Tests pour les cas limites du training."""

    def test_training_with_nan_values(self, small_synthetic_data):
        """Vérifie la gestion des NaN dans les données."""
        from train import train_and_evaluate

        X, y = small_synthetic_data
        X_nan = X.copy()
        X_nan[0, 0, 0] = np.nan

        # Devrait lever une erreur ou gérer proprement
        with pytest.raises((ValueError, RuntimeError)):
            train_and_evaluate(X_nan, y, cv=2, verbose=False)

    def test_training_with_inf_values(self, small_synthetic_data):
        """Vérifie la gestion des Inf dans les données."""
        from train import train_and_evaluate

        X, y = small_synthetic_data
        X_inf = X.copy()
        X_inf[0, 0, 0] = np.inf

        with pytest.raises((ValueError, RuntimeError)):
            train_and_evaluate(X_inf, y, cv=2, verbose=False)

    def test_training_with_single_sample_per_class(self):
        """Vérifie le comportement avec très peu de données."""
        from train import train_and_evaluate

        # 2 échantillons seulement
        X = np.random.randn(2, 16, 160)
        y = np.array([1, 2])

        # CV impossible avec si peu de données
        with pytest.raises(ValueError):
            train_and_evaluate(X, y, cv=2, verbose=False)

    def test_csp_with_singular_covariance(self):
        """Vérifie la robustesse CSP avec covariance singulière."""
        from transforms.csp import MyCSP

        # Données avec colonnes identiques (covariance singulière)
        n_epochs, n_channels, n_times = 20, 10, 100
        X = np.random.randn(n_epochs, n_channels, n_times)
        X[:, 1, :] = X[:, 0, :]  # Colonne dupliquée
        y = np.array([1] * 10 + [2] * 10)

        csp = MyCSP(n_components=4, reg=0.01)  # Régularisation devrait aider

        # Ne devrait pas échouer grâce à la régularisation
        X_csp = csp.fit_transform(X, y)
        assert X_csp.shape == (n_epochs, 4)
```

---

#### Test d'Error Cases

```python
"""
Tests pour les cas d'erreur.
Fichier: tests/test_error_handling.py
"""
import pytest
import numpy as np


class TestErrorHandling:
    """Tests de gestion d'erreurs."""

    def test_invalid_subject_number(self):
        """Vérifie l'erreur pour numéro de sujet invalide."""
        from preprocess import preprocess_subject

        with pytest.raises(Exception):  # FileNotFoundError ou ValueError
            preprocess_subject(subject=999, runs=[6])

    def test_invalid_run_number(self):
        """Vérifie l'erreur pour numéro de run invalide."""
        from preprocess import get_run_type

        with pytest.raises(ValueError, match="Invalid run number"):
            get_run_type(99)

    def test_mixed_run_types_rejected(self):
        """Vérifie le rejet des runs de types différents."""
        from preprocess import preprocess_subject

        # Run 6 = hands_feet, Run 4 = left_right
        with pytest.raises(ValueError, match="same type"):
            preprocess_subject(subject=1, runs=[4, 6])

    def test_csp_with_wrong_n_classes(self, synthetic_eeg_3class):
        """Vérifie l'erreur CSP avec plus de 2 classes."""
        from transforms.csp import MyCSP

        X, y = synthetic_eeg_3class
        csp = MyCSP(n_components=4)

        with pytest.raises(ValueError, match="exactly 2 classes"):
            csp.fit(X, y)

    def test_predict_before_fit(self):
        """Vérifie l'erreur si predict avant fit."""
        from transforms.csp import MyCSP

        X = np.random.randn(10, 64, 480)
        csp = MyCSP(n_components=4)

        with pytest.raises(RuntimeError, match="not fitted"):
            csp.transform(X)
```

---

## 7. Dependencies & Requirements

### Problème : Loose Constraints

Le fichier `requirements.txt` utilise des contraintes lâches (`>=`) qui peuvent causer des problèmes de reproductibilité.

**Situation actuelle** :
```
numpy>=1.21.0
scipy>=1.7.0
scikit-learn>=1.0.0
```

### Recommandations

1. **Utiliser le fichier lock** : `requirements-lock.txt` existe et devrait être utilisé en production
2. **Pinning des versions** : Utiliser `==` pour les dépendances critiques
3. **Séparer dev/prod** : Créer `requirements-dev.txt` pour les outils de dev

### Audit CVE - Commande

```bash
# Installation de pip-audit
pip install pip-audit

# Audit des vulnérabilités
pip-audit -r requirements-lock.txt

# Audit avec sortie JSON
pip-audit -r requirements-lock.txt -f json -o audit-report.json

# Mise à jour automatique si vulnérabilités trouvées
pip-audit -r requirements-lock.txt --fix
```

### Dépendances à surveiller

| Package | Version Lock | CVE Connues (2024) |
|---------|--------------|-------------------|
| pillow | 10.3.0 | Mettre à jour si CVE |
| numpy | 1.26.4 | OK |
| scipy | 1.12.0 | OK |
| tqdm | 4.66.3 | À vérifier |

---

## 8. Performance

### Patterns à Optimiser

#### 1. Boucles Python vs Vectorisation

**Fichier** : `src/features.py` - `PSDExtractor.transform()`

```python
# AVANT - Boucles imbriquées (lent)
for epoch_idx in range(n_epochs):
    for ch_idx in range(n_channels):
        freqs, psd = signal.welch(X[epoch_idx, ch_idx, :], ...)

# APRÈS - Vectorisation avec apply_along_axis
def _compute_psd_channel(data, fs, nperseg, noverlap, freq_bands):
    freqs, psd = signal.welch(data, fs=fs, nperseg=nperseg, noverlap=noverlap)
    return np.array([np.mean(psd[(freqs >= l) & (freqs <= h)])
                     for l, h in freq_bands.values()])

# Appliqué sur toutes les époques et canaux
features = np.apply_along_axis(
    _compute_psd_channel, 2, X,
    self.fs, self.nperseg, self.noverlap, self.freq_bands
)
```

#### 2. CSP - Déjà bien optimisé

Le fichier `csp.py` utilise déjà `np.einsum` pour le calcul de covariance vectorisé. C'est une bonne pratique.

```python
# Déjà optimisé - utilise einsum
covs = np.einsum('ijk,ilk->ijl', X, X) / (n_times - 1)
```

---

### Recommandations de Caching/Batching

```python
"""
Exemple de caching pour les données préprocessées.
"""
import hashlib
import os
from functools import lru_cache
import numpy as np


def get_cache_key(subject: int, runs: list, **params) -> str:
    """Génère une clé de cache unique."""
    key_str = f"{subject}_{sorted(runs)}_{sorted(params.items())}"
    return hashlib.md5(key_str.encode()).hexdigest()[:16]


def preprocess_with_cache(
    subject: int,
    runs: list,
    cache_dir: str = '.cache/preprocessed',
    **kwargs
):
    """
    Prétraitement avec mise en cache sur disque.
    Évite de re-télécharger et re-traiter les données.
    """
    cache_key = get_cache_key(subject, runs, **kwargs)
    cache_path = os.path.join(cache_dir, f"{cache_key}.npz")

    if os.path.exists(cache_path):
        data = np.load(cache_path)
        return data['X'], data['y']

    # Prétraitement normal
    X, y, _ = preprocess_subject(subject, runs, **kwargs)

    # Sauvegarder en cache
    os.makedirs(cache_dir, exist_ok=True)
    np.savez_compressed(cache_path, X=X, y=y)

    return X, y


# Pour les prédictions en batch
def predict_batched(pipeline, X, batch_size=32):
    """Prédiction par batch pour économiser la mémoire."""
    predictions = []
    for i in range(0, len(X), batch_size):
        batch = X[i:i+batch_size]
        predictions.extend(pipeline.predict(batch))
    return np.array(predictions)
```

---

## 9. Priority Action Items

### Immédiat (< 1 jour) - 4-6h

- [ ] **SEC-01** : Implémenter `load_model_secure()` avec validation SHA256
- [ ] **SEC-02** : Ajouter `validate_and_resolve_path()` pour bloquer path traversal
- [ ] Remplacer `except Exception` par des exceptions spécifiques dans `preprocess.py`
- [ ] Ajouter le flag `--debug` dans `mybci.py`

### Court terme (< 1 semaine) - 8-12h

- [ ] Créer `logging_config.py` avec rotation des logs
- [ ] Migrer tous les `print()` vers `logging` (15 fichiers)
- [ ] Ajouter tests de sécurité (persistence, path traversal)
- [ ] Ajouter tests edge cases (NaN, Inf, données insuffisantes)
- [ ] Documenter les changements dans CHANGELOG.md

### Moyen terme (< 1 mois) - 16-24h

- [ ] Optimiser `PSDExtractor` et `BandPowerExtractor` (vectorisation)
- [ ] Implémenter le caching des données préprocessées
- [ ] Ajouter validation des entrées utilisateur
- [ ] Compléter la couverture de tests à 80%+
- [ ] Audit pip-audit et mise à jour des dépendances

### Long terme (1-3 mois) - 20-40h

- [ ] Refactorer pour supporter d'autres datasets EEG
- [ ] Ajouter CI/CD avec tests automatiques
- [ ] Implémenter le monitoring des prédictions
- [ ] Documentation utilisateur complète

---

## 10. Commit Recommendations

### Groupe 1 : Security Fixes (Priority: Critical)

```
feat(security): add SHA256 integrity check for model files

- Add compute_file_hash() for SHA256 computation
- Implement save_model_secure() with hash generation
- Implement load_model_secure() with hash verification
- Add .sha256 sidecar files for all models

Fixes: SEC-01
```

```
feat(security): add path traversal protection

- Add validate_and_resolve_path() utility
- Integrate path validation in save_model and save_plot
- Block ../ and absolute paths outside allowed directories

Fixes: SEC-02, SEC-03
```

### Groupe 2 : Error Handling (Priority: High)

```
refactor(errors): replace broad except with specific handlers

- Replace except Exception in preprocess.py
- Add specific handlers for FileNotFoundError, ValueError, OSError
- Improve error messages with context

Fixes: #issue
```

```
feat(logging): add centralized logging with rotation

- Add logging_config.py with setup_logging()
- Configure RotatingFileHandler for log rotation
- Add --debug flag to CLI
- Migrate print() to logging in all modules
```

### Groupe 3 : Testing (Priority: Medium)

```
test(security): add security-focused test cases

- Add test_persistence_security.py
- Test hash verification, corruption detection
- Test path traversal blocking
```

```
test(edge-cases): add edge case tests

- Add tests for NaN/Inf handling
- Add tests for insufficient data
- Add tests for singular covariance
```

### Groupe 4 : Performance (Priority: Low)

```
perf(features): optimize PSD extraction with vectorization

- Replace nested loops with np.apply_along_axis
- Add batched prediction utility
- Add preprocessed data caching
```

---

## 11. Files Requiring Most Attention (Top 10)

| Rang | Fichier | Raison | Priorité |
|------|---------|--------|----------|
| 1 | `src/training/persistence.py` | Vulnérabilité sécurité critique | Critical |
| 2 | `src/preprocess.py` | except Exception, gestion erreurs | High |
| 3 | `src/visualization/_base.py` | Path traversal possible | High |
| 4 | `src/mybci.py` | Point d'entrée, logging à améliorer | Medium |
| 5 | `src/features.py` | Optimisation performance | Medium |
| 6 | `src/predict.py` | Validation entrées manquante | Medium |
| 7 | `src/training/core.py` | Logging inconsistant | Low |
| 8 | `src/training/comparison.py` | except trop large | Low |
| 9 | `src/transforms/csp.py` | Documentation edge cases | Low |
| 10 | `tests/conftest.py` | Fixtures à enrichir | Low |

---

## 12. Final Recommendations Summary

| Catégorie | Score | Priorité | Action Principale |
|-----------|-------|----------|-------------------|
| **Security** | 5/10 | Critical | Implémenter validation SHA256 + path protection |
| **Error Handling** | 6/10 | High | Remplacer except larges, centraliser logging |
| **Testing** | 7/10 | Medium | Ajouter tests sécurité et edge cases |
| **Code Quality** | 7.5/10 | Medium | Unifier logging, compléter type hints |
| **Performance** | 7/10 | Low | Vectoriser extracteurs, ajouter caching |
| **Documentation** | 7.5/10 | Low | Documenter changements sécurité |

---

## 13. Appendix

### Template de Test Fixture Enrichie

```python
@pytest.fixture
def corrupted_model_file(tmp_path, trained_csp_pipeline):
    """Génère un fichier modèle corrompu pour tests de sécurité."""
    pipeline, X, y = trained_csp_pipeline
    model_path = str(tmp_path / "corrupted.pkl")

    # Sauvegarder normalement
    save_model_secure(pipeline, model_path)

    # Corrompre
    with open(model_path, 'ab') as f:
        f.write(b'\x00\x00CORRUPT')

    return model_path
```

### Exemple de Commit Message Complet

```
feat(security): implement secure model loading with SHA256 verification

Problem:
The current load_model() function uses joblib.load() without any
integrity verification. A malicious .pkl file could execute arbitrary
code during deserialization.

Solution:
- Add compute_file_hash() to calculate SHA256 hashes
- Modify save_model() to generate .sha256 sidecar files
- Add load_model_secure() with hash verification before loading
- Raise ValueError on hash mismatch with detailed error message

Breaking Changes:
- New models will include .sha256 files
- Old models without hash files will load with warning

Testing:
- Added test_persistence_security.py with 5 new tests
- All existing tests pass

Security: This addresses a critical vulnerability (SEC-01) where
untrusted model files could compromise the system.

Refs: CODE_AUDIT_REPORT.md Section 5
```

---

## Changelog Suggéré

### v1.1.0 - Security & Stability Update

#### Security
- **BREAKING**: Model files now include SHA256 hash verification
- Added path traversal protection in save functions
- Updated error handling to prevent information leakage

#### Changed
- Replaced all `print()` with structured `logging`
- Added `--debug` flag for verbose output
- Centralized logging configuration with rotation

#### Fixed
- Replaced broad `except Exception` with specific handlers
- Fixed potential memory issues with large datasets

#### Added
- New security tests (`test_persistence_security.py`)
- Edge case tests for NaN/Inf handling
- Performance optimization for PSD extraction

---

## Estimation du Temps par Groupe

| Groupe | Sévérité | Temps Estimé | Description |
|--------|----------|--------------|-------------|
| Security Fixes | Critical | 4-6 heures | SHA256, path validation |
| Error Handling | High | 6-8 heures | Logging, exceptions |
| Testing | Medium | 8-12 heures | Nouveaux tests |
| Performance | Low | 4-6 heures | Optimisations |
| **Total** | | **22-32 heures** | ~4-5 jours de travail |

---

*Rapport généré automatiquement par Claude Code Audit*
*Date: 2025-12-11*
