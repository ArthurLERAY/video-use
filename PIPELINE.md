# PIPELINE — Extraction de clips & montages verticaux depuis une VOD (100 % local, sans ElevenLabs)

Runbook **model-agnostic** : n'importe quel agent LLM (ou humain) doit pouvoir exécuter ce pipeline
de bout en bout en suivant ce document, sans improviser. Chaque étape donne la commande exacte,
le format d'entrée/sortie, les paramètres et les critères de vérification. Les seuls points de
jugement (sélection éditoriale, mesure de la facecam) ont une procédure explicite.

Validé le 2026-07-11 sur une VOD réelle de 2h27 (macOS Apple Silicon).

---

## 0. Vue d'ensemble

```
VOD (.mp4, jusqu'à 6h)
  │
  ├─ 1. transcribe_local.py   → edit/transcripts/<stem>.json   (word-level, format Scribe)
  ├─ 2. identify_speaker.py   → speaker_id "streamer"/"guest_N" injectés dans le transcript
  ├─ 3. audio_peaks.py        → edit/peaks/<stem>.md            (pics de loudness = candidats temps forts)
  ├─ 4. pack_transcripts.py   → edit/takes_packed.md            (transcript lisible, horodaté, étiqueté)
  ├─ 5. sélection éditoriale  → edit/clips_candidates.json      (LLM, brief verbatim §5)
  ├─ 6a. extract_clips.py     → edit/clips/*.mp4 + clips.md     (liste de clips bruts)
  └─ 6b. compose_vertical.py  → edit/<montage>.mp4              (montage 9:16 sous-titré)
```

Coûts/durées observés (VOD 2h27, Mac M-series) :

| Étape | Durée | Coût |
|---|---|---|
| Transcription (whisper large-v3-turbo, MLX) | ~8 min | 0 € |
| Diarisation (pyannote community-1, MPS) | ~5 min | 0 € |
| Identification streameuse (ECAPA) | ~2 min | 0 € |
| Pics audio | ~1 min | 0 € |
| Sélection éditoriale (LLM) | ~3 min | ~60k tokens — **seul poste LLM** |
| Extraction 20 clips / montage 50 s | ~7 min / ~3 min | 0 € |

**Règle modèle** : les sous-agents LLM de ce pipeline utilisent **Opus** (`model: "opus"`),
jamais Fable (choix explicite de l'utilisateur, coût). La logique hors sélection éditoriale
ne consomme aucun token.

---

## 1. Installation (une seule fois)

### 1.1 Environnement Python

Le repo utilise un venv **uv** isolé (ne pas installer dans l'environnement conda de la machine,
il contient des builds torch dev à ne pas perturber). Tous les helpers se lancent avec
`.venv/bin/python` depuis la racine du repo.

```bash
cd <repo video-use>
uv venv --python 3.12
uv pip install -e . mlx-whisper "pyannote.audio>=3.1" speechbrain soundfile
```

Vérification : `.venv/bin/python -c "import mlx_whisper, pyannote.audio, speechbrain, soundfile"`
doit sortir sans erreur (un warning torchcodec est **normal**, voir §8.2).

### 1.2 ffmpeg

`ffmpeg` et `ffprobe` doivent être sur le PATH. Si erreur `Library not loaded ... .dylib`
(dépendances Homebrew supprimées par un `brew autoremove`), voir §8.1 — ne PAS faire
`brew reinstall ffmpeg` si c'est un tap custom.

### 1.3 Token Hugging Face (diarisation)

1. Créer un token « read » sur hf.co/settings/tokens.
2. Accepter les conditions (formulaire instantané, gratuit) de :
   **pyannote/speaker-diarization-community-1** ← le seul requis avec pyannote.audio 4.x.
3. Écrire `HF_TOKEN=hf_...` dans `.env` à la racine du repo (gitignoré).

Vérifier l'accès (doit répondre 200) :
```bash
curl -s -o /dev/null -w "%{http_code}" -H "Authorization: Bearer $HF_TOKEN" \
  https://huggingface.co/pyannote/speaker-diarization-community-1/resolve/main/config.yaml
```

### 1.4 Organisation des fichiers

```
<videos_dir>/                     ex. ~/Movies/vods/
├── <vod>.mp4                     la VOD source, jamais modifiée
├── reference/
│   ├── <extrait_voix>.mp4        30-60s de la streameuse parlant SEULE
│   └── streamer_ref.npy          empreinte vocale (produite par l'étape 2a)
└── edit/                         TOUTES les sorties du pipeline (créé automatiquement)
    ├── cache/<stem>_16k.wav      WAV mono 16kHz partagé entre étapes (extrait une seule fois)
    ├── transcripts/<stem>.json   transcript word-level format Scribe
    ├── diarization/<stem>.json   segments de parole + rapport de similarité
    ├── peaks/<stem>.{json,md}    pics de loudness
    ├── takes_packed.md           transcript packé lisible
    ├── clips_candidates.json     sélection éditoriale
    ├── clips/                    clips extraits + clips.md
    ├── <montage>.mp4 + .srt      montages verticaux + leur spec JSON
    └── project.md                journal des sessions (à tenir, voir §9)
```

Ne jamais écrire dans le repo video-use pendant une session d'édition (règle du skill).

---

## 2. Étape 1 — Transcription locale

```bash
.venv/bin/python helpers/transcribe_local.py <videos_dir>/<vod>.mp4
```

- Modèle par défaut : `mlx-community/whisper-large-v3-turbo` (~1,6 Go téléchargés au premier usage).
- `--language fr` optionnel (auto-détection sinon). `--model` pour changer de modèle.
- **Cache** : si `edit/transcripts/<stem>.json` existe, l'étape est sautée. Ne JAMAIS re-transcrire
  une source inchangée ; pour forcer, supprimer le JSON.
- Sortie : JSON **format Scribe** — contrat consommé par tout l'aval :

```json
{"language_code": "fr", "text": "...", "words": [
  {"text": "Bonjour", "start": 0.0, "end": 0.46, "type": "word", "speaker_id": null},
  {"text": " ", "start": 5.4, "end": 6.38, "type": "spacing"}
]}
```

Particularités gérées par le script (ne pas réimplémenter) : fusion des fragments d'apostrophe
de Whisper (« aujourd » + « 'hui »), `condition_on_previous_text=False` et
`hallucination_silence_threshold=2.0` contre les hallucinations sur silences/musique.

**Vérification** : le script affiche `words: N, language: XX`. N doit être plausible
(≈ 60-80 mots/min de parole effective). Ouvrir le JSON et contrôler 2-3 timestamps au hasard.

---

## 3. Étape 2 — Identification de la streameuse (empreinte vocale, PAS temps de parole)

Principe : la reconnaissance « le locuteur qui parle le plus = la streameuse » **ne marche pas**
(elle peut parler moins que les invités). On enrôle sa voix une fois, puis on matche par
similarité cosinus d'embeddings ECAPA — indépendant du temps de parole.

### 2a. Enrôlement (une fois par streameuse, réutilisable sur toutes ses VODs)

Entrée : 30-60 s où elle parle **seule** (fichier dédié, ou n'importe quelle source + `--start/--end`).

```bash
.venv/bin/python helpers/identify_speaker.py enroll <reference.mp4> \
  [--start 12.0 --end 55.0] -o <videos_dir>/reference/streamer_ref.npy
```

Minimum accepté 5 s (refusé en dessous), warning sous 15 s, idéal 30-60 s.

### 2b. Diarisation + étiquetage (par VOD, APRÈS l'étape 1)

```bash
.venv/bin/python helpers/identify_speaker.py label <videos_dir>/<vod>.mp4 \
  --reference <videos_dir>/reference/streamer_ref.npy [--num-speakers N] [--threshold 0.30]
```

- Diarisation pyannote community-1 sur MPS (fallback CPU automatique). ~5 min pour 2h27.
- Le script imprime le **rapport de similarité** — LE point de contrôle de l'étape :

```
cluster similarity report (threshold 0.30):
  SPEAKER_00: sim=+0.773  talk=1658s  → streamer ← STREAMER
  SPEAKER_01: sim=+0.136  talk=1063s  → guest_1
```

**Interprétation (voix humaines réelles)** : même locuteur ≈ 0.5-0.8 ; locuteurs différents ≈ 0.0-0.3.
Un match sain montre un écart net (ex. 0.77 vs 0.14). Si le meilleur score est < 0.30 : aucun
cluster n'est étiqueté streamer et un WARNING s'affiche → vérifier la qualité de l'extrait de
référence (voix seule ? micro propre ?) avant de baisser `--threshold`.

- Effets : réécrit les `speaker_id` du transcript (`streamer`, `guest_1`… par temps de parole
  décroissant) + sidecar `edit/diarization/<stem>.json` (segments + mapping + similarités).
- `--num-speakers N` si le nombre d'intervenants est connu (améliore la diarisation).

### 2c. Re-packer le transcript étiqueté

```bash
.venv/bin/python helpers/pack_transcripts.py --edit-dir <videos_dir>/edit
```

Sortie `edit/takes_packed.md` : une ligne par phrase `[début-fin] S<locuteur> texte`.
C'est **l'artefact que lit le LLM** pour la sélection (≈ 90 Ko pour 2h27 → lisible en 1 passe ;
au-delà de ~3-4h, fenêtrer par tranches de 30-60 min en sous-agents parallèles).

---

## 4. Étape 3 — Pics audio (pré-filtre gratuit des temps forts)

Remplace le signal « débit du chat Twitch » (non utilisé dans ce pipeline, choix utilisateur).

```bash
.venv/bin/python helpers/audio_peaks.py <videos_dir>/<vod>.mp4 [--top 40] [--min-gap 45]
```

Méthode : RMS par 0,5 s → lissage 4 s → soustraction de la médiane glissante 5 min (baseline)
→ top-N des « lifts » avec écart minimal entre pics. Rires/cris/hype = lift fort (+20 dB et plus)
avec sustain de plusieurs secondes.

Sortie : `edit/peaks/<stem>.md` (lisible LLM, timestamps HH:MM:SS triés par force) + `.json`.
Sur la VOD de validation, 14 des 20 meilleurs clips coïncidaient avec un pic mesuré.

---

## 5. Étape 4 — Sélection éditoriale (SEULE étape LLM)

Lancer un sous-agent (**modèle : Opus**) avec ce brief **verbatim** — remplacer uniquement
les chemins et la description du contenu :

```
Tu es monteur vidéo pour du contenu Twitch/TikTok. Tu dois présélectionner les moments forts
d'une VOD de <description : jeu, nb de joueurs, langue, durée>.

LIS ces deux fichiers :
1. <videos_dir>/edit/takes_packed.md — transcript en phrases horodatées [début-fin] en secondes
2. <videos_dir>/edit/peaks/<stem>.md — pics d'intensité audio avec timestamps HH:MM:SS et dB

MISSION : identifie 15 à 20 moments candidats pour des clips courts (10-60s). Cherche :
- punchlines, vannes, chambrage entre amis
- fous rires (croise avec les pics audio : pic fort + texte drôle autour = excellent candidat)
- moments de panique/échec/clutch dans le jeu
- phrases absurdes ou quotables hors contexte
- débuts d'histoires racontées

Le transcript vient de Whisper : il y a des erreurs de reconnaissance, ignore les phrases
incohérentes isolées, mais une phrase mal transcrite entourée d'un gros pic audio peut quand
même être un fou rire exploitable.

RÈGLES :
- start/end en secondes, alignés sur les bornes des phrases du transcript (1-3 phrases de
  contexte avant la punchline, ~2s de réaction après)
- durée cible d'un clip : 10 à 60 secondes
- pour chaque candidat, indique s'il coïncide avec un pic audio (et sa force)

SORTIE (réponse finale = uniquement ce JSON, aucun texte autour) :
[{"start": 123.4, "end": 152.0, "quote": "la phrase clé", "why": "raison en une ligne",
  "peak": "+33dB à 00:26:19" ou null, "score": 1-10}, ...]
Classe par score décroissant.
```

Écrire le JSON retourné dans `edit/clips_candidates.json`. Les étiquettes de voix du transcript
permettent de pondérer (moments impliquant la streameuse) — le champ « Voix » de clips.md (étape 6a)
donne la répartition mesurée par clip.

---

## 6. Étape 5 — Production

### 6a. Liste de clips bruts

```bash
.venv/bin/python helpers/extract_clips.py <videos_dir>/<vod>.mp4
```

- Lit `edit/clips_candidates.json`. Padding fixe : **-0,25 s avant / +0,75 s après** les bornes
  (absorbe la dérive des timestamps Whisper, garde l'air de réaction). Réencode x264 crf 20
  (coupes précises à la frame ; le stream-copy couperait aux keyframes).
- Nommage : `clip_{rang:02d}_s{score}_{HHMMSS}.mp4`. Récap : `edit/clips/clips.md` (citation,
  raison, pic, **répartition des voix par clip** calculée depuis la diarisation).

### 6a-bis. Clips « cris de la streameuse » (courts et nerveux, zéro token)

Format préféré de l'utilisateur : clips de 5-12 s centrés sur un cri de la streameuse
(contexte 1-3 s → cri → réaction courte). Détection purement algorithmique :

```bash
.venv/bin/python helpers/scream_clips.py <videos_dir>/<vod>.mp4 [--top 12] [--min-lift 15] [--min-gap 30]
.venv/bin/python helpers/extract_clips.py <videos_dir>/<vod>.mp4 \
  --candidates <videos_dir>/edit/clips_screams_candidates.json \
  --out-dir clips_screams --pad-before 0.15 --pad-after 0.5
```

Méthode : lift de loudness à lissage court (1 s) mesuré **uniquement dans les segments
diarisés « streamer »** ; bornes calées sur les silences naturels du transcript
(gap ≥ 0,4 s avant, ≥ 0,5 s après). Prérequis : étapes 1 et 2b faites. Un candidat sans
citation (« ») = cri sans mots reconnus, souvent exploitable quand même — vérifier à l'oreille.
Padding réduit (0,15/0,5 s) : le rythme prime sur l'air de réaction.

### 6a-ter. Mode marqueurs — « pré-mâché » pour le streamer (RECOMMANDÉ pour le jeu compétitif)

Cas d'usage : sur un jeu comme Valorant, un clutch/ace est indétectable par l'audio ou le
transcript (le streamer est silencieux pendant l'action). La vérité terrain = les **marqueurs
de stream** posés en live (`/marker <description>` dans le chat, broadcaster/éditeurs), récupérés
via Helix `Get Stream Markers` (scope OAuth `user:read:broadcast`, champ `position_seconds`).

```bash
.venv/bin/python helpers/marker_clips.py <vod.mp4> --markers markers.json \
  [--before 30] [--after 30] [--language fr] [--no-transcript] [--out-dir clips_markers]
```

- Fenêtre **[-30 s, +30 s]** autour de chaque marqueur (le streamer marque pendant ou juste
  après le moment). Fenêtres qui se chevauchent = **fusionnées** en un seul clip.
- **Transcription fenêtrée uniquement** (tranche audio du clip, pas la VOD entière) → VOD
  traitée en quelques minutes. Aucune diarisation, aucun LLM, **aucun token HF requis** —
  l'installation minimale suffit : `uv pip install -e . mlx-whisper`.
- Clips livrés tels quels (aspect source, coupes précises x264 crf 20) + `manifest.json`
  machine-readable pour une application hôte :

```json
[{"file": "clip_01_clutch-1v3.mp4", "start": 3652.2, "end": 3728.5, "duration_s": 76.3,
  "markers": [{"id": "hx-123", "position_seconds": 3721, "description": "clutch 1v3"}],
  "title": "clutch 1v3", "transcript": "..."}]
```

Format d'entrée `markers.json` : liste de `{position_seconds, description, id?}` (un objet
marker Helix brut fonctionne, les champs en trop sont ignorés).

### 6b. Montage vertical 9:16 sous-titré

#### Étape préalable OBLIGATOIRE : mesurer la facecam (elle peut BOUGER en cours de stream)

1. Extraire une frame **dans chaque segment retenu** :
   `ffmpeg -ss <t> -i <vod> -frames:v 1 -vf scale=960:-2 frame_t<t>.jpg`
2. Regarder chaque frame : position de la facecam (les streamers la déplacent entre les parties —
   sur la VOD de validation : en haut à gauche en début de stream, en haut à droite à la fin).
3. Pour chaque position distincte, définir un crop `w:h:x:y` (coordonnées **source 1920×1080**) :
   - l'aspect w/h DOIT valoir `1080/top_h` (1,8 pour top_h=600) — sinon distorsion
     (le script émet un warning au-delà de 3 % d'écart) ;
   - si la cam touche un bord de l'écran, **aligner le crop sur ce bord** (ex. cam collée à droite :
     `x = 1920 - w`) pour éviter une bande noire.
4. Crop gameplay : bande centrale pleine hauteur, aspect `1080/(1920-top_h)`.
   Pour top_h=600 : `crop=884:1080:518:0`. Vérifier que ce crop ne chevauche pas la facecam.

#### Choisir les segments (règles de cut)

- Structure : **hook** (le moment le plus intriguant/absurde, pose le contexte en ≤ 12 s)
  → **escalade** → **chute** (la punchline la plus forte en dernier).
- Bornes sur les **frontières de mots** du transcript packé, padding 30-200 ms aux bords
  (50 ms avant le premier mot, 500-700 ms après le dernier : l'air de réaction fait partie du beat).
- **Supprimer les silences internes > 2 s** en scindant un moment en deux segments consécutifs.
- Durée totale cible : 45-55 s. Un segment = 4-16 s.

#### Écrire le spec JSON et rendre

Format complet documenté en tête de `helpers/compose_vertical.py`. Exemple réel validé :
`tiktok_v1_spec.json` (5 segments, 48 s, deux positions de cam) conservé dans `edit/` de la
VOD de validation.

```bash
.venv/bin/python helpers/compose_vertical.py <videos_dir>/edit/<montage>_spec.json
```

Le script applique les règles de production du skill sans intervention : extraction par segment
(vstack cam/jeu + fondus audio 30 ms), concat lossless `-c copy`, SRT master sur la timeline de
sortie (`render.build_master_srt`, chunks de 2 mots MAJUSCULES), **nettoyage des cues sans
caractère alphanumérique** (artefacts Whisper « ! »), sous-titres incrustés **EN DERNIER**
(style `SUB_FORCE_STYLE` de render.py — MarginV=90 = zone sûre UI TikTok/Reels/Shorts, ne pas
descendre sous 75), et vérifie la durée finale vs attendue.

---

## 7. Auto-évaluation OBLIGATOIRE avant livraison

Ne jamais présenter un rendu sans ces contrôles (cap : 3 passes de correction, ensuite signaler) :

1. **Durée** : `ffprobe` ≈ somme des segments (±0,2 s).
2. **Grille de frames** : un échantillon par segment + zones de coupe :
   ```bash
   ffmpeg -i final.mp4 -vf "select='eq(n\,F1)+eq(n\,F2)+...',scale=360:640,tile=3x2" \
     -frames:v 1 -vsync 0 tiles.jpg
   ```
   Contrôler sur l'image : facecam cadrée sans bande noire sur CHAQUE segment (la cam bouge !),
   gameplay pertinent (pas de menu/écran noir), sous-titres lisibles et non masqués, pas de
   flash/discontinuité aux coupes.
3. **Audio** : `volumedetect` sur le rendu (mean_volume ≈ -30 à -20 dB) ; écouter les coupes
   si doute (les fondus 30 ms doivent supprimer tout pop).
4. **Sous-titres** : aucune cue de ponctuation seule ; vérifier 2-3 cues vs l'audio.
5. Si un défaut est trouvé : corriger le spec/le SRT, re-rendre, re-contrôler.

Défauts déjà rencontrés et leurs corrections : bande noire cam (→ crop aligné au bord),
cues « ! » (→ nettoyage désormais automatique dans compose_vertical.py).

---

## 8. Dépannage (pièges rencontrés et résolus)

### 8.1 ffmpeg cassé (`Library not loaded ... .dylib`)
Cause : `brew autoremove` a supprimé des dépendances ; le ffmpeg peut venir d'un **tap custom**
(`homebrew-ffmpeg/ffmpeg`) que `brew reinstall` refusera. Diagnostic + correction :
```bash
otool -L $(which ffmpeg) | awk '{print $1}' | grep "^/usr/local" | while read l; do [ -f "$l" ] || echo "MISSING: $l"; done
brew install <les formules manquantes : lame opus svt-av1 theora libogg libvorbis x264 x265 ...>
```

### 8.2 Warning torchcodec au chargement de pyannote
**Normal et géré** : torchcodec ne trouve pas les dylibs ffmpeg. `identify_speaker.py` contourne
en passant l'audio préchargé en mémoire (`{"waveform": tensor, "sample_rate": sr}`).
Ne pas essayer de réparer torchcodec.

### 8.3 pyannote 4.x — trois pièges d'API
- `Pipeline.from_pretrained(..., token=...)` (plus `use_auth_token`, géré par fallback).
- Le nom `pyannote/speaker-diarization-3.1` est **redirigé** vers `community-1` → c'est le gate
  community-1 qu'il faut accepter (un 403 GatedRepoError malgré un token valide = gate manquant).
- Le pipeline renvoie un `DiarizeOutput`, l'Annotation est dans `.speaker_diarization` (géré).

### 8.4 Divers
- Whisper découpe les apostrophes en tokens séparés → fusion automatique dans transcribe_local.py.
- Voix de synthèse (TTS) : les embeddings ECAPA de deux voix TTS du même moteur sont quasi
  identiques (~0.9) — les tests de similarité ne sont significatifs que sur voix humaines.
- `edit/cache/<stem>_16k.wav` est partagé : le supprimer force la ré-extraction partout.

---

## 9. Journal de session

Après chaque session, appendre à `<videos_dir>/edit/project.md` : stratégie, décisions
(segments choisis + pourquoi), corrections faites, points en suspens. C'est la mémoire du
projet d'édition — la lire en début de session suivante.
