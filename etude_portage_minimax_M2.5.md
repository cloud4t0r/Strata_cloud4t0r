# Plan de portage : MiniMax M2.5 → Strata (HIP, RDNA4 + RDNA2)

## Matériel cible
- RX 9070 16 Go (gfx1201, RDNA4) — carte principale
- RX 6800 16 Go (gfx1030, RDNA2) — tier helper expert cache
- Ryzen 9 5950X (AVX2, sans AVX-512), 64 Go DDR4
- MSI X570-A PRO (slots x4 chipset partagés, uplink ~8 Go/s)
- Fedora 44 (Linux)

## Recensement des octets (M2.5)

| Partie | Params | Précision | Taille | Où |
|---|---|---|---|---|
| Attention GQA (62 couches) | 3.12B | Q8_0 | ~3.1 GB | VRAM 9070 |
| Embeddings + lm_head | 1.23B | Q6_K | ~0.9 GB | VRAM 9070 |
| MTP head (1 couche, multi-step) | ~3.7B | Q6_K | ~2.8 GB | VRAM 9070 |
| Normes (incl. q_norm/k_norm) | ~0.4M | BF16 | ~0.01 GB | VRAM 9070 |
| Spine total | | | ~6.8 GB | |
| KV cache int8 @ 64K | — | int8 | 7.9 GB | VRAM 9070 |
| Fixe VRAM @ 64K | | | ~14.7 GB | |
| Cache experts VRAM (9070) | — | IQ3_S | ~1.3 GB ≈ 245 experts | reste |
| Pool experts routés (15 872) | 225.8B | IQ3_S (~3 bpw) | ~84 GB | RAM + SSD |
| Cache experts 6800 (tier) | — | IQ3_S | ~13 GB ≈ 2 450 experts | VRAM 6800 |

Constat clé : à 64K, le spine + KV int8 consomme ~14.7 Go sur 16 Go → le cache d'experts sur la 9070 est quasi nul (~245 experts, ~1.5 %). Le tier 6800 (13 Go, ~2 450 experts) porte l'essentiel de la résidence (~15.5 %). Taux de hit attendu ~60-70 %. Octets effectifs depuis la RAM/SSD : ~0.8-1.0 Go/token.

Décode estimé : 12-20 tok/s (conservateur). Avec MTP (1.5-1.7x) : ~18-30 tok/s effectifs. Préfill : ~300-500 tok/s.

Alternative 32K : KV 4.0 Go → cache 9070 ~8.0 Go ≈ 1 500 experts → résidence combinée ~25 % → hit ~75-80 % → décode ~18-28 tok/s. Recommandé si la qualité à 64K n'est pas indispensable.

## Delta d'architecture vs Qwen3.8-Flash-Next

### À réutiliser sans modification
Chargement GGUF, arène d'experts RAM épinglée, expert cache adaptatif (swap), doorbell/CUDA graphs, serveur OpenAI/Anthropic, path MTP (si GGUF inclut la tête).

### À réécrire / supprimer (spécifique Qwen3.8)
| Composant Qwen3.8 | Action pour M2.5 |
|---|---|
| GDN (36 couches récurrentes) | Supprimer — attention dense GQA standard |
| QSA (12 couches sparse) | Supprimer — attention dense GQA standard |
| PLE n-gram (51B, SSD) | Supprimer — pas de table n-gram |
| Gated Residual 4-branches | Supprimer — residual standard |
| Router softmax top-10/512 | Réécrire — sigmoid + expert bias + top-8/256 + renormalisation |
| Expert SwiGLU 2560→640→2560 | Adapter — SwiGLU 3072→1536→3072 |
| MTP fetch depuis checkpoint BF16 | Adapter — vérifier si GGUF inclut la tête MTP |
| Shared expert (1 par couche) | Supprimer — M2.5 n'a pas d'expert partagé |
| q_norm/k_norm RMSNorm | Ajouter — présent dans M2.5 (absent de Qwen3.8) |
| Partial RoPE | Adapter — facteur rotatif partiel (rotary_dim/head_dim) |

### Nouveaux éléments
- RMSNorm sur Q et K dans chaque couche d'attention.
- Sigmoid gating avec biais par expert (e_score_correction_bias, buffer 256 floats/couche) — le biais n'affecte que la sélection (top-k), pas les poids (renormalisation par somme des top-k).
- Pas de softmax dans le router.

## Plan de portage (jalons)

| Jalon | Contenu | Critère d'acceptance |
|---|---|---|
| M0 | Build HIP multi-arch (gfx1030+gfx1201) ; audit topologie X570-A PRO ; spike cross-arch helper ; vérifier GGUF M2.5 inclut MTP head | échange doorbell/peer fonctionne sur les 2 cartes ; si non → repli mono-GPU |
| M1 | Packer GGUF→pack Strata (blobs 3072×1536, spine, MTP, profile) ; --admission-only | plan mémoire = recensement ±10 % ; 15 872 blobs extraits |
| M2 | Chemin dense (GQA attention, q_norm/k_norm, RoPE partiel, residual standard) ; parité logits vs llama.cpp | KL < 0.05 sur 43 tests ; greedy identique llama.cpp |
| M3 | Chemin expert (blobs/CPU pool/sigmoid router) ; décode IQ3_S | ≥ 12 tok/s @ 4K ; ≥ 300 tok/s préfill |
| M4 | MTP + cache hit rate | acceptation MTP ≥ 1.5x ; hit rate ≥ 65 % |
| M5 | Bench final (4K/32K/64K) + docs | hit rate VRAM, jeux de mesures, comparaison avec GLM-4.5-Air |

## Risques et mitigations

| Risque | Impact | Mitigation |
|---|---|---|
| VRAM saturée @ 64K (14.7 Go / 16 Go) | cache experts quasi nul sur la 9070 | config 32K recommandée ; ou Q4 attention + MTP sur CPU |
| Multi-arch gfx1030+gfx1201 dans un run | tier 6800 inutilisable | M0 spike ; repli mono-GPU documenté |
| Cross-arch jamais prouvé publiquement | perte du tier | PR #1184 validé sur gfx1032+gfx1030 ; notre cas non testé |
| 5950X sans AVX-512 | misses CPU ~15-25 % plus lents | déjà dans les estimations ; IQ3_S utilise ggml (pas de chemin AVX-512) |
| GGUF M2.5 sans tête MTP | pas de draft | vérifier à M0 ; si absent, fetch depuis checkpoint BF16 (seuls les tenseurs MTP ~3.7B) |
| Licence Modified-MIT | restrictions commerciales possibles | vérifier les termes avant distribution |
| X570-A PRO : uplink partagé | préfill limité | les deux cartes sur le chipset → ~8 Go/s partagés ; optimiser --pcie-frac bas |
| 6800 en Gen2 x4 (si M.2 chipset occupé) | tier inutile | vérifier à M0 ; déplacer NVMe sur M.2 CPU si possible |

## Questions avant implémentation

1. Contexte : 32K (recommandé, cache ~1 500 experts sur la 9070) ou 64K (cache ~245, décode plus lent) ?
2. Quant : IQ3_S (~84-92 Go, ~20-28 Go sur SSD) ou IQ2_K (~78 Go, ~14 Go sur SSD, qualité moindre) ?
3. GLM-4.5-Air vs MiniMax M2.5 : GLM tient en RAM (decode ~22-30 tok/s vs ~12-20 pour M2.5) avec un pool 2.5x plus petit. Confirmez M2.5 malgré le décode plus lent et la VRAM plus tendue ?
4. MTP : si les GGUF M2.5 n'incluent pas la tête MTP, êtes-vous prêt à télécharger les ~3.7B tenseurs depuis le checkpoint BF16 ?

