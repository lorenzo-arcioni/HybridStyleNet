# RAG-ColorNet: Retrieval-Augmented Grading Network
## Architettura Completa per Photographer-Specific Color Grading in Regime Few-Shot

---

## Indice

1. [Definizione Formale del Problema](#1-definizione-formale-del-problema)
2. [Visione d'Insieme dell'Architettura](#2-visione-dinsieme-dellarchitettura)
3. [Componente 1 — Scene Encoder](#3-componente-1--scene-encoder)
4. [Componente 2 — Cluster Assignment](#4-componente-2--cluster-assignment)
5. [Componente 3 — Local Retrieval Module](#5-componente-3--local-retrieval-module)
6. [Componente 4 — Bilateral Grid Renderer](#6-componente-4--bilateral-grid-renderer)
7. [Componente 5 — Confidence Mask](#7-componente-5--confidence-mask)
8. [Forward Pass Completo](#8-forward-pass-completo)
9. [Funzione di Loss Composita](#9-funzione-di-loss-composita)
10. [Training: Tre Fasi](#10-training-tre-fasi)
11. [Meta-Learning con Reptile](#11-meta-learning-con-reptile)
12. [Few-Shot Adaptation](#12-few-shot-adaptation)
13. [Memoria Incrementale](#13-memoria-incrementale)
14. [Dataset](#14-dataset)
15. [Proprietà Matematiche e Teoremi](#15-proprietà-matematiche-e-teoremi)
16. [Risultati Attesi](#16-risultati-attesi)
17. [Analisi della Complessità Computazionale](#17-analisi-della-complessità-computazionale)
18. [Ablation Studies](#18-ablation-studies)
19. [Confronto con lo Stato dell'Arte](#19-confronto-con-lo-stato-dellarte)

---

## 1. Definizione Formale del Problema

### 1.1 Intuizione

Uno stile fotografico professionale non è una trasformazione continua arbitraria. È un insieme finito di **modalità stilistiche** — internamente chiamate cluster — ognuna delle quali corrisponde a un contesto fotografico specifico (golden hour, interno, overcast, ecc.). Quando un fotografo grada una foto, esegue implicitamente due operazioni:

1. **Assegnazione modale**: riconosce il contesto cromatico globale della scena e seleziona il preset (cluster) di stile appropriato
2. **Trasferimento locale**: una volta applicato il preset, applica il trattamento cromatico che ha usato su foto simili in passato — non una media, ma una combinazione pesata degli esempi più pertinenti per ogni zona dell'immagine.

Questo processo è esattamente quello che RAG-ColorNet modella esplicitamente.

### 1.2 Formulazione Matematica

**Training set del fotografo:**

$$\mathcal{D}_\phi = \{(I_i^{src}, I_i^{tgt})\}_{i=1}^N, \quad N \in [50, 500]$$

dove $I_i^{src} \in [0,1]^{H \times W \times 3}$ è l'immagine sorgente in sRGB e $I_i^{tgt} \in [0,1]^{H \times W \times 3}$ è il JPEG editato dal fotografo.

**Struttura a cluster dello stile:**

Lo stile del fotografo è modellato come una mixture di $K$ modalità stilistiche:

$$\mathcal{D}_\phi = \bigcup_{k=1}^{K} \mathcal{D}_\phi^{(k)}, \quad \mathcal{D}_\phi^{(k)} = \{(I_i^{src}, I_i^{tgt}) : z_i = k\}$$

dove $z_i \in \{1, \ldots, K\}$ è l'assegnazione al cluster della coppia $i$-esima. $K$ è appreso dai dati e varia per fotografo — tipicamente $K \in [1, 15]$.

**Modello:**

$$f_\theta: [0,1]^{H \times W \times 3} \times \mathcal{D}_\phi \to [0,1]^{H \times W \times 3}$$

Il modello non ha solo parametri $\theta$ — ha anche accesso esplicito al database $\mathcal{D}_\phi$ come memoria non parametrica. L'output dipende sia dai parametri appresi che dalle coppie di training disponibili.

**Obiettivo:**

$$\theta^* = \arg\min_\theta\; \mathbb{E}_{(I^{src}, I^{tgt}) \sim \mathcal{D}_\phi}\left[\mathcal{L}\!\left(f_\theta(I^{src}, \mathcal{D}_\phi),\; I^{tgt}\right)\right]$$

### 1.3 Proprietà Desiderate

| Proprietà | Descrizione | Come è garantita |
|---|---|---|
| **Photographer-specificity** | Output fedele allo stile del fotografo | Database esplicito per fotografo |
| **Miglioramento incrementale** | Più dati → grading migliore | Retrieval su database crescente |
| **Correttezza tecnica** | No artefatti, bilanciamento corretto | Pipeline RAW deterministica + bilateral grid |
| **Invarianza alla risoluzione** | Funziona a qualsiasi risoluzione | Coordinate normalizzate + DINOv2 patch-based |
| **Few-shot stability** | Funziona con N=50 | Parametri trainable minimi, retrieval non parametrico |
| **Latenza inferenza** | < 10s su CPU consumer | Bilateral grid + retrieval approssimato |

---

## 2. Visione d'Insieme dell'Architettura

### 2.1 Schema Generale

```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 PREPROCESSING (deterministico, §Pipeline RAW)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

RAW (ARW/DNG) ──→ linearize → demosaic → WB → color matrix
              ──→ gamma → downsample → normalize
              ──→ I_src ∈ [0,1]^{B×3×H×W}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 SCENE ENCODER (parzialmente frozen)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

I_src ──→ DINOv2-Small (frozen) ──→ F_sem ∈ R^{B×N×384}
      └──→ Color Histogram Lab  ──→ h ∈ R^{B×192}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 CLUSTER ASSIGNMENT (trainable, leggero)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

h ──→ ClusterNet (MLP 2-layer) ──→ p ∈ R^{B×K}  (soft assignment)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 LOCAL RETRIEVAL MODULE (cuore dell'architettura)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Per ogni cluster k (pesato da p_k):
    Database_k = {(F_sem_i, edit_i)} per i ∈ D_phi^(k)
    
    Cross-Image Local Attention:
    query(i,j)  = f_patch_new(i,j)        ← zona della nuova foto
    key_m(i',j')= f_patch_m(i',j')        ← zona di foto m nel cluster
    
    retrieved_edit = Σ_k p_k · Σ_m A_{km} · edit_km
    
retrieved_edit ∈ R^{B×C×H/14×W/14}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 BILATERAL GRID RENDERER
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

retrieved_edit + F_sem ──→ GridNet (leggero)
                       ──→ G_global (8×8×8) + G_local (16×16×8)
                       ──→ Bilateral Slicing con guida semantica
                       ──→ I_global, I_local

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 CONFIDENCE MASK + BLENDING
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

α = MaskNet(F_sem, |I_local - I_global|)
I_pred = α · I_local + (1-α) · I_global
```

### 2.2 Filosofia dell'Architettura

**Retrieval invece di compressione.** Le architetture parametriche (bilateral grid condizionata su style prototype) comprimono lo stile del fotografo in un vettore di pesi. Questa compressione è lossy — perde le sfumature specifiche di ogni contesto fotografico. RAG-ColorNet non comprime nulla: mantiene tutti gli esempi in memoria e usa il retrieval per trovare il più pertinente. Il vantaggio è duplice: nessuna perdita di informazione, e miglioramento automatico con più dati.

**Locale invece di globale.** Il retrieval opera a livello di patch, non di immagine intera. Questo permette di trovare, per ogni zona della nuova foto, l'esempio di training più pertinente per quella specifica zona — indipendentemente da cosa succede nel resto dell'immagine.

**Semantica per il retrieval, cromatica per il rendering.** DINOv2 fornisce feature semantiche per il retrieval (trovare zone simili tra immagini diverse). La bilateral grid con guida cromatica fa il rendering (trasformare i pixel in modo edge-aware). I due moduli hanno responsabilità separate e complementari.

---

## 3. Componente 1 — Scene Encoder

### 3.1 DINOv2-Small come Backbone Semantico

**Motivazione.** DINOv2 (Oquab et al., 2023) è addestrato con self-supervised learning su 142M immagini. Le sue patch features producono rappresentazioni dove zone semanticamente simili (pelle, cielo, vegetazione) sono vicine nello spazio delle feature indipendentemente dalla loro posizione spaziale o dalla loro luminanza. Questa proprietà è esattamente quella necessaria per il retrieval locale nel color grading: vogliamo trovare zone con lo stesso tipo di contenuto, non zone con la stessa posizione.

**Configurazione:**

- Modello: `dinov2_vits14` (Small, 21M parametri)
- Patch size: $14 \times 14$ pixel
- Dimensione embedding: $d_{dino} = 384$
- Input: immagine normalizzata con statistiche ImageNet
- Output: patch tokens $F_{sem} \in \mathbb{R}^{B \times N_{patches} \times 384}$

dove $N_{patches} = \lfloor H/14 \rfloor \cdot \lfloor W/14 \rfloor$.

**Operazione formale:**

$$F_{sem} = \text{DINOv2}(I_{src}) \in \mathbb{R}^{B \times N \times 384}$$

I patch tokens vengono estratti dall'ultimo layer del transformer, escludendo il CLS token. Ogni token $F_{sem}[b, n, :]$ rappresenta una patch $14 \times 14$ dell'immagine e porta informazione contestuale dell'intera scena grazie all'attenzione globale di DINOv2.

**Stato: completamente frozen.** I pesi di DINOv2 non vengono mai aggiornati in nessuna fase del training. Questo è fondamentale per la stabilità nel few-shot regime: con N=50-300 coppie, qualsiasi update sul backbone porta a catastrophic forgetting delle rappresentazioni semantiche pre-addestrate.

**VRAM:** DINOv2-Small in fp16 occupa ~42MB. A risoluzione $512 \times 384$ con patch size 14: $N = 36 \times 27 = 972$ token. Costo del forward pass: ~0.3s su RTX 3080.

### 3.2 Color Histogram in Spazio CIE Lab

**Motivazione.** Il cluster di appartenenza di una foto è determinato principalmente dalla distribuzione cromatica globale — golden hour ha molti rossi/arancio, interno ha poca saturazione, ecc. Un color histogram in Lab cattura questa distribuzione in modo compatto, differenziabile, e invariante alla posizione spaziale dei pixel.

**Formulazione.** Sia $I_{Lab} = \text{rgb2lab}(I_{src}) \in \mathbb{R}^{B \times H \times W \times 3}$ l'immagine nello spazio CIE Lab. Per ogni canale $c \in \{L^*, a^*, b^*\}$, si calcola un istogramma soft con $B_{hist} = 64$ bin:

$$h_c(k) = \frac{\sum_{i,j} \exp\!\left(-\frac{(I_{Lab,c}(i,j) - \mu_k)^2}{2\sigma_{bin}^2}\right)}{\sum_{k'}\sum_{i,j} \exp\!\left(-\frac{(I_{Lab,c}(i,j) - \mu_{k'})^2}{2\sigma_{bin}^2}\right)}$$

con $\mu_k = v_{min}^c + (k + 0.5) \cdot \Delta\mu^c$ centri uniformi e $\sigma_{bin} = \Delta\mu^c / 2$.

Il vettore di istogramma finale è la concatenazione sui tre canali:

$$\mathbf{h} = [h_{L^*}; h_{a^*}; h_{b^*}] \in \mathbb{R}^{3 \times 64} \equiv \mathbb{R}^{192}$$

**Proprietà:**
- **Differenziabile**: il kernel gaussiano è $C^\infty$, quindi il gradiente fluisce attraverso l'istogramma
- **Invariante alla posizione**: cattura solo la distribuzione dei colori, non dove sono
- **Compatto**: 192 scalari descrivono l'intera distribuzione cromatica dell'immagine
- **Interpretabile**: visualizzando $h_{a^*}$ e $h_{b^*}$ si capisce immediatamente il cast cromatico

### 3.3 Feature Descriptor per Retrieval

Per il retrieval locale, ogni patch necessita di un descrittore che combini informazione semantica (cosa c'è) e cromatica (com'è il colore):

$$\mathbf{q}(n) = \text{LayerNorm}\!\left(\left[\mathbf{f}_{sem}(n);\; \mathbf{c}_{patch}(n)\right]\right) \in \mathbb{R}^{384 + 32}$$

dove:
- $\mathbf{f}_{sem}(n) \in \mathbb{R}^{384}$: patch token DINOv2 (semantica)
- $\mathbf{c}_{patch}(n) \in \mathbb{R}^{32}$: statistiche cromatiche della patch (media e std di $L^*, a^*, b^*$ + istogramma locale 13-bin) proiettate a 32 dim via linear layer trainable

La LayerNorm bilancia i due contributi che hanno scale diverse (feature DINOv2 vs statistiche cromatiche).

---

## 4. Componente 2 — Cluster Assignment

### 4.1 Inizializzazione dei Cluster

I cluster vengono inizializzati **una sola volta** all'inizio del few-shot adaptation, prima di qualsiasi training. Si applica K-Means sui color histograms $\{\mathbf{h}_i\}_{i=1}^N$ delle coppie di training:

$$\{C_k\}_{k=1}^K = \text{KMeans}(\{\mathbf{h}_i\}_{i=1}^N, K^*)$$

dove $K^*$ è determinato automaticamente con il criterio del gomito (elbow method) sulla varianza intra-cluster:

$$K^* = \arg\min_K \left\{\text{WCSS}(K) : \frac{\text{WCSS}(K) - \text{WCSS}(K+1)}{\text{WCSS}(K-1) - \text{WCSS}(K)} > \tau_{elbow}\right\}$$

con $\tau_{elbow} = 0.1$. In pratica $K^* \in [1, 15]$ per la maggior parte dei fotografi.

I centroidi $\{C_k\}$ inizializzano i pesi del ClusterNet. Le coppie di training vengono assegnate ai cluster: $\mathcal{D}_\phi^{(k)} = \{(I_i^{src}, I_i^{tgt}) : \arg\min_{k'} \|\mathbf{h}_i - C_{k'}\| = k\}$.

### 4.2 ClusterNet

**Architettura:**

$$p = \text{ClusterNet}(\mathbf{h}) = \text{Softmax}\!\left(\mathbf{W}_2\, \text{ReLU}(\mathbf{W}_1\, \mathbf{h} + \mathbf{b}_1) + \mathbf{b}_2\right)$$

con $\mathbf{W}_1 \in \mathbb{R}^{256 \times 192}$, $\mathbf{W}_2 \in \mathbb{R}^{K \times 256}$.

Output: $p \in \Delta^{K-1}$ vettore di probabilità sul simplex, con $\sum_k p_k = 1$.

**Parametri:** $192 \times 256 + 256 + 256 \times K + K \approx 50K + 256K$ — meno di 100K parametri totali per $K \leq 15$.

**Interpretazione:** $p_k$ è la probabilità che la foto appartenga al cluster $k$. Un'immagine di pura golden hour avrà $p_{golden} \approx 1$. Un'immagine di transizione (golden hour in interni) avrà massa distribuita su più cluster.

### 4.3 Soft Assignment vs Hard Assignment

Si usa soft assignment invece di hard per tre ragioni:

1. **Differenziabilità**: il soft assignment è differenziabile, permettendo il gradient flow attraverso il cluster assignment durante il training
2. **Foto di transizione**: molte foto professionali sono in condizioni di luce miste — il soft assignment permette di combinare i trattamenti di cluster diversi in proporzione alla loro rilevanza
3. **Robustezza agli errori di classificazione**: se una foto è al confine tra due cluster, il soft assignment usa entrambi invece di fare una scelta binaria potenzialmente sbagliata

---

## 5. Componente 3 — Local Retrieval Module

Questo è il contributo principale di RAG-ColorNet e la componente più originale rispetto allo stato dell'arte.

### 5.1 Struttura del Database per Cluster

Per ogni cluster $k$, il database contiene le rappresentazioni pre-calcolate di tutte le coppie assegnate a quel cluster:

$$\mathcal{M}^{(k)} = \left\{\left(\mathbf{K}_i^{(k)},\; \mathbf{E}_i^{(k)}\right)\right\}_{i : z_i = k}$$

dove:
- $\mathbf{K}_i^{(k)} = \{\mathbf{q}_i(n)\}_{n=1}^{N_i} \in \mathbb{R}^{N_i \times 416}$: descrittori di tutte le patch dell'immagine $i$ nel cluster $k$
- $\mathbf{E}_i^{(k)} = \text{DINOv2}(I_i^{tgt}) - \text{DINOv2}(I_i^{src}) \in \mathbb{R}^{N_i \times 384}$: edit signature — come è cambiata ogni patch nel grading del fotografo

**NOTA:** Le chiavi $\mathbf{K}_i$ e le edit signatures $\mathbf{E}_i$ sono **pre-calcolate e cachate** durante il preprocessing del few-shot adaptation. A inference time non si ricalcola nulla dal database — si fa solo retrieval.

### 5.2 Cross-Image Local Attention

Data la nuova immagine con query $\mathbf{Q} = \{\mathbf{q}_{new}(n)\}_{n=1}^{N_{new}} \in \mathbb{R}^{N_{new} \times 416}$:

**Step 1 — Proiezioni lineari (trainable):**

$$\tilde{\mathbf{Q}} = \mathbf{Q}\mathbf{W}^Q \in \mathbb{R}^{N_{new} \times d_r}$$
$$\tilde{\mathbf{K}}_i = \mathbf{K}_i^{(k)}\mathbf{W}^K \in \mathbb{R}^{N_i \times d_r}$$
$$\tilde{\mathbf{V}}_i = \mathbf{E}_i^{(k)}\mathbf{W}^V \in \mathbb{R}^{N_i \times d_r}$$

con $\mathbf{W}^Q, \mathbf{W}^K \in \mathbb{R}^{416 \times d_r}$ e $\mathbf{W}^V \in \mathbb{R}^{384 \times d_r}$, $d_r = 256$.

**Step 2 — Attention intra-immagine per ogni foto di training:**

Per ogni foto $i$ nel cluster $k$:

$$\mathbf{A}_i = \text{Softmax}\!\left(\frac{\tilde{\mathbf{Q}}\tilde{\mathbf{K}}_i^T}{\sqrt{d_r}}\right) \in \mathbb{R}^{N_{new} \times N_i}$$

$$\mathbf{R}_i = \mathbf{A}_i \tilde{\mathbf{V}}_i \in \mathbb{R}^{N_{new} \times d_r}$$

$A_i[n, m]$ è il peso che la patch $n$ della nuova immagine assegna alla patch $m$ dell'immagine di training $i$. Una patch di pelle della nuova immagine assegnerà peso alto alle patch di pelle dell'immagine $i$, recuperando il loro editing cromatico.

**Step 3 — Aggregazione across le foto del cluster:**

Le foto di training non contribuiscono tutte allo stesso modo — quelle globalmente più simili alla nuova foto devono pesare di più. La similarità globale è:

$$s_i = \frac{1}{N_{new}} \sum_n \max_m A_i[n, m]$$

normalizzata: $\tilde{s}_i = s_i / \sum_{i'} s_{i'}$.

Il retrieved edit per il cluster $k$ è:

$$\mathbf{R}^{(k)} = \sum_{i : z_i = k} \tilde{s}_i \cdot \mathbf{R}_i \in \mathbb{R}^{N_{new} \times d_r}$$

**Step 4 — Mixture across cluster:**

$$\mathbf{R}_{final} = \sum_{k=1}^{K} p_k \cdot \mathbf{R}^{(k)} \in \mathbb{R}^{N_{new} \times d_r}$$

**Reshape a mappa spaziale:**

$$\mathbf{R}_{spatial} = \text{reshape}(\mathbf{R}_{final}) \in \mathbb{R}^{B \times d_r \times H_{14} \times W_{14}}$$

dove $H_{14} = \lfloor H/14 \rfloor$, $W_{14} = \lfloor W/14 \rfloor$.

### 5.3 Proprietà del Retrieval Locale

**Proprietà 1 (Miglioramento Monotono con i Dati).**
Sia $\mathcal{D}_\phi^{(k)}(t)$ il database del cluster $k$ al tempo $t$ con $N_k(t)$ coppie. Per ogni foto di test $I^{src}$, l'errore di retrieval atteso:

$$\epsilon(t) = \mathbb{E}\left[\left\|\mathbf{R}^{(k)}(t) - \mathbf{E}^*\right\|^2\right]$$

è monotonicamente non crescente in $N_k(t)$: $\epsilon(t+1) \leq \epsilon(t)$ quando si aggiunge una coppia al database.

*Argomentazione:* Aggiungere una coppia al database aggiunge una chiave-valore $(K_i, E_i)$ al set di retrieval. Se la nuova coppia è più simile alla query della coppia più simile esistente, il retrieval migliorerà. Se non lo è, il softmax assegnerà peso basso alla nuova coppia e l'output non peggiora. $\square$

**Proprietà 2 (Convergenza al Nearest Neighbor).**
Nel limite $N_k \to \infty$ con distribuzione densa nello spazio delle query, il retrieved edit converge al nearest neighbor esatto:

$$\lim_{N_k \to \infty} \mathbf{R}^{(k)} = \mathbf{E}_{nn}^* = \mathbf{E}_{i^*}$$

dove $i^* = \arg\max_i \tilde{s}_i$. Questo è il comportamento ottimale: con abbastanza dati, ogni zona della nuova foto ottiene esattamente l'editing che il fotografo ha applicato alla zona più simile nel suo storico. $\square$

### 5.4 Gestione della Memoria e Retrieval Efficiente

Con $N = 300$ coppie, il database totale contiene al massimo $300 \times 972 \approx 291K$ vettori da 416 dimensioni — circa 485MB in fp32, 242MB in fp16. Questo è gestibile in RAM ma non sempre in VRAM.

**Strategia:** il database viene mantenuto in RAM (CPU) e trasferito in VRAM solo durante il retrieval. Per retrieval efficiente si usa **FAISS** (Facebook AI Similarity Search) con indice IVF-PQ per cluster, riducendo il costo del retrieval da $O(N \cdot N_{new})$ a $O(\sqrt{N} \cdot N_{new})$ con perdita trascurabile di qualità.

**Top-M retrieval:** invece di usare tutte le $N_k$ coppie nel cluster, si recuperano solo le top-$M=10$ più simili globalmente (usando $s_i$) e si applica l'attention solo su quelle. Questo riduce drasticamente il costo computazionale:

$$\text{FLOPs}_{retrieval} = O(M \cdot N_{new} \cdot d_r) = O(10 \times 972 \times 256) \approx 2.5M \text{ per immagine}$$

trascurabile rispetto al resto del forward pass.

---

## 6. Componente 4 — Bilateral Grid Renderer

### 6.1 Motivazione del Design

Il retrieved edit $\mathbf{R}_{spatial} \in \mathbb{R}^{B \times d_r \times H_{14} \times W_{14}}$ è una rappresentazione densa di "cosa cambiare" per ogni zona dell'immagine, espressa nello spazio delle feature DINOv2. Non è direttamente applicabile ai pixel RGB. La bilateral grid fa la conversione da feature space a pixel space in modo:

- **Edge-aware**: pixel con caratteristiche simili ricevono trasformazioni simili
- **Resolution-agnostic**: funziona a qualsiasi risoluzione tramite coordinate normalizzate
- **Differenziabile**: permette il gradient flow end-to-end

### 6.2 GridNet: da Retrieved Edit a Coefficienti

**Architettura:**

Il GridNet è un decoder leggero che produce i coefficienti delle due bilateral grid (globale e locale) a partire dal retrieved edit e dalle feature DINOv2:

**Input fusion:**

$$\mathbf{F}_{grid} = \text{Conv}_{1\times1}\!\left([\mathbf{R}_{spatial};\; \text{proj}(\mathbf{F}_{sem})]\right) \in \mathbb{R}^{B \times 256 \times H_{14} \times W_{14}}$$

dove $\text{proj}: \mathbb{R}^{384} \to \mathbb{R}^{128}$ è una proiezione lineare trainable che riduce la dimensionalità delle feature DINOv2.

**Global Branch:**

$$\mathbf{f}_{global} = \text{GAP}(\mathbf{F}_{grid}) \in \mathbb{R}^{B \times 256}$$

$$G_{global} = \text{reshape}\!\left(\mathbf{W}_{gb,2}\, \text{ReLU}(\mathbf{W}_{gb,1}\, \mathbf{f}_{global})\right) \in \mathbb{R}^{B \times 12 \times 8 \times 8 \times L_b}$$

con $L_b = 8$ bin di luminanza.

**Local Branch:**

$$G_{local} = \text{reshape}\!\left(\text{Conv}_{1\times1}\!\left(\text{AdaptiveAvgPool}(\mathbf{F}_{grid},\; (16,16))\right)\right) \in \mathbb{R}^{B \times 12 \times 16 \times 16 \times L_b}$$

**Inizializzazione all'identità:**

Entrambi i layer finali sono inizializzati per produrre la trasformazione identità:

```python
nn.init.zeros_(global_branch.last_layer.weight)
identity_bias = torch.tensor([1,0,0, 0,1,0, 0,0,1, 0,0,0], dtype=torch.float32)
global_branch.last_layer.bias.data = identity_bias.repeat(8 * 8 * L_b)
```

### 6.3 Guida Semantica per Bilateral Slicing

**Innovazione rispetto a HDRNet.** La guida classica di HDRNet usa solo la luminanza $g_{lum} = 0.299R + 0.587G + 0.114B$. Questo ha un limite: due pixel con la stessa luminanza ma contenuto semantico diverso (pelle chiara vs muro bianco) ricevono la stessa trasformazione.

RAG-ColorNet usa una **guida ibrida cromatica-semantica**:

$$g(i,j) = \alpha_g \cdot g_{chroma}(i,j) + (1 - \alpha_g) \cdot g_{sem}(i,j)$$

dove:

$$g_{chroma}(i,j) = \frac{0.5 \cdot L^*(i,j) + 0.25 \cdot |a^*(i,j)| + 0.25 \cdot |b^*(i,j)|}{114.0}$$

$$g_{sem}(i,j) = \sigma\!\left(\text{MLP}_{guide}(\mathbf{f}_{sem}(n_{ij}))\right) \in [0,1]$$

con $\text{MLP}_{guide}: \mathbb{R}^{384} \to \mathbb{R}^1$ (2 layer, ~25K parametri trainable) e $n_{ij}$ l'indice del patch DINOv2 che contiene il pixel $(i,j)$.

Il parametro $\alpha_g = 0.5$ è fisso. $g_{sem}$ produce valori diversi per pelle e sfondo anche a parità di luminanza — discriminando cromaticamente zone semanticamente diverse.

### 6.4 Trilinear Slicing

Per ogni pixel $(i,j)$ con guida $g(i,j) \in [0,1]$:

$$x_g^{glob}(j) = \frac{j}{W-1} \cdot 7, \quad y_g^{glob}(i) = \frac{i}{H-1} \cdot 7, \quad l_g(i,j) = g(i,j) \cdot 7$$

$$[\mathbf{A}^{glob}_{ij}, \mathbf{b}^{glob}_{ij}] = \text{TrilinearInterp}(G_{global},\; x_g^{glob},\; y_g^{glob},\; l_g)$$

$$I_{global}(i,j) = \mathbf{A}^{glob}_{ij} \cdot I_{src}(i,j) + \mathbf{b}^{glob}_{ij}$$

Analogamente per $G_{local}$ con coordinate $x_g^{loc}(j) = \frac{j}{W-1} \cdot 15$.

---

## 7. Componente 5 — Confidence Mask

### 7.1 Architettura

La confidence mask $\alpha \in [0,1]^{B \times 1 \times H \times W}$ determina pixel per pixel il peso relativo del ramo locale vs globale.

**Input:**
- Feature DINOv2 upsampled: $\mathbf{F}_{sem}^{up} \in \mathbb{R}^{B \times 128 \times H/4 \times W/4}$ (bilinear upsampling + proiezione)
- Divergenza tra le due trasformazioni: $\mathbf{D} = |I_{local} - I_{global}| \in \mathbb{R}^{B \times 3 \times H \times W}$, downsampled a $H/4 \times W/4$

$$\mathbf{z}_{mask} = \text{ReLU}\!\left(\text{Conv}_{3\times3}([\mathbf{F}_{sem}^{up};\; \mathbf{D}_{down}])\right) \in \mathbb{R}^{B \times 64 \times H/4 \times W/4}$$

$$\alpha_{low} = \sigma\!\left(\text{Conv}_{1\times1}(\mathbf{z}_{mask})\right) \in [0,1]^{B \times 1 \times H/4 \times W/4}$$

$$\alpha = \text{BilinearUp}(\alpha_{low},\; (H, W))$$

### 7.2 Blending Finale

$$I_{pred} = \alpha \odot I_{local} + (1 - \alpha) \odot I_{global}$$

$$I_{out} = \gamma_{sRGB}\!\left(\text{clip}(I_{pred}, 0, 1)\right)$$

---

## 8. Forward Pass Completo

### 8.1 Propagazione delle Dimensioni

| Stage | Output | Shape | Note |
|---|---|---|---|
| Input | $I_{src}$ | $(B, 3, H, W)$ | fp16 |
| DINOv2 | $F_{sem}$ | $(B, N, 384)$ | $N = \lfloor H/14\rfloor \cdot \lfloor W/14\rfloor$ |
| Color Histogram | $\mathbf{h}$ | $(B, 192)$ | fp32 |
| ClusterNet | $p$ | $(B, K)$ | simplex |
| Descriptor | $\mathbf{Q}$ | $(B, N, 416)$ | sem + chroma |
| Retrieval | $\mathbf{R}_{spatial}$ | $(B, 256, H_{14}, W_{14})$ | |
| GridNet Global | $G_{global}$ | $(B, 12, 8, 8, 8)$ | |
| GridNet Local | $G_{local}$ | $(B, 12, 16, 16, 8)$ | |
| Guida semantica | $g$ | $(B, H, W)$ | $\in [0,1]$ |
| Bilateral Slicing | $I_{global}, I_{local}$ | $(B, 3, H, W)$ | |
| Confidence Mask | $\alpha$ | $(B, 1, H, W)$ | $\in [0,1]$ |
| Output | $I_{out}$ | $(B, 3, H, W)$ | sRGB $\in [0,1]$ |

### 8.2 Pseudocodice

```python
def forward(I_src, database, cluster_net, grid_net, mask_net):
    # Step 1: Feature extraction
    F_sem = dinov2(I_src)                          # (B, N, 384) — frozen
    h     = color_histogram_lab(I_src)             # (B, 192)
    
    # Step 2: Cluster assignment
    p     = cluster_net(h)                         # (B, K)
    
    # Step 3: Build query descriptors
    c_patch = chromatic_patch_features(I_src)      # (B, N, 32)
    Q       = layernorm(concat(F_sem, c_patch))    # (B, N, 416)
    
    # Step 4: Local retrieval
    R_final = zeros(B, N, 256)
    for k in range(K):
        if p[:, k].max() < threshold: continue    # skip cluster irrilevanti
        R_k = cross_image_local_attention(
            Q, database[k].keys, database[k].values, top_M=10
        )                                          # (B, N, 256)
        R_final += p[:, k:k+1].unsqueeze(-1) * R_k
    
    R_spatial = reshape_to_spatial(R_final)        # (B, 256, H14, W14)
    
    # Step 5: Grid prediction
    G_global, G_local = grid_net(R_spatial, F_sem) # (B,12,8,8,8), (B,12,16,16,8)
    
    # Step 6: Semantic guide
    g = compute_semantic_guide(I_src, F_sem)       # (B, H, W)
    
    # Step 7: Bilateral slicing
    I_global = bilateral_slice(G_global, I_src, g) # (B, 3, H, W)
    I_local  = bilateral_slice(G_local,  I_src, g) # (B, 3, H, W)
    
    # Step 8: Confidence mask
    D     = (I_local - I_global).abs()
    alpha = mask_net(F_sem, D)                     # (B, 1, H, W)
    
    # Step 9: Blending + gamma
    I_pred = alpha * I_local + (1 - alpha) * I_global
    I_out  = gamma_srgb(I_pred.clamp(0, 1))
    
    return I_out, alpha, G_global, G_local, p
```

---

## 9. Funzione di Loss Composita

### 9.1 Panoramica

$$\mathcal{L} = \lambda_{\Delta E}\,\mathcal{L}_{\Delta E} + \lambda_{L1Lab}\,\mathcal{L}_{L1Lab} + \lambda_{hist}\,\mathcal{L}_{hist} + \lambda_{perc}\,\mathcal{L}_{perc} + \lambda_{chroma}\,\mathcal{L}_{chroma} + \lambda_{cluster}\,\mathcal{L}_{cluster} + \lambda_{retrieval}\,\mathcal{L}_{retrieval} + \lambda_{TV}\,\mathcal{L}_{TV} + \lambda_{lum}\,\mathcal{L}_{lum} + \lambda_{entropy}\,\mathcal{L}_{entropy}$$

### 9.2 Loss Cromatiche (invariate dalla tesi originale)

**ΔE Loss (CIEDE2000):**
$$\mathcal{L}_{\Delta E} = \frac{1}{HW}\sum_{i,j} \Delta E_{00}(I^{pred}_{Lab}(i,j),\; I^{tgt}_{Lab}(i,j))$$

**L1 Lab (warm-up):**
$$\mathcal{L}_{L1Lab} = \frac{1}{HW}\sum_{i,j} \|I^{pred}_{Lab}(i,j) - I^{tgt}_{Lab}(i,j)\|_1$$

**Color Histogram (EMD):**
$$\mathcal{L}_{hist} = \frac{1}{3}\sum_{c} \sum_k |\text{CDF}_c^{pred}(k) - \text{CDF}_c^{tgt}(k)|$$

**Perceptual (DINOv2 invece di VGG16):**
$$\mathcal{L}_{perc} = \frac{1}{N \cdot d_{dino}} \|\text{DINOv2}(I^{pred}) - \text{DINOv2}(I^{tgt})\|_F^2$$

Si usa DINOv2 già estratto nel forward pass — nessun costo aggiuntivo, feature semanticamente più ricche di VGG16.

**Chroma Consistency:**
$$\mathcal{L}_{chroma} = \mathcal{L}_{sat} + 0.5 \cdot \mathcal{L}_{hue}$$

**Luminance Preservation** (confronto con $I^{src}$, non $I^{tgt}$):
$$\mathcal{L}_{lum} = \frac{1}{HW}\sum_{i,j} |L^{*,pred}(i,j) - L^{*,src}(i,j)|$$

### 9.3 Loss Nuove

**Cluster Assignment Loss:**

Supervisiona che il ClusterNet assegni correttamente le foto ai cluster inizializzati con K-Means:

$$\mathcal{L}_{cluster} = -\sum_k z_k \log(p_k + \varepsilon)$$

dove $z_k \in \{0,1\}$ è l'assegnazione hard determinata da K-Means durante il preprocessing. Questa loss è attiva **solo nelle prime 5 epoche** — dopo di che il ClusterNet è sufficientemente stabile e lasciare questa loss attiva potrebbe impedire raffinamenti utili.

**Retrieval Quality Loss:**

Supervisiona che il retrieval trovi le edit signatures giuste. Per ogni coppia di training $(I_i^{src}, I_i^{tgt})$, il retrieved edit usando tutte le altre coppie come database deve approssimare la vera edit signature:

$$\mathcal{L}_{retrieval} = \frac{1}{N} \sum_i \frac{1}{N_i} \sum_n \|\mathbf{R}_i(n) - \mathbf{E}_i(n)\|^2$$

dove $\mathbf{R}_i(n)$ è il retrieved edit per la patch $n$ dell'immagine $i$ usando $\mathcal{D}_\phi \setminus \{i\}$ come database (leave-one-out). Questa loss è il cuore dell'apprendimento: forza le proiezioni $\mathbf{W}^Q, \mathbf{W}^K, \mathbf{W}^V$ a produrre spazi di feature in cui zone simili hanno edit simili.

**Total Variation sulla Bilateral Grid:**
$$\mathcal{L}_{TV} = \frac{1}{|\mathcal{G}|}\sum_{i,j} \left(\|\mathbf{A}_{i+1,j} - \mathbf{A}_{ij}\|_F + \|\mathbf{A}_{i,j+1} - \mathbf{A}_{ij}\|_F\right)$$

applicata a entrambe le grid $G_{global}$ e $G_{local}$.

**Entropy Loss sulla Confidence Mask:**
$$\mathcal{L}_{entropy} = -\frac{1}{HW}\sum_{i,j} [\alpha_{ij}\log(\alpha_{ij} + \varepsilon) + (1-\alpha_{ij})\log(1-\alpha_{ij}+\varepsilon)]$$

### 9.4 Curriculum dei Pesi

| Epoca | $\lambda_{\Delta E}$ | $\lambda_{L1Lab}$ | $\lambda_{hist}$ | $\lambda_{perc}$ | $\lambda_{chroma}$ | $\lambda_{cluster}$ | $\lambda_{retrieval}$ | $\lambda_{TV}$ | $\lambda_{lum}$ | $\lambda_{entropy}$ |
|---|---|---|---|---|---|---|---|---|---|---|
| 1–5 | 0.0 | 0.8 | 0.4 | 0.0 | 0.0 | 0.3 | 0.5 | 0.01 | 0.3 | 0.01 |
| 6–10 | 0.3 | 0.4 | 0.3 | 0.3 | 0.1 | 0.0 | 0.3 | 0.01 | 0.3 | 0.01 |
| 11+ | 0.5 | 0.0 | 0.3 | 0.6 | 0.2 | 0.0 | 0.2 | 0.01 | 0.3 | 0.01 |

**Motivazione del curriculum:**

Epoche 1–5: si stabilizzano contemporaneamente il ClusterNet ($\mathcal{L}_{cluster}$) e le proiezioni di retrieval ($\mathcal{L}_{retrieval}$), con $\mathcal{L}_{L1Lab}$ dominante per la correttezza cromatica. $\mathcal{L}_{lum}$ attiva dall'inizio per preservare la struttura.

Epoche 6–10: $\mathcal{L}_{cluster}$ si azzera (cluster ormai stabili), si introduce $\mathcal{L}_{perc}$ e si riduce $\mathcal{L}_{retrieval}$. Transizione da L1Lab a ΔE.

Epoche 11+: regime stabile. $\mathcal{L}_{retrieval}$ rimane con peso ridotto per prevenire deriva delle proiezioni.

---

## 10. Training: Tre Fasi

### 10.1 Panoramica

| Fase | Dataset | Obiettivo | Durata stimata (RTX 3080) |
|---|---|---|---|
| **Pre-training** | FiveK + PPR10K + Lightroom presets | Imparare feature di retrieval e rendering cromatico | ~10h |
| **Meta-training Reptile** | Per-photographer tasks + sintetici | $\theta_{meta}$: inizializzazione per adattamento rapido | ~15h |
| **Few-shot Adaptation** | N coppie del fotografo target | Costruire database, raffinare ClusterNet e proiezioni | ~1-2h |

### 10.2 Fase 1: Pre-Training

**Dataset:** unione di FiveK (5 fotografi × 1000 coppie), PPR10K (10K ritratti), e coppie generate da 500 preset Lightroom pubblici applicati a 200 immagini base = ~100K coppie totali.

**Obiettivo:** addestrare le proiezioni $\mathbf{W}^Q, \mathbf{W}^K, \mathbf{W}^V$, il GridNet, il MaskNet, e $\text{MLP}_{guide}$ a fare retrieval e rendering su una distribuzione ampia di stili fotografici.

**Loss semplificata:**

$$\mathcal{L}_{pre} = \mathcal{L}_{\Delta E} + 0.5\,\mathcal{L}_{perc} + 0.3\,\mathcal{L}_{retrieval}$$

**Ottimizzazione:** AdamW, $\eta = 10^{-4}$, batch size $B = 4$, risoluzione $512 \times 384$, fp16 con GradScaler.

**Parti trainable in questa fase:**
- Proiezioni retrieval $\mathbf{W}^Q, \mathbf{W}^K, \mathbf{W}^V$: ✅
- GridNet (Global + Local Branch): ✅
- MaskNet: ✅
- $\text{MLP}_{guide}$: ✅
- ClusterNet: ✅ (pre-training generico, poi reinizializzato per ogni fotografo)
- DINOv2: ❌ frozen sempre

### 10.3 Fase 2: Meta-Training Reptile

**Algoritmo:** Reptile (Nichol et al., 2018) — primo ordine, compatibile fp16.

**Task:** ogni task $\mathcal{T}$ è un fotografo con support set $\mathcal{D}^{sup}$ (15 coppie) e query set $\mathcal{D}^{qry}$ (5 coppie). I task sintetici sono generati interpolando stili di fotografi diversi in Lab space.

**Inner loop** ($k = 5$ passi, $\alpha = 10^{-3}$):

$$\theta_\mathcal{T}^{(t+1)} = \theta_\mathcal{T}^{(t)} - \alpha\,\nabla_{\theta}\mathcal{L}_\mathcal{T}^{sup}(\theta_\mathcal{T}^{(t)})$$

**Outer loop** ($M = 2$ task per batch, $\epsilon = 10^{-2}$):

$$\theta \leftarrow \theta + \frac{\epsilon}{M}\sum_{m=1}^{M}(\tilde{\theta}_{\mathcal{T}_m} - \theta)$$

**Parti trainable nel meta-training:**
- Proiezioni retrieval: ✅ (meta-aggiornate)
- GridNet: ✅ (meta-aggiornato)
- MaskNet: ✅ (meta-aggiornato)
- $\text{MLP}_{guide}$: ✅ (meta-aggiornato)
- ClusterNet: ❌ (reinizializzato per ogni fotografo in fase 3)
- DINOv2: ❌ frozen sempre

### 10.4 Fase 3: Few-Shot Adaptation

**Input:** $\theta_{meta}$ da fase 2 + N coppie del fotografo target.

**Step 0 — Preprocessing del database (nessun training):**

```
Per ogni coppia (src_i, tgt_i):
    1. Calcola F_sem_i = DINOv2(src_i)            → cache
    2. Calcola E_i = DINOv2(tgt_i) - DINOv2(src_i) → cache
    3. Calcola h_i = color_histogram_lab(src_i)   → cache
    4. Calcola q_i = build_query_descriptor(F_sem_i, src_i) → cache

K* = elbow_criterion(KMeans(h_i))
Inizializza ClusterNet con centroidi K-Means
Assegna ogni coppia al cluster: z_i = argmin_k ||h_i - C_k||
Costruisce database per cluster: M^(k) = {(q_i, E_i) : z_i = k}
```

**Step 1 — Adaptation parziale (epoche 1–10):**

Parametri frozen: DINOv2, primi layer del GridNet.
Parametri trainable: ClusterNet, $\mathbf{W}^Q, \mathbf{W}^K, \mathbf{W}^V$, ultimi layer GridNet, MaskNet, $\text{MLP}_{guide}$.

AdamW, $\eta_A = 5 \times 10^{-5}$, $\lambda_{wd} = 2 \times 10^{-3}$.

**Step 2 — Fine-tuning globale (epoche 11–30):**

Tutti i parametri trainable. $\eta_B = 2.5 \times 10^{-5}$, cosine annealing.

**Early stopping:** holdout 20% delle coppie, patience 5 epoche.

**Totale parametri trainable nel few-shot:**
- ClusterNet: ~100K
- Proiezioni retrieval: $3 \times 416 \times 256 \approx 320K$
- GridNet leggero: ~500K
- MaskNet: ~100K
- $\text{MLP}_{guide}$: ~25K
- **Totale: ~1.05M parametri**

Con N=300 coppie e risoluzione $512 \times 384$: rapporto dati/parametri $= 300 \times 512 \times 384 \times 3 / 1.05M \approx 168$ — molto favorevole per il few-shot.

---

## 11. Meta-Learning con Reptile

### 11.1 Perché Reptile è la Scelta Giusta

| Criterio | MAML | Reptile |
|---|---|---|
| Ordine gradienti | 2° (Hessiani) | 1° |
| Compatibilità fp16 | Instabile | ✅ Stabile |
| Memoria per iterazione | ~18GB | ~5GB |
| Tempo per iterazione (RTX 3080) | ~8s | ~2s |
| Qualità meta-inizializzazione | ✅✅ | ✅ |

Per RAG-ColorNet con DINOv2 frozen, la differenza di qualità tra MAML e Reptile è ulteriormente ridotta: la parte più difficile del meta-learning (imparare la rappresentazione della scena) è già risolta da DINOv2. Reptile deve solo meta-ottimizzare le proiezioni di retrieval e il GridNet, che sono moduli più semplici.

### 11.2 Task Augmentation Migliorata

Invece della sola interpolazione lineare in Lab space (che produce stili intermedi ma non nuovi), si usa una augmentation più ricca:

**Interpolazione di stile:** $I^{tgt,\lambda} = \mathcal{L}^{-1}(\lambda \mathcal{L}(I^{tgt,i}) + (1-\lambda)\mathcal{L}(I^{tgt,j}))$

**Perturbazione di cluster:** si scambia casualmente una frazione delle coppie tra cluster diversi dello stesso fotografo, forzando il modello a imparare assignment robusti.

**Cross-photographer retrieval:** si usa il database del fotografo $i$ per gradare le foto del fotografo $j$ — il retrieved edit sarà sbagliato, ma il modello impara a riconoscere quando il retrieval è inaffidabile (segnale utile per il MaskNet).

---

## 12. Few-Shot Adaptation

### 12.1 Il Comportamento Incrementale

La proprietà più importante di RAG-ColorNet è che il miglioramento con i dati è **non parametrico** — non richiede retraining. Aggiungere una nuova coppia $(I_{new}^{src}, I_{new}^{tgt})$ al database richiede:

```
1. Calcola F_sem_new = DINOv2(I_new^src)           → ~0.3s
2. Calcola E_new = DINOv2(I_new^tgt) - DINOv2(I_new^src)  → ~0.3s
3. Calcola h_new = color_histogram_lab(I_new^src)  → ~0.01s
4. Assegna al cluster: z_new = argmin_k ||h_new - C_k||
5. Aggiungi (q_new, E_new) a M^(z_new)
```

Tempo totale: ~0.6s. Zero retraining. Il prossimo grading userà automaticamente la nuova coppia nel retrieval.

### 12.2 Aggiornamento dei Cluster

Periodicamente (ogni 50 nuove coppie) si può eseguire un re-clustering online:

```
1. Ricalcola K* con elbow criterion su tutti gli h_i aggiornati
2. Se K* è cambiato: esegui KMeans e re-assegna le coppie
3. Fine-tune ClusterNet per 5 epoche sul nuovo assignment
   (solo ClusterNet, ~2 minuti)
```

Questo gestisce il caso in cui un fotografo sviluppa nuovi stili nel tempo.

---

## 13. Memoria Incrementale

### 13.1 Struttura della Memoria

```
Photographer Database:
├── cluster_0/
│   ├── keys:   (N_0, 416) float16  — query descriptors
│   ├── values: (N_0, 384) float16  — edit signatures
│   └── meta:   {filename, date, lighting_condition}
├── cluster_1/
│   └── ...
└── cluster_K/
    └── ...

Cluster centroids: (K, 192) float32
ClusterNet weights: ~100K params
```

### 13.2 Occupazione di Memoria

Con $N = 300$ coppie, $K = 8$ cluster, $N_{patches} = 972$ per immagine:

| Struttura | Dimensione | Memoria (fp16) |
|---|---|---|
| Keys (416 dim) | $300 \times 972 \times 416$ | ~235MB |
| Values (384 dim) | $300 \times 972 \times 384$ | ~217MB |
| Totale database | — | ~452MB RAM |
| In VRAM durante retrieval (top-M=10) | $10 \times 972 \times 800$ | ~15MB |

Il database vive in RAM. Solo le top-M coppie vengono spostate in VRAM durante il retrieval — costo VRAM trascurabile.

### 13.3 Scalabilità

Con $N = 3000$ coppie (uso professionale intensivo):

- Database RAM: ~4.5GB — gestibile con RAM moderna (16GB+)
- Retrieval con FAISS IVF-PQ: $O(\sqrt{N} \times N_{new})$ invece di $O(N \times N_{new})$
- Tempo retrieval: <0.1s per immagine a piena risoluzione

---

## 14. Dataset

### 14.1 Per il Pre-Training

| Dataset | Coppie | Caratteristiche | Uso |
|---|---|---|---|
| MIT-Adobe FiveK | 25K (5×5K) | 5 esperti, RAW/graded, vari generi | Training principale |
| PPR10K | 10K | Ritratti, human masks | Ritratti professionali |
| Lightroom Presets | ~100K sintetici | 500 preset × 200 immagini | Diversità stilistica |

### 14.2 Per il Meta-Training

Task reali: 5 fotografi FiveK + 3 fotografi PPR10K = 8 task reali.
Task sintetici: $\binom{8}{2} = 28$ coppie di fotografi × 10 valori di $\lambda$ = 280 task sintetici.

### 14.3 Per il Few-Shot

Il fotografo target fornisce N coppie (src, tgt) esportate direttamente dal suo software di editing. Formato: sRGB TIFF 16-bit o JPEG alta qualità.

**Requisiti minimi:**
- $N \geq 50$ con $\theta_{meta}$
- Diversità delle scene: copertura di tutti i contesti fotografici del fotografo
- Consistenza temporale: stesso "periodo stilistico"

---

## 15. Proprietà Matematiche e Teoremi

### Teorema 1: Invarianza alla Risoluzione

RAG-ColorNet è resolution-agnostic: per qualsiasi coppia di risoluzioni $(H_1, W_1)$ e $(H_2, W_2)$ con stesso aspect ratio, la predizione a risoluzione $H_1 \times W_1$ è uguale alla predizione a $H_2 \times W_2$ ricampionata a $H_1 \times W_1$, a meno di errori di ricampionamento $O(1/\min(H_1, H_2))$.

*Argomentazione:*
(a) DINOv2 con patch size 14 opera su patch locali — invariante alla risoluzione assoluta.
(b) Il color histogram è una statistica globale normalizzata — invariante.
(c) Le coordinate del bilateral slicing sono normalizzate in $[0,1]^2$ — invarianti.
(d) La confidence mask usa bilinear upsampling con coordinate normalizzate — invariante.
$\square$

### Teorema 2: Miglioramento Monotono

Sia $\epsilon_N = \mathbb{E}[\Delta E_{00}(f_\theta(I^{src}, \mathcal{D}_\phi^N), I^{tgt})]$ l'errore atteso con $N$ coppie nel database. Allora $\epsilon_{N+1} \leq \epsilon_N$ in aspettativa quando la nuova coppia è campionata dalla stessa distribuzione del fotografo.

*Argomentazione:* Aggiungere una coppia al database non può peggiorare il retrieval in aspettativa — il softmax assegnerà peso basso a coppie non pertinenti. La bilateral grid poi converte il retrieved edit (migliorato o invariato) in pixel. $\square$

### Teorema 3: Convergenza di Reptile

Sia $\mathcal{L}$ $L$-smooth e lower-bounded da $\mathcal{L}^*$. Con $\alpha < 1/L$ e $\epsilon = O(1/\sqrt{T})$, Reptile converge: $\min_{t \leq T} \|\nabla_\theta \bar{\mathcal{L}}(\theta_t)\| \leq O(1/\sqrt{T})$.

*Riferimento:* Nichol et al. (2018), Theorem 1. $\square$

### Teorema 4: Differenziabilità della Loss

La Color-Aesthetic Loss $\mathcal{L}$ è differenziabile quasi ovunque rispetto a $I^{pred}$ e rispetto ai parametri $\theta$.

*Argomentazione:* Tutti i termini della loss sono composizioni di operazioni differenziabili q.o. (radici quadrate con $\varepsilon$-smoothing, kernels gaussiani, bilinear interpolation, ReLU). Il retrieval con softmax è differenziabile ovunque. $\square$

### Teorema 5: Complessità del Retrieval

Con top-M retrieval e FAISS IVF-PQ, la complessità del retrieval è:

$$\text{FLOPs}_{retrieval} = O(M \cdot N_{new} \cdot d_r) = O(10 \times 972 \times 256) \approx 2.5M$$

indipendente da $N$ (dimensione del database). Il costo di costruzione dell'indice FAISS è $O(N \log N)$ ma viene eseguito una sola volta durante il preprocessing.

---

## 16. Risultati Attesi

### 16.1 Metriche Quantitative

| Metrica | Target | Baseline HDRNet | Atteso RAG-ColorNet |
|---|---|---|---|
| $\overline{\Delta E}_{00}$ ↓ | < 3.0 | 6.5 | **2.8–3.5** |
| SSIM $L^*$ ↑ | > 0.96 | 0.96 | **0.97–0.98** |
| LPIPS ↓ | < 0.08 | 0.12 | **0.06–0.09** |
| $\Delta\mu_{NIMA}$ ↑ | > 0.5 | 0.2 | **0.6–1.0** |

### 16.2 Scalabilità con N

| N coppie | $\overline{\Delta E}_{00}$ atteso | Note |
|---|---|---|
| 50 | 4.0–5.0 | Solo $\theta_{meta}$, retrieval sparso |
| 100 | 3.5–4.0 | Cluster emergono chiaramente |
| 300 | 2.8–3.5 | Target ottimale |
| 1000 | 2.0–2.8 | Match quasi-esatti per scene comuni |
| 5000 | 1.5–2.0 | Qualità professionale su generi coperti |

### 16.3 Scenari di Successo e Fallimento

**Successo (probabilità 70–80%):** Il fotografo ha uno stile consistente con cluster ben separati. Le 300 coppie coprono diversi contesti fotografici. Il retrieved edit converge a match di alta qualità.

**Successo parziale (probabilità 15–20%):** Il fotografo ha uno stile molto variabile o evoluto nel tempo. Il sistema produce output stilisticamente coerenti ma non indistinguibili dal grading manuale.

**Fallimento (probabilità 5–10%):** Training set troppo omogeneo (tutte le foto dello stesso tipo). Stile del fotografo troppo dipendente da editing locale fine (dodge & burn su sub-regioni minuscole) che la bilateral grid non riesce a catturare.

---

## 17. Analisi della Complessità Computazionale

### 17.1 Training

**Pre-training:**
- Dataset: ~135K coppie
- Batch size: 4, risoluzione $512 \times 384$, fp16
- Epoche: 50
- Tempo stimato: ~10h su RTX 3080

**Meta-training:**
- 10000 iterazioni Reptile
- 2 task per iterazione × 5 inner steps = 10 forward-backward pass per iterazione
- Tempo per iterazione: ~2s
- Tempo totale: ~5.5h su RTX 3080

**Few-shot adaptation:**
- Preprocessing database: $N \times 0.6s \approx 3$ minuti per N=300
- Training: 30 epoche × ~2 min/epoca = ~1h
- **Totale: ~1.1h**

### 17.2 Inferenza

| Componente | Tempo stimato (RTX 3080, fp16, $3000 \times 2000$) |
|---|---|
| DINOv2 forward | ~0.3s |
| Color histogram + ClusterNet | ~0.05s |
| Build query descriptors | ~0.05s |
| FAISS retrieval (top-10) | ~0.05s |
| Cross-image attention (M=10) | ~0.1s |
| GridNet | ~0.2s |
| Bilateral slicing (×2) | ~0.4s |
| Confidence mask + blending | ~0.2s |
| **Totale** | **~1.4s** ✅ |

Ampiamente entro il budget di 10s. Con TensorRT o int8 quantization: ~0.5s.

### 17.3 Occupazione VRAM durante il Training

| Componente | VRAM (fp16, batch=4, $512\times384$) |
|---|---|
| DINOv2-Small (frozen) | ~42MB |
| Activations forward pass | ~2.5GB |
| Gradienti | ~1.5GB |
| AdamW optimizer states | ~2GB |
| Database retrieval (top-10 in VRAM) | ~15MB |
| **Totale** | **~6.1GB** ✅ |

Entro i 10GB di RTX 3080, con ~4GB di margine per picchi.

---

## 18. Ablation Studies

| ID | Configurazione | Misura il contributo di |
|---|---|---|
| **A0** | HDRNet generico (nessuna personalizzazione) | Lower bound assoluto |
| **A1** | RAG-ColorNet senza retrieval (style prototype globale) | Retrieval locale |
| **A2** | RAG-ColorNet senza cluster (un solo cluster) | Cluster assignment |
| **A3** | RAG-ColorNet con guida cromatica pura (senza $g_{sem}$) | Guida semantica |
| **A4** | RAG-ColorNet con guida semantica pura (senza $g_{chroma}$) | Guida cromatica |
| **A5** | RAG-ColorNet senza meta-learning (random init) | Meta-training Reptile |
| **A6** | RAG-ColorNet con VGG16 invece di DINOv2 | Qualità feature semantiche |
| **A7** | RAG-ColorNet con N=50 coppie | Sample efficiency |
| **A8** | RAG-ColorNet con N=300 coppie, senza aggiornamento incrementale | Memoria incrementale |
| **A9** | RAG-ColorNet completo, N=300 | — |
| **A10** | RAG-ColorNet completo, N=300+200 incrementali | Miglioramento incrementale |

**Expected ranking** per $\overline{\Delta E}_{00}$ ↓:
$$A0 > A5 > A2 > A1 > A6 > A3 \approx A4 > A7 > A8 > A9 > A10$$

Il gap più grande atteso è tra A1 (senza retrieval) e A9 (completo): il retrieval locale è il contributo principale. Il gap tra A8 e A10 dimostra il miglioramento incrementale — il risultato più importante per un prodotto.

---

## 19. Confronto con lo Stato dell'Arte

| Metodo | End-to-End | Few-Shot | Photographer-Specific | Incrementale | $\overline{\Delta E}_{00}$ | Tempo inferenza |
|---|---|---|---|---|---|---|
| HDRNet (2017) | ✅ | ❌ | ❌ | ❌ | ~6.5 | 15ms |
| Deep Preset (2020) | ✅ | ❌ | ⚠️ | ❌ | ~5.0 | 50ms |
| CSRNet (2020) | ✅ | ❌ | ⚠️ | ❌ | ~5.2 | 15ms |
| PromptIR (2023) | ✅ | ✅ | ⚠️ | ❌ | ~4.5 | 200ms |
| Imagen AI (2024) | ✅ | ❌ (3K+) | ✅✅ | ✅ | ~3.0 | 330ms |
| **RAG-ColorNet (ours)** | ✅ | ✅✅ | ✅✅ | ✅✅ | **~3.0** | ~1.4s |

**Note sul confronto:**
- Imagen AI è il riferimento commerciale: richiede 3000+ immagini, closed-source, subscription.
- RAG-ColorNet mira allo stesso livello qualitativo con 300 immagini, open-source, su GPU consumer.
- Il tempo di inferenza è più alto di HDRNet perché include DINOv2 e retrieval — accettabile per uso professionale.

---

*Fine del documento. Versione 1.0 — soggetta a revisione dopo feedback.*
