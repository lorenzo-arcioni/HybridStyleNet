## 5. Architetture Proposte

### 5.1 Architettura Principale: HybridStyleNet

#### Motivazione del Design

L'architettura proposta nasce da un'analisi critica del task: un fotografo professionista **guarda l'intera scena** prima di decidere come gradare ogni zona. Un tramonto implica toni caldi ovunque, anche nel soggetto in basso. Un interno nuvoloso implica raffreddamento generale anche nelle ombre. Questa dipendenza contestuale globale è il limite fondamentale di una CNN pura, che per design processa l'immagine localmente.

La soluzione non è però un Vision Transformer puro, che su immagini ad alta risoluzione avrebbe complessità $O(n^2)$ proibitiva e richiederebbe ordini di grandezza più dati di training di quanti ne siano disponibili nel regime few-shot. La soluzione corretta è un **encoder ibrido CNN + Swin Transformer**, dove:

- La **CNN** (MobileNetV3-Small, stage 1-3) estrae feature locali con inductive bias fotografico forte: texture, bordi, skin tone, erba, cielo — dove la convoluzione è imbattibile ed efficiente
- Lo **Swin Transformer** (stage 4-5) ragiona sulle relazioni globali tra regioni: "questo è un tramonto → le regioni in basso devono ricevere toni caldi coerenti con il cielo"

Tutto il modello è addestrato **end-to-end** su coppie $(I^{src}, I^{tgt})$ senza nessun parametro di editing esplicito. Il training utilizza **fp16 mixed precision** in tutte le fasi, riducendo il consumo di memoria GPU di circa il 40% rispetto a fp32, con perdita trascurabile di accuratezza numerica.

---

#### Schema Generale

```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 TRAINING TIME: costruzione del Style Prototype
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

{(src_i, tgt_i)}_{i=1}^N   (100-200 coppie del fotografo)
        │
        ▼
  edit_delta_i = Enc(tgt_i) - Enc(src_i)   per ogni coppia
        │
        ▼
┌───────────────────────────────────────────┐
│  SET TRANSFORMER (permutation-invariant)  │
│                                           │
│  Self-attention tra le N edit_delta_i     │
│  4 teste di attenzione                    │
│  → pesa coppie "tipiche" più degli outlier│
│  → output: s ∈ ℝ^256 (style prototype)   │
└───────────────────────────────────────────┘
        │
        │  s viene memorizzato e riusato a ogni inferenza
        ▼

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 TEST TIME: inferenza su nuova immagine
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

RAW H_raw × W_raw
        │
        │  Pipeline §6.2: linearize → demosaic → lens corr. → WB
        │  → sRGB → gamma → downsample (s = L_train/max(H_raw,W_raw))
        ▼
Input: I^src ∈ [0,1]^{H×W×3}  (risoluzione variabile)
        │
        ▼
┌───────────────────────────────────────────────────────┐
│  CNN STEM: MobileNetV3-Small stage 1-3  (fp16)        │
│                                                       │
│  Stage 1: stride 2 → [B, 16,  H₁, W₁]               │
│  Stage 2: stride 2 → [B, 24,  H₂, W₂]               │
│  Stage 3: stride 2 → [B, 48,  H₃, W₃]  ← P3         │
│                                                       │
│  Hₖ = ⌊H/2ᵏ⌋,  Wₖ = ⌊W/2ᵏ⌋                        │
│  Inductive bias locale forte: texture, bordi,         │
│  skin tone, struttura fine                            │
└───────────────────────────────────────────────────────┘
        │  P3: [B, 48, H₃, W₃]
        ▼
┌───────────────────────────────────────────────────────┐
│  SWIN TRANSFORMER stage 4-5 (con RoPE)  (fp16)        │
│                                                       │
│  Stage 4: Window attention M×M + Shifted windows      │
│           → [B, 96,  H₄, W₄]  ← P4                   │
│                                                       │
│  Stage 5: Window attention M×M + Shifted windows      │
│           → [B, 192, H₅, W₅]  ← P5                   │
│                                                       │
│  T(H,W) = H₅·W₅ token (varia con la risoluzione)     │
│  RoPE → generalizza a qualsiasi (H,W) senza retraining│
│                                                       │
│  Ogni token in P5 → regione (H/H₅)×(W/W₅) px         │
└───────────────────────────────────────────────────────┘
        │
        ▼
┌───────────────────────────────────────────────────────┐
│  CROSS-ATTENTION con Style Prototype  (fp16)          │
│                                                       │
│  Query  = P5 features (nuova immagine)                │
│  Keys   = {Enc(src_i)} dal training set (K=20)        │
│  Values = {edit_delta_i} dal training set (K=20)      │
│                                                       │
│  "Quale edit del training set è più rilevante         │
│   per QUESTA specifica immagine?"                     │
│                                                       │
│  → AdaIN conditioning con s su tutti i branch        │
└───────────────────────────────────────────────────────┘
        │
        ├──────────────────────────┐
        ▼                          ▼
┌────────────────────┐   ┌──────────────────────────┐
│  GLOBAL BRANCH     │   │  LOCAL BRANCH            │
│  (P5 + s)          │   │  (P3, P4 + s)            │
│                    │   │                          │
│  Bilateral Grid    │   │  Bilateral Grid          │
│  Coarse: 8×8×8     │   │  Compromise: 16×16×8     │
│                    │   │  canali SPADE: 128        │
│  AdaIN cond.       │   │                          │
│  (global color     │   │  SPADE cond.             │
│  mood, WB, cast)   │   │  (spatial-aware:         │
│                    │   │  skin, sky, shadow)      │
└────────────────────┘   └──────────────────────────┘
        │                          │
        └────────────┬─────────────┘
                     ▼
        ┌────────────────────────┐
        │  CONFIDENCE MASK       │
        │  α(x,y) = σ(MaskNet(P3,P4))  │
        │  learned, spatial      │
        └────────────────────────┘
                     │
                     ▼
     I_out = α⊙I_local + (1-α)⊙I_global
                     │
                     ▼
              I_graded ✓  (H × W, risoluzione piena)
```

---

#### Componenti Dettagliati

##### **1. CNN Stem: MobileNetV3-Small Stage 1-3**

MobileNetV3-Small è scelto come encoder leggero per tre ragioni fondamentali nel contesto few-shot e del budget computazionale su GPU consumer. In primo luogo i **blocchi depthwise separable con SE**: ogni blocco applica una convoluzione depthwise separable seguita da un modulo Squeeze-and-Excitation con funzione di attivazione Hard-Swish, catturando relazioni inter-canale cromatiche fondamentali per il color grading con un numero di parametri nettamente inferiore a EfficientNet-B4. In secondo luogo la **leggerezza del modello**: MobileNetV3-Small ha circa 2.5M di parametri contro i 19M di EfficientNet-B4, riducendo il rischio di overfitting nel regime few-shot e il consumo di memoria con **fp16 mixed precision** su RTX 3080. In terzo luogo il **pre-training ImageNet**: il modello sa già riconoscere semanticamente la scena prima del training sul fotografo — cruciale nel regime few-shot dove non ci sono abbastanza coppie per imparare tali rappresentazioni da zero.

La trasformazione matematica dei tre stage è la seguente. Sia $I_{src} \in [0,1]^{H \times W \times 3}$ con $(H, W)$ arbitrari. Il CNN stem applica una sequenza di blocchi Bottleneck Inverted Residual (bneck) $\mathcal{F}^{(k)}$ con stride 2 ad ogni stage:

$$P_3 = \mathcal{F}^{(3)}\!\left(\mathcal{F}^{(2)}\!\left(\mathcal{F}^{(1)}(I_{src})\right)\right) \in \mathbb{R}^{B \times 48 \times H_3 \times W_3}$$

**Blocco bneck con SE e Hard-Swish.** Sia $\mathbf{x} \in \mathbb{R}^{C_{in} \times H' \times W'}$ l'input al blocco con expansion ratio $e$ (tipicamente $e = 4$–$6$ in MobileNetV3-Small). Il blocco è:

$$\mathbf{x}_{exp} = \text{BN}(\delta_{hs}(\text{Conv}_{1\times1}^{C_{in} \to eC_{in}}(\mathbf{x})))$$

$$\mathbf{x}_{dw} = \text{BN}(\delta_{hs}(\text{DWConv}_{k\times k}^{eC_{in}}(\mathbf{x}_{exp})))$$

dove $\text{DWConv}_{k\times k}^C$ è la depthwise convolution con kernel $k \times k$ su $C$ canali separatamente:

$$(\text{DWConv}^C(\mathbf{x}))_c = \mathbf{w}_c * \mathbf{x}_c, \quad c = 1,\ldots,C$$

Il modulo **Squeeze-and-Excitation** su $\mathbf{x}_{dw}$:

$$\mathbf{z} = \frac{1}{H'W'}\sum_{i,j}\mathbf{x}_{dw}(\cdot,i,j) \in \mathbb{R}^{eC_{in}} \quad \text{(squeeze: global average pool)}$$

$$\mathbf{e} = \sigma\!\left(\mathbf{W}_2\,\delta_{re}(\mathbf{W}_1\,\mathbf{z})\right) \in (0,1)^{eC_{in}}, \quad \mathbf{W}_1 \in \mathbb{R}^{\lfloor eC_{in}/4\rfloor \times eC_{in}},\ \mathbf{W}_2 \in \mathbb{R}^{eC_{in} \times \lfloor eC_{in}/4\rfloor}$$

con reduction ratio $r = 4$. Il **excitation** ri-pondera i canali:

$$\mathbf{x}_{se} = \mathbf{e} \odot \mathbf{x}_{dw}$$

Proiezione finale e **residual connection** (solo se stride $= 1$ e $C_{in} = C_{out}$):

$$\mathbf{x}_{proj} = \text{BN}(\text{Conv}_{1\times1}^{eC_{in} \to C_{out}}(\mathbf{x}_{se}))$$

$$\mathcal{F}^{(k)}(\mathbf{x}) = \begin{cases} \mathbf{x} + \mathbf{x}_{proj} & \text{se stride} = 1 \text{ e } C_{in} = C_{out} \\ \mathbf{x}_{proj} & \text{altrimenti} \end{cases}$$

Le funzioni di attivazione sono: **Hard-Swish** $\delta_{hs}(x) = x \cdot \text{ReLU6}(x+3)/6$ nei layer più profondi, e **ReLU** $\delta_{re}(x) = \max(0,x)$ nel modulo SE interno — esattamente come specificato nell'architettura MobileNetV3-Small originale.

**Proprietà fondamentale (Inductive Bias Locale):** Le convoluzioni hanno receptive field limitato. Al layer $l$ con kernel $k \times k$ e stride $s$, il receptive field effettivo cresce come $r_l = r_{l-1} + (k-1) \cdot \prod_{j < l} s_j$. Nei primi 3 stage di MobileNetV3-Small, con stride complessivo $2^3 = 8$, ogni feature in $P_3$ vede una regione di circa $20$–$30$ pixel dell'immagine originale. Questo è sufficiente per catturare texture locali, bordi e pattern cromatici fini, ma insufficiente per ragionare su dipendenze globali ("cielo arancione → soggetto caldo"). Questa limitazione motiva il secondo modulo.

**Strategia di congelamento nel few-shot adaptation**: i parametri degli stage 1 e 2 di $\mathcal{F}^{(1)}, \mathcal{F}^{(2)}$ vengono congelati nelle prime 10 epoche della fase di adattamento. Formalmente, si partiziona il parametro space $\Theta = \Theta_{frozen} \cup \Theta_{adapt}$ dove $\Theta_{frozen} = \{\theta_1, \theta_2\}$ e il gradiente viene bloccato: $\nabla_{\Theta_{frozen}} \mathcal{L} := 0$. Questo previene il catastrophic forgetting delle rappresentazioni di basso livello apprese su ImageNet.

**Nota su fp16:** Le operazioni di convoluzione e BN nel CNN stem girano in fp16; le batch norm accumulano le statistiche in fp32 internamente (come da standard PyTorch AMP) per evitare problemi di underflow nel running mean/variance.

---

##### **2. Swin Transformer Stage 4-5 con RoPE**

**Motivazione e complessità computazionale.** Un Vision Transformer standard con patch size $p \times p$ su un'immagine $H \times W$ produce $T = \frac{H \cdot W}{p^2}$ token. Con $p = 16$ la complessità della self-attention globale è:

$$T_{ViT}(H,W) = \frac{H \cdot W}{p^2}, \qquad \text{FLOPs}_{ViT} = O\!\left(T_{ViT}^2 \cdot d\right) = O\!\left(\frac{H^2 W^2}{p^4} \cdot d\right)$$

Per immagini ad alta risoluzione (es. $H = 3000,\ W = 2000$) questo produce $T_{ViT} = 23{,}375$ token e $\approx 546M \cdot d$ operazioni: circa 45 secondi su RTX 3080, incompatibile con il budget di 10 secondi.

Lo **Swin Transformer** risolve questo dividendo l'immagine in finestre non sovrapposte di $M \times M$ token e applicando self-attention solo all'interno di ogni finestra. Su $P_3 \in \mathbb{R}^{B \times 48 \times H_3 \times W_3}$, lo stage 4 con patch size 2 produce $T = \frac{H_3 \cdot W_3}{4}$ token. Il numero di finestre è $\frac{H_3}{2M} \times \frac{W_3}{2M}$, ciascuna con $M^2$ token:

$$\text{FLOPs}_{\text{Swin}} \propto T \cdot M^2 \cdot d = \frac{H_3 W_3}{4} \cdot M^2 \cdot d$$

Il rapporto di efficienza rispetto al ViT puro, che ha $T_{ViT} = H_3 W_3 / 4$ token con self-attention globale $O(T_{ViT}^2)$, è:

$$\frac{\text{FLOPs}_{ViT}}{\text{FLOPs}_{Swin}} = \frac{T_{ViT}}{M^2} = \frac{H_3 W_3}{4 M^2}$$

Con $M=7$, il rapporto vale $\frac{H_3 W_3}{4 \cdot 49}$: per un esempio concreto a $H=3000,\ W=2000$ si ottiene $\approx \mathbf{477\times}$; cresce quadraticamente con la risoluzione, rendendo Swin sempre più vantaggioso per immagini ad alta risoluzione.

**Self-Attention con finestre (W-MSA).** Per una finestra contenente token $\{z_1, \ldots, z_{M^2}\} \subset \mathbb{R}^d$, la W-MSA con $h = 4$ teste di attenzione calcola:

$$\text{W-MSA}(\mathbf{Z}) = \text{Concat}(\text{head}_1, \ldots, \text{head}_4) \mathbf{W}^O$$

dove per la testa $k$:

$$\text{head}_k = \text{Softmax}\!\left(\frac{\mathbf{Q}_k \mathbf{K}_k^T}{\sqrt{d/4}} + \mathbf{B}_k\right) \mathbf{V}_k$$

con $\mathbf{Q}_k = \mathbf{Z}\mathbf{W}_k^Q,\ \mathbf{K}_k = \mathbf{Z}\mathbf{W}_k^K,\ \mathbf{V}_k = \mathbf{Z}\mathbf{W}_k^V \in \mathbb{R}^{M^2 \times (d/4)}$ e $\mathbf{B}_k \in \mathbb{R}^{M^2 \times M^2}$ matrice di bias posizionale relativo appresa.

La scelta di $h = 4$ teste (ridotta rispetto alle 8 del progetto originale) è motivata dalla dimensione ridotta dei canali ($d = 96$ in stage 4, $d = 192$ in stage 5): con 4 teste si ottengono dimensioni per testa $d/h = 24$ e $48$ rispettivamente, sufficienti per una rappresentazione espressiva mantenendo un footprint di memoria compatibile con fp16 su RTX 3080.

**Shifted Window (SW-MSA).** Per garantire connettività tra finestre adiacenti, i layer alternano W-MSA con SW-MSA, dove le finestre sono traslate di $(\lfloor M/2 \rfloor, \lfloor M/2 \rfloor)$ token. Con $L$ layer Swin e finestre $M \times M$, ogni token ha ricevuto informazione da regioni distanti fino a $\lfloor L/2 \rfloor \cdot M$ token in ogni direzione — sufficiente per catturare dipendenze globali a qualsiasi risoluzione.

**Rotary Position Embedding (RoPE).** Il positional encoding assoluto standard aggiunge un vettore $\mathbf{p}_m \in \mathbb{R}^d$ al token in posizione $m$, appreso durante il training. Questo crea un problema di **distribution shift**: il modello viene addestrato su crop $512 \times 384$ per efficienza di memoria, ma all'inferenza opera su risoluzioni arbitrariamente maggiori. Le posizioni viste all'inferenza sono al di fuori del range visto al training.

RoPE risolve questo codificando la posizione come **rotazione** nel piano complesso. Per la posizione $m$, il vettore $\mathbf{q}_m$ viene trasformato come:

$$f(\mathbf{q}, m)_j = q_j e^{im\theta_j}, \quad \theta_j = \frac{1}{10000^{2j/d}}$$

Il prodotto scalare tra query alla posizione $m$ e key alla posizione $n$ diventa:

$$f(\mathbf{q}, m)^T f(\mathbf{k}, n) = \sum_{j=1}^{d/2} \text{Re}\!\left[(q_{2j-1} + iq_{2j})(k_{2j-1} - ik_{2j}) e^{i(m-n)\theta_j}\right] = g(\mathbf{q}, \mathbf{k}, m-n)$$

**Proprietà chiave**: il prodotto dipende solo dalla differenza $m-n$ (distanza relativa), non dai valori assoluti $m$ e $n$. Di conseguenza il modello generalizza immediatamente a qualsiasi risoluzione senza distribution shift posizionale.

Le due trasformazioni Swin producono feature maps a dimensioni simboliche:

$$P_4 = \text{SwinStage4}(P_3;\,\text{RoPE}) \in \mathbb{R}^{B \times 96 \times H_4 \times W_4}$$

$$P_5 = \text{SwinStage5}(P_4;\,\text{RoPE}) \in \mathbb{R}^{B \times 192 \times H_5 \times W_5}$$

Ogni token in $P_5$ rappresenta una regione $(H/H_5) \times (W/W_5)$ pixel dell'immagine originale e, dopo i layer Swin, porta informazione contestuale dell'intera scena. Tutte le operazioni di attenzione vengono eseguite in fp16 con accumulazione in fp32 per la softmax, seguendo le best practice di stabilità numerica con mixed precision.

**Gradient checkpointing sull'encoder.** I layer Swin del CNN stem vengono eseguiti con **gradient checkpointing**: durante il forward pass i tensori intermedi non vengono mantenuti in memoria; durante il backward pass vengono ricalcolati on-the-fly. Questo riduce il picco di occupazione VRAM di circa il 40–50% a scapito di un incremento del tempo di training del ~30%. Con risoluzione di training $512 \times 384$ e batch size 4 in fp16, questa scelta è necessaria per mantenere l'utilizzo VRAM entro i 10 GB disponibili su RTX 3080.

---

##### **3. Set Transformer per il Style Prototype**

Sia $\mathcal{D}_\phi = \{(I_i^{src}, I_i^{tgt})\}_{i=1}^N$ il training set del fotografo $\phi$. L'obiettivo è costruire un vettore $\mathbf{s} \in \mathbb{R}^{256}$ che rappresenti lo stile del fotografo in modo robusto agli outlier (sessioni di editing anomale, foto in condizioni inusuali).

**Step 1 — Calcolo delle edit delta.** Per ogni coppia, l'encoder condiviso $\text{Enc}: \mathbb{R}^{H \times W \times 3} \to \mathbb{R}^{192}$ (il CNN stem MobileNetV3-Small con global average pool, che produce il vettore dalla dimensione del canale finale $P_3$ dopo proiezione lineare a 192) estrae feature semantiche:

$$\boldsymbol{\delta}_i = \text{Enc}(I_i^{tgt}) - \text{Enc}(I_i^{src}) \in \mathbb{R}^{192}, \quad i = 1, \ldots, N$$

$\boldsymbol{\delta}_i$ è il vettore di editing nello spazio delle feature: rappresenta come il fotografo ha trasformato la scena $i$ dalle sue feature originali a quelle editate. L'aggregazione naive $\bar{\boldsymbol{\delta}} = \frac{1}{N}\sum_i \boldsymbol{\delta}_i$ è sensibile agli outlier: un'unica coppia anomala può spostare significativamente la media.

**Step 2 — Aggregazione con Set Transformer.** Sia $\Delta = [\boldsymbol{\delta}_1, \ldots, \boldsymbol{\delta}_N]^T \in \mathbb{R}^{N \times 192}$. Il Set Transformer applica $L_{ST}=2$ layer di self-attention:

$$\tilde{\Delta}^{(0)} = \Delta \mathbf{W}_{in} \in \mathbb{R}^{N \times 256} \quad \text{(proiezione input)}$$

$$\tilde{\Delta}^{(l+1)} = \text{LayerNorm}\!\left(\tilde{\Delta}^{(l)} + \text{MHSA}\!\left(\tilde{\Delta}^{(l)}\right)\right)$$

dove MHSA è la Multi-Head Self-Attention con $h_{ST} = 4$ teste:

$$\text{MHSA}(\tilde{\Delta}) = \text{Concat}(\text{head}_1, \ldots, \text{head}_4)\mathbf{W}^O_{ST}$$

$$\text{head}_k = \text{Softmax}\!\left(\frac{\tilde{\Delta}\mathbf{W}_{ST,k}^Q (\tilde{\Delta}\mathbf{W}_{ST,k}^K)^T}{\sqrt{256/4}}\right) \tilde{\Delta}\mathbf{W}_{ST,k}^V$$

Il meccanismo di self-attention impara a pesare le delta: coppie stilisticamente coerenti con le altre (cioè "tipiche" dello stile del fotografo) ricevono attention weight alto, le coppie anomale ricevono weight basso. Il risultato $\tilde{\Delta}^{(L_{ST})} \in \mathbb{R}^{N \times 256}$ viene aggregato con pooling:

$$\mathbf{s} = \mathbf{W}_{out} \cdot \frac{1}{N}\sum_{i=1}^N \tilde{\Delta}^{(L_{ST})}_i \in \mathbb{R}^{256}$$

**Teorema 1 (Invarianza Permutazionale):** Il Set Transformer è invariante a permutazioni dell'input: $\forall \pi \in S_N$, $\text{SetTransformer}(\{\boldsymbol{\delta}_{\pi(i)}\}) = \text{SetTransformer}(\{\boldsymbol{\delta}_i\})$.

*Dimostrazione:* La self-attention è invariante alle permutazioni dell'input poiché i query e key vengono calcolati indipendentemente per ogni token e la softmax opera su tutti i token simultaneamente. Il pooling finale $\frac{1}{N}\sum_i$ è anch'esso invariante. La composizione di operazioni invarianti è invariante. $\square$

Questa proprietà è fondamentale: l'ordine con cui le coppie di training vengono caricate non influenza il prototype — il che è desiderabile perché l'ordine è arbitrario.

---

##### **4. Cross-Attention per In-Context Style Conditioning**

Il vettore $\mathbf{s}$ cattura lo stile globale del fotografo, ma non considera il contenuto della specifica immagine da gradare. Se il training set contiene 30 tramonti e 70 ritratti in studio, la media pesata $\mathbf{s}$ è dominata dai ritratti — ma un tramonto dovrebbe essere gradato usando primariamente le coppie tramonti del training set.

Il meccanismo di cross-attention risolve questo problema, realizzando una forma di **retrieval-augmented conditioning**: dato il contenuto dell'immagine di test, si recuperano dinamicamente le edit delta più rilevanti dal training set. Per contenere il costo computazionale e il footprint di memoria in fp16, si usa un **subset fisso di $K = 20$ coppie** del training set come memoria del cross-attention, selezionate all'inizio della fase di adattamento come le 20 coppie a più alta diversità di contenuto (misurata dalla varianza delle feature Enc).

Sia $\mathbf{Z}_{test} = \text{Flatten}(P_5) \in \mathbb{R}^{T(H,W) \times 192}$ con $T(H,W) = H_5 \cdot W_5$ token (variabile con la risoluzione). Siano $\mathbf{K}_{train} = [\text{Enc}(I_1^{src}), \ldots, \text{Enc}(I_K^{src})]^T \in \mathbb{R}^{K \times 192}$ le feature delle $K=20$ immagini sorgente del subset fisso, e $\mathbf{V}_{train} = [\boldsymbol{\delta}_1, \ldots, \boldsymbol{\delta}_K]^T \in \mathbb{R}^{K \times 192}$ le corrispondenti edit delta.

Le proiezioni lineari sono:

$$\mathbf{Q} = \mathbf{Z}_{test}\mathbf{W}^Q \in \mathbb{R}^{T \times d_c}, \quad \mathbf{K} = \mathbf{K}_{train}\mathbf{W}^K \in \mathbb{R}^{K \times d_c}, \quad \mathbf{V} = \mathbf{V}_{train}\mathbf{W}^V \in \mathbb{R}^{K \times d_c}$$

con $d_c = 192$. Il meccanismo di cross-attention è:

$$\mathbf{A} = \text{Softmax}\!\left(\frac{\mathbf{Q}\mathbf{K}^T}{\sqrt{d_c}}\right) \in \mathbb{R}^{T \times K}$$

$$\text{context} = \mathbf{A}\mathbf{V} \in \mathbb{R}^{T \times 192}$$

L'elemento $A_{t,n} \geq 0$ (con $\sum_n A_{t,n} = 1$) rappresenta il peso che il token $t$ dell'immagine di test assegna alla coppia $n$ del subset fisso. Un token nella regione "cielo" dell'immagine di test assegnerà peso alto alle coppie del subset che hanno cieli simili, recuperando la corrispondente edit delta come conditioning — realizzando automaticamente il sub-style condizionato al contenuto della scena.

**Motivazione del subset fisso $K=20$.** L'uso di tutte le $N$ coppie di training nel cross-attention avrebbe complessità $O(T \cdot N)$ con $N$ fino a 200, che a risoluzione piena risulta proibitivo per memoria e tempo. Con $K=20$ fissato, la matrice di attenzione $\mathbf{A} \in \mathbb{R}^{T \times 20}$ è compatta e il costo computazionale è costante rispetto alla dimensione del training set. La selezione per massima diversità garantisce che le 20 coppie coprano lo spazio visivo del fotografo il più uniformemente possibile.

**Proprietà (Attention Softmax come Mixture of Experts):** $\text{context}_t = \sum_{n=1}^K A_{t,n} \boldsymbol{\delta}_n^{proj}$ è una media pesata delle edit delta proiettate, con pesi dipendenti dalla similarità tra il token di test e i sorgenti del subset. Nel limite $T \to \infty$, la distribuzione di attenzione collassa su $\arg\max_n \langle \mathbf{Q}_t, \mathbf{K}_n \rangle$ — il nearest neighbor stilistico esatto nel subset.

Il tensore `context` viene risagomato a mappa spaziale $\mathbb{R}^{B \times 192 \times H_5 \times W_5}$ e interpolato bilinearmente alle risoluzioni di $P_3$ e $P_4$ per il conditioning nei branch.

---

##### **5. AdaIN Conditioning nel Global Branch**

L'Adaptive Instance Normalization (AdaIN) è il meccanismo che traduce il vettore di stile $\mathbf{s} \in \mathbb{R}^{256}$ in una modulazione delle feature maps di $P_5$.

Sia $\mathbf{h} \in \mathbb{R}^{C \times H' \times W'}$ una feature map con $C$ canali. Per il canale $c$, i parametri di normalizzazione dipendenti dallo stile sono:

$$\gamma_c(\mathbf{s}) = \mathbf{w}_{\gamma,c}^T \mathbf{s} + b_{\gamma,c}, \qquad \beta_c(\mathbf{s}) = \mathbf{w}_{\beta,c}^T \mathbf{s} + b_{\beta,c}$$

con $\mathbf{w}_{\gamma,c}, \mathbf{w}_{\beta,c} \in \mathbb{R}^{256}$ appresi. L'operazione AdaIN è:

$$\text{AdaIN}(\mathbf{h}, \mathbf{s})_c = \gamma_c(\mathbf{s}) \cdot \frac{\mathbf{h}_c - \mu_c(\mathbf{h})}{\sigma_c(\mathbf{h}) + \epsilon} + \beta_c(\mathbf{s})$$

dove $\mu_c(\mathbf{h}) = \frac{1}{H'W'}\sum_{i,j} h_c(i,j)$ e $\sigma_c(\mathbf{h}) = \sqrt{\frac{1}{H'W'}\sum_{i,j}(h_c(i,j)-\mu_c)^2 + \epsilon}$ sono media e deviazione standard della feature map nel canale $c$.

**Interpretazione:** AdaIN rimuove le statistiche del primo e secondo ordine della feature (normalizza a media 0, varianza 1) e le rimpiazza con quelle dettate dallo stile $\mathbf{s}$. Huang & Bethge (2017) dimostrano che le statistiche di secondo ordine (covarianza) delle feature maps di una CNN encodano lo stile visivo: manipolandole si trasferisce lo stile mantenendo la struttura spaziale intatta.

Le feature modulate $\tilde{P}_5 = \text{AdaIN}(P_5, \mathbf{s}) \in \mathbb{R}^{B \times 192 \times H_5 \times W_5}$ vengono poi elaborate dal Global Branch:

$$\mathbf{f}_{global} = \text{GAP}(\tilde{P}_5) \in \mathbb{R}^{B \times 192} \quad \text{(Global Average Pool, elimina le dimensioni spaziali)}$$

$$G_{global} = \text{reshape}\!\left(\mathbf{W}_{gb,2} \cdot \delta\!\left(\mathbf{W}_{gb,1} \cdot \mathbf{f}_{global}\right)\right) \in \mathbb{R}^{B \times 12 \times 8 \times 8 \times L_b}$$

con $\mathbf{W}_{gb,1} \in \mathbb{R}^{256 \times 192},\ \mathbf{W}_{gb,2} \in \mathbb{R}^{(12 \cdot 8^2 \cdot L_b) \times 256}$ e $\delta$ ReLU. La grid globale ha risoluzione spaziale **fissa** $8 \times 8$ indipendentemente da $(H, W)$: il GAP ha già eliminato le dimensioni spaziali prima dei layer FC.

---

##### **6. SPADE Conditioning nel Local Branch**

SPADE (Park et al., CVPR 2019) estende AdaIN con conditioning **spazialmente variabile**: i parametri $\gamma$ e $\beta$ sono funzioni della posizione $(x,y)$, non costanti sul canale.

Sia $\mathbf{m} \in \mathbb{R}^{C_m \times H_m \times W_m}$ la mappa di conditioning (il tensore `context` risagomato), con $C_m = 128$ canali (ridotti rispetto alla versione originale per contenere il consumo di memoria in fp16). I parametri SPADE alla posizione $(x,y)$ sono:

$$\gamma_c(x,y) = \left[\text{Conv}_\gamma(\mathbf{m})\right]_{c,x,y}, \qquad \beta_c(x,y) = \left[\text{Conv}_\beta(\mathbf{m})\right]_{c,x,y}$$

dove $\text{Conv}_\gamma, \text{Conv}_\beta: \mathbb{R}^{128 \times H_m \times W_m} \to \mathbb{R}^{C \times H' \times W'}$ sono convoluzioni $3 \times 3$. L'operazione è:

$$\text{SPADE}(\mathbf{h}, \mathbf{m})_{c,x,y} = \gamma_c(x,y) \cdot \frac{h_{c,x,y} - \mu_c(\mathbf{h})}{\sigma_c(\mathbf{h}) + \epsilon} + \beta_c(x,y)$$

**Differenza fondamentale rispetto ad AdaIN:** In AdaIN, $\gamma_c$ e $\beta_c$ sono scalari (costanti su tutto il piano spaziale). In SPADE sono mappe spaziali: ogni pixel riceve un conditioning diverso. Questo permette al Local Branch di applicare trasformazioni cromatiche diverse su regioni diverse dell'immagine — esattamente come un fotografo applica skin tone warm sui volti e desaturation sul cielo nella stessa immagine. La riduzione a $C_m = 128$ canali riduce il costo delle convoluzioni SPADE del 75% rispetto a $C_m = 256$, con impatto trascurabile sulla qualità grazie alla compressione dell'informazione già operata dal cross-attention.

Il Local Branch processa le feature di $P_4$ e $P_3$ con blocchi SPADE-ResBlock e upsampling, producendo la bilateral grid di compromesso a risoluzione $16 \times 16$:

$$\mathbf{x}_4 = \text{SPADEResBlock}(P_4,\ \mathbf{m}_{H_4}) \in \mathbb{R}^{B \times 96 \times H_4 \times W_4}$$

$$\mathbf{x}_3 = \text{SPADEResBlock}(\text{Up}(\mathbf{x}_4) + P_3,\ \mathbf{m}_{H_3}) \in \mathbb{R}^{B \times 96 \times H_3 \times W_3}$$

$$G_{local} = \text{reshape}\!\left(\text{Conv}_{1 \times 1}\!\left(\text{AdaptiveAvgPool}(\mathbf{x}_3,\; (16,16))\right)\right) \in \mathbb{R}^{B \times 12 \times 16 \times 16 \times L_b}$$

dove $\mathbf{m}_{H_4}$ e $\mathbf{m}_{H_3}$ sono il tensore `context` (proiettato a 128 canali con $\text{Conv}_{1\times1}$) interpolato bilinearmente alle rispettive risoluzioni $(H_4, W_4)$ e $(H_3, W_3)$, e Up denota upsampling bilineare $\times 2$. L'`AdaptiveAvgPool` riduce qualsiasi tensore di dimensioni spaziali $(H_3, W_3)$ a $(16, 16)$ — la grid locale ha risoluzione fissa $16 \times 16$ indipendentemente dalla risoluzione dell'input.

**Scelta della risoluzione $16 \times 16$.** La risoluzione $16 \times 16$ è un compromesso tra la grid globale $8 \times 8$ e la grid fine $32 \times 32$ del progetto originale. Essa cattura variazioni spaziali a livello di zona (sky, skin, shadow, midground) senza il costo di memoria e tempo di una grid $32 \times 32$, che in fp16 con batch size 4 avrebbe richiesto circa 800 MB aggiuntivi di VRAM per le mappe di coefficienti.

**SPADE ResBlock completo.** Sia $\mathbf{h} \in \mathbb{R}^{C \times H' \times W'}$ l'input del blocco e $\mathbf{m} \in \mathbb{R}^{128 \times H' \times W'}$ la mappa di conditioning (context interpolato, proiettato a 128 canali). Il blocco implementa due rami: il **residual path** applica due operazioni SPADE-normalizzazione + convoluzione, e lo **shortcut** adatta le dimensioni dei canali se necessario.

*Residual path:*

$$\mathbf{h}_1 = \text{Conv}_{3\times3}\!\left(\delta\!\left(\text{SPADE}(\mathbf{h},\, \mathbf{m})\right)\right)$$

$$\mathbf{h}_2 = \text{Conv}_{3\times3}\!\left(\delta\!\left(\text{SPADE}(\mathbf{h}_1,\, \mathbf{m})\right)\right)$$

*Shortcut path* (attivo solo se $C_{in} \neq C_{out}$):

$$\mathbf{h}_{sc} = \text{Conv}_{1\times1}^{C_{in} \to C_{out}}(\mathbf{h})$$

*Output del blocco (skip connection):*

$$\text{SPADEResBlock}(\mathbf{h}, \mathbf{m}) = \mathbf{h}_2 + \mathbf{h}_{sc}$$

La skip connection permette al gradiente di fluire direttamente dall'output all'input del blocco senza attraversare le normalizzazioni SPADE — fondamentale per stabilizzare il training con gradienti di SPADE che dipendono dalla mappa di conditioning variabile $\mathbf{m}$.

**Differenza da ResNet classico.** In un ResNet standard, la normalizzazione è Batch Normalization (BN), i cui parametri $\gamma, \beta$ sono scalari appresi globalmente per canale. In SPADE, $\gamma_c(x,y)$ e $\beta_c(x,y)$ sono mappe spaziali derivate da $\mathbf{m}$ — ogni posizione spaziale riceve normalizzazione diversa. Questo permette al Local Branch di applicare trasformazioni cromatiche spazialmente eterogenee (ad esempio: skin tone warm nei volti, desaturazione nel cielo) con la stessa architettura di base ResNet.

---

##### **7. Confidence Mask**

La Confidence Mask $\alpha \in [0,1]^{H \times W}$ determina pixel per pixel quanto peso assegnare al ramo locale rispetto al ramo globale. Viene predetta da una rete leggera che processa le feature di media risoluzione:

$$\tilde{P}_4 = \text{BilinearUp}(P_4,\; (H_3, W_3)) \in \mathbb{R}^{B \times 96 \times H_3 \times W_3}$$

$$\mathbf{z}_{mask} = \delta\!\left(\text{Conv}_{3\times3}([\tilde{P}_4;\ P_3])\right) \in \mathbb{R}^{B \times 64 \times H_3 \times W_3}$$

$$\alpha_{low} = \sigma\!\left(\text{Conv}_{1\times1}(\mathbf{z}_{mask})\right) \in [0,1]^{B \times 1 \times H_3 \times W_3}$$

$$\alpha = \text{BilinearUp}(\alpha_{low},\; (H, W)) \in [0,1]^{B \times 1 \times H \times W}$$

dove $[\cdot;\cdot]$ denota concatenazione lungo i canali e $\sigma$ è la funzione sigmoide. Il upsampling bilineare finale garantisce che la mappa sia smooth — le transizioni tra zone a diverso conditioning sono graduali, evitando bordi artefattuali.

---

### 5.2 Perché CNN + Swin e Non ViT Puro

La scelta dell'encoder ibrido non è arbitraria. Confronto quantitativo (i valori numerici si riferiscono all'esempio $H=3000, W=2000$ per concretezza; la colonna CNN+Swin scala simbolicamente per qualsiasi risoluzione):

| | CNN Pura (MobileNetV3-Small) | ViT Puro (patch 16) | CNN + Swin (proposto) |
|--|------------------------------|--------------------|-----------------------|
| **Token/patch** | — | $T_{ViT}(H,W) = HW/p^2$ | $T(H,W) = H_5 W_5$ |
| **Complessità attention** | — | $O(T_{ViT}^2 \cdot d)$ | $O(T \cdot M^2 \cdot d)$ |
| **Tempo encoder (es. $H=3000, W=2000$)** | ~0.5s | ~45s ❌ | ~1.8s ✅ |
| **Context globale** | ⚠️ Limitato | ✅✅ | ✅✅ |
| **Feature locali** | ✅✅ | ⚠️ | ✅✅ |
| **Few-shot (100 coppie)** | ✅ | ❌ overfitting | ✅ |
| **Pre-training utile** | ✅ ImageNet | ⚠️ parziale | ✅ entrambi |
| **"Tramonto → warm"** | ❌ | ✅✅ | ✅✅ |
| **Parametri totali** | ~2.5M | ~86M | ~8M |
| **fp16 VRAM (batch 4, 512×384)** | ~2 GB | OOM ❌ | ~6 GB ✅ |

La CNN stem leggera (MobileNetV3-Small) con inductive bias locale gestisce ciò che la CNN fa meglio; lo Swin con 4 teste di attenzione gestisce le relazioni globali con costo computazionale lineare, rimanendo entro i budget di memoria e tempo imposti dall'hardware consumer.

---
### 5.3 Meta-Learning: Reptile con Task Augmentation

#### Motivazione: il problema del meta-overfitting con pochi fotografi

Con soli 5 fotografi disponibili in MIT-Adobe FiveK, un algoritmo di meta-learning classico come MAML affronta un duplice problema: (1) **meta-overfitting** — il modello impara ad adattarsi rapidamente ai 5 stili specifici dei fotografi A–E, ma non acquisisce la capacità generalizzata di adattarsi a uno stile arbitrario; (2) **costo computazionale proibitivo** — MAML richiede backpropagation attraverso il ciclo di ottimizzazione interno (derivate del secondo ordine, o Hessiani), che su GPU consumer con fp16 e gradient checkpointing risulta instabile e lento.

**Reptile** (Nichol et al., 2018) risolve il secondo problema mantenendo le garanzie del primo: è un algoritmo di meta-learning del primo ordine che non richiede la computazione dell'Hessiano, rendendolo fino a $4\times$ più veloce di MAML per iterazione e completamente stabile in mixed precision fp16.

**Formulazione di Reptile.** Sia $\mathcal{T}$ un task campionato dalla distribuzione $p_{aug}$. Reptile esegue $k$ passi di SGD partendo da $\theta$ sul task $\mathcal{T}$, ottenendo parametri adattati $\tilde{\theta}_\mathcal{T}$, poi aggiorna i meta-parametri in direzione della differenza:

$$\tilde{\theta}_\mathcal{T} = \mathcal{U}_\alpha^k(\theta, \mathcal{D}_\mathcal{T}) = \theta - \alpha \sum_{t=0}^{k-1} \nabla_{\theta_t} \mathcal{L}_\mathcal{T}(\theta_t)$$

$$\theta \leftarrow \theta + \epsilon\,(\tilde{\theta}_\mathcal{T} - \theta)$$

dove $\alpha = 10^{-3}$ è il learning rate interno, $k = 5$ è il numero di passi interni, e $\epsilon = 10^{-2}$ è il meta-step size. Con un batch di $M = 2$ task per iterazione, l'aggiornamento è la media delle direzioni di adattamento:

$$\theta \leftarrow \theta + \frac{\epsilon}{M}\sum_{m=1}^{M}(\tilde{\theta}_{\mathcal{T}_m} - \theta)$$

**Interpretazione geometrica di Reptile.** Nichol et al. (2018) dimostrano che l'aggiornamento Reptile è un'approssimazione del primo ordine dell'aggiornamento MAML. Formalmente, lo spostamento $\tilde{\theta}_\mathcal{T} - \theta$ approssima $-\alpha k \nabla_\theta \mathbb{E}_\mathcal{T}[\mathcal{L}_\mathcal{T}(\theta_\mathcal{T}^*)]$ più un termine di ordine superiore che favorisce la convergenza verso parametri con bassa curvatura rispetto alla distribuzione dei task — esattamente la proprietà desiderata di un'inizializzazione universale.

**Confronto MAML vs Reptile su GPU consumer con fp16:**

| | MAML | Reptile |
|--|------|---------|
| Ordine gradienti | 2° (Hessiani) | 1° |
| Compatibilità fp16 | Instabile (overflow Hessiani) | ✅ Stabile |
| Tempo per iterazione (RTX 3080) | ~8s | ~2s |
| Memoria per iterazione | ~18 GB ❌ | ~6 GB ✅ |
| Qualità meta-inizializzazione | ✅✅ | ✅ (leggermente inferiore) |

Formalmente, se $\mathcal{L}$ è $L$-smooth, Reptile converge a un punto stazionario della meta-loss con velocità $O(1/\sqrt{K})$ dove $K$ è il numero di iterazioni, analogamente a SGD su funzioni smooth.

#### Task Augmentation: generazione di task sintetici

Dati due fotografi $\phi_i$ e $\phi_j$ che condividono le stesse immagini sorgente $\{I_k^{src}\}$ ma producono target diversi $\{I_k^{tgt,i}\}$ e $\{I_k^{tgt,j}\}$, definiamo un task sintetico con parametro $\lambda \in (0,1)$ tramite interpolazione lineare in spazio CIE Lab:

$$I_k^{tgt,\lambda} = \mathcal{L}^{-1}\!\Big(\lambda\,\mathcal{L}(I_k^{tgt,i}) + (1-\lambda)\,\mathcal{L}(I_k^{tgt,j})\Big)$$

dove $\mathcal{L}: \mathbb{R}^{H\times W\times 3}_{\text{RGB}} \to \mathbb{R}^{H\times W\times 3}_{\text{Lab}}$ è la conversione in spazio CIE Lab e $\mathcal{L}^{-1}$ la sua inversa.

**Conversione inversa $\mathcal{L}^{-1}$: CIE Lab → sRGB.** L'inversa della pipeline §6.4.1 percorre i passi in ordine inverso:

*Passo 1 — Lab → XYZ:*

$$f^{-1}(t) = \begin{cases} t^3 & t > 6/29 \\ 3(6/29)^2\!\left(t - 4/29\right) & t \leq 6/29 \end{cases}$$

$$X = X_n\,f^{-1}\!\left(\frac{L^*+16}{116} + \frac{a^*}{500}\right), \quad Y = Y_n\,f^{-1}\!\left(\frac{L^*+16}{116}\right), \quad Z = Z_n\,f^{-1}\!\left(\frac{L^*+16}{116} - \frac{b^*}{200}\right)$$

*Passo 2 — XYZ D65 → sRGB lineare:*

$$\begin{pmatrix} R_{lin} \\ G_{lin} \\ B_{lin} \end{pmatrix} = \mathbf{M}_{XYZ\to sRGB} \begin{pmatrix} X/X_n \\ Y/Y_n \\ Z/Z_n \end{pmatrix}$$

(con $\mathbf{M}_{XYZ\to sRGB}$ dalla sezione 6.2.7, e clip a $[0,1]$).

*Passo 3 — sRGB lineare → sRGB (gamma encoding):* applicazione di $\gamma_{sRGB}$ dalla sezione 6.2.8.

L'inversione è ben definita eccetto per colori fuori gamut (valori Lab che non corrispondono a colori sRGB realizzabili), dove il clipping introduce una piccola distorsione. Per i task sintetici di augmentation questo è accettabile: i valori interpolati $I_k^{tgt,\lambda}$ per $\lambda \in (0.1, 0.9)$ restano quasi sempre entro gamut se gli estremi $I_k^{tgt,i}$ e $I_k^{tgt,j}$ lo sono.

**Perché l'interpolazione in Lab e non in RGB.** Lo spazio sRGB è non lineare rispetto alla percezione: la distanza euclidea in RGB non corrisponde alla differenza percepita. In CIE Lab, per costruzione, una differenza $\Delta E = 1$ è al limite della discriminabilità umana, e la distanza euclidea è percettivamente uniforme. L'interpolazione lineare in Lab produce dunque uno stile genuinamente intermedio: $\lambda = 0.5$ tra un fotografo "warm" e uno "cool" dà un fotografo "neutro" in senso percettivo, non in senso aritmetico RGB.

Con $\binom{5}{2} = 10$ coppie di fotografi e $\lambda \sim \mathcal{U}(0.1, 0.9)$, la distribuzione dei task sintetici diventa:

$$p_{aug}(\mathcal{T}) = \sum_{i<j} w_{ij} \int_0^1 p(\mathcal{T}|\phi_i,\phi_j,\lambda)\,d\lambda$$

che è continua e copre lo spazio degli stili in modo molto più denso dei 5 task discreti originali.

#### Formulazione Reptile con batch di $M = 2$ task

Sia $\mathcal{T} = (\mathcal{D}^{sup}_\mathcal{T}, \mathcal{D}^{qry}_\mathcal{T})$ un task generico con support set $\mathcal{D}^{sup} = \{(I_k^{src}, I_k^{tgt})\}_{k=1}^{K_s}$ e query set $\mathcal{D}^{qry} = \{(I_k^{src}, I_k^{tgt})\}_{k=1}^{K_q}$, con $K_s = 15$ e $K_q = 5$.

**Inner loop** (adattamento al task $\mathcal{T}$, $k = 5$ passi di SGD in fp16):

$$\mathcal{L}_\mathcal{T}^{sup}(\theta) = \frac{1}{K_s}\sum_{k=1}^{K_s} \mathcal{L}\!\left(f_\theta(I_k^{src}),\ I_k^{tgt}\right)$$

$$\theta_\mathcal{T}^{(0)} = \theta, \qquad \theta_\mathcal{T}^{(t+1)} = \theta_\mathcal{T}^{(t)} - \alpha\,\nabla_{\theta_\mathcal{T}^{(t)}}\mathcal{L}_\mathcal{T}^{sup}\!\left(\theta_\mathcal{T}^{(t)}\right), \quad t=0,\ldots,k-1$$

con $\alpha = 10^{-3}$. Definiamo $\tilde{\theta}_\mathcal{T} := \theta_\mathcal{T}^{(k)}$ i parametri adattati al task.

**Outer loop — Reptile** (meta-aggiornamento su $M = 2$ task campionati da $p_{aug}$):

$$\theta \leftarrow \theta + \frac{\epsilon}{M}\sum_{m=1}^{M}(\tilde{\theta}_{\mathcal{T}_m} - \theta), \quad \epsilon = 10^{-2}$$

Non è richiesta alcuna differenziazione di secondo ordine: l'aggiornamento usa solo le differenze di parametri $\tilde{\theta}_\mathcal{T} - \theta$, calcolabili interamente in fp16 senza rischio di overflow numerico.

**Stabilità numerica fp16.** Le operazioni di inner loop vengono eseguite con GradScaler di PyTorch AMP: il loss scalato di un fattore $S = 2^{10}$ previene underflow dei gradienti in fp16, e la divisione per $S$ prima dell'aggiornamento dei parametri ripristina la scala corretta. L'assenza di Hessiani elimina il principale punto di instabilità numerica di MAML in mixed precision.

**Teorema 2 (Convergenza di Reptile, Nichol et al. 2018).** Sia $\mathcal{L}_\mathcal{T}$ $L$-smooth per ogni task $\mathcal{T}$ e la meta-loss $\bar{\mathcal{L}}(\theta) = \mathbb{E}_\mathcal{T}[\mathcal{L}_\mathcal{T}(\tilde{\theta}_\mathcal{T})]$ lower-bounded da $\bar{\mathcal{L}}^*$. Con $\alpha < 1/L$ e $\epsilon = O(1/\sqrt{K})$, Reptile converge: per ogni $\varepsilon > 0$ esiste $K = O\!\left((\bar{\mathcal{L}}(\theta_0) - \bar{\mathcal{L}}^*)/\varepsilon^2\right)$ tale che $\min_{k \leq K} \|\nabla_\theta\,\bar{\mathcal{L}}(\theta_k)\| \leq \varepsilon$. $\square$

**Output del meta-training:** $\theta_{meta}$ è un insieme di parametri tali che, dato qualsiasi task $\mathcal{T}$ con $K_s = 15$ coppie di supporto, soli $k = 5$ passi di SGD sono sufficienti a specializzare il modello su quello stile. Questa è l'inizializzazione per la fase successiva.

---

### 5.4 Few-Shot Adaptation: Freeze-Then-Unfreeze

Partendo da $\theta_{meta}$, la fase di adattamento al fotografo target $\phi$ con dataset $\mathcal{D}_\phi = \{(I_k^{src}, I_k^{tgt})\}_{k=1}^N$, $N \in [50, 200]$, segue una strategia di congelamento progressivo per prevenire il **catastrophic forgetting** — il fenomeno per cui l'aggiornamento con dati del nuovo task cancella conoscenza generale appresa in precedenza.

#### Partizione del parametro space

Il parametro space $\Theta$ è partizionato in tre sottoinsiemi con profondità decrescente nella rete:

$$\Theta = \underbrace{\Theta_{freeze}}_{\text{CNN stage 1–2, Swin stage 4}} \;\cup\; \underbrace{\Theta_{slow}}_{\text{CNN stage 3, Swin stage 5}} \;\cup\; \underbrace{\Theta_{adapt}}_{\text{Branches, Set-Transformer, Cross-Attention}}$$

#### Fase 3A — adattamento parziale (epoche 1–10)

Solo $\Theta_{slow} \cup \Theta_{adapt}$ vengono aggiornati. Il gradiente su $\Theta_{freeze}$ è nullo:

$$\nabla_{\Theta_{freeze}}\,\mathcal{L} := \mathbf{0}$$

L'ottimizzatore è AdamW con decoupled weight decay in fp16 (con GradScaler):

$$m_t = \beta_1 m_{t-1} + (1-\beta_1)\,g_t, \qquad v_t = \beta_2 v_{t-1} + (1-\beta_2)\,g_t^2$$

$$\theta_{t+1} = \theta_t - \eta_A\!\left(\frac{\hat{m}_t}{\sqrt{\hat{v}_t}+\varepsilon} + \lambda_{wd}\,\theta_t\right)$$

con $\eta_A = 5\times10^{-5}$, $\beta_1 = 0.9$, $\beta_2 = 0.999$, $\varepsilon = 10^{-8}$, $\lambda_{wd} = 2\times10^{-3}$.

#### Fase 3B — fine-tuning globale (epoche 11–30)

Tutti i parametri vengono sbloccati con learning rate ridotto $\eta_B = \eta_A/2 = 2.5\times10^{-5}$ e cosine annealing:

$$\eta(t) = \eta_B\cdot\frac{1}{2}\!\left(1 + \cos\!\left(\frac{\pi\,(t - t_0)}{T_{max}}\right)\right), \quad t \in [t_0,\, t_0 + T_{max}]$$

con $t_0 = 10$, $T_{max} = 20$.

**Motivazione teorica del congelamento.** Sia $\mathcal{R}(\theta)$ la capacità di generalizzazione del modello (inverso della complessità di Rademacher sul dataset target). Con $N \leq 200$ campioni, il bound di generalizzazione è:

$$\mathbb{E}[\mathcal{L}(\theta)] \leq \hat{\mathcal{L}}(\theta) + \mathcal{O}\!\left(\sqrt{\frac{|\Theta|}{N}}\right)$$

Congelare $\Theta_{freeze}$ riduce il numero effettivo di parametri liberi $|\Theta_{adapt} \cup \Theta_{slow}| \ll |\Theta|$, stringendo il bound di generalizzazione e limitando il rischio di overfitting.

**Early stopping.** Sia $\mathcal{D}_\phi^{val} \subset \mathcal{D}_\phi$ un holdout del $20\%$ delle coppie. Se per $p = 5$ epoche consecutive la validation loss non decresce:

$$\mathcal{L}_{val}(t) \geq \min_{t' \leq t-p} \mathcal{L}_{val}(t')$$

il training si interrompe e si ripristina il checkpoint con loss minima.

---

### 5.5 Baseline per Ablation Studies

La baseline minimale $\mathcal{B}_0$ è definita come una rete senza componenti Transformer, senza Set Transformer, senza meta-learning e senza conditioning dipendente dallo stile. L'encoder leggero $\mathcal{E}_{light}$ è MobileNetV3-Small con global average pool:

$$\mathbf{f} = \frac{1}{H'W'}\sum_{i,j} \mathcal{E}_{light}(I_{src})_{i,j} \in \mathbb{R}^{48}$$

Due bilateral grid vengono predette da strati fully-connected applicati a $\mathbf{f}$:

$$G_1 = \text{reshape}\!\left(\mathbf{W}_1 \delta(\mathbf{U}_1 \mathbf{f})\right) \in \mathbb{R}^{12\times 8\times 8\times 8}, \quad G_2 = \text{reshape}\!\left(\mathbf{W}_2 \delta(\mathbf{U}_2 \mathbf{f})\right) \in \mathbb{R}^{12\times 16\times 16\times 8}$$

con $\delta$ ReLU. La rete viene addestrata con inizializzazione random direttamente su $\mathcal{D}_\phi$. $\mathcal{B}_0$ serve come lower bound rigoroso: ogni ΔE peggiore di $\mathcal{B}_0$ in un'ablation indicherebbe un errore implementativo.

---

### 5.6 Stima Tempi di Inferenza (RTX 3080, esempio $H=3000, W=2000$, fp16)

| Componente | Tempo stimato |
|------------|--------------|
| MobileNetV3-Small stem (stage 1-3) | ~0.5s |
| Swin Transformer stage 4-5 (4 teste) | ~0.8s |
| Cross-attention (K=20) + style conditioning | ~0.1s |
| Global BilGrid 8×8×8 | ~0.1s |
| Local BilGrid 16×16×8 | ~0.4s |
| Confidence mask + blending | ~0.3s |
| **Totale** | **~2.2s** ✅ |

Ampiamente dentro il budget di 10 secondi. L'uso di fp16 mixed precision contribuisce a ridurre il tempo di inferenza di circa il 30% rispetto a fp32, oltre a dimezzare il footprint di VRAM. Ulteriori ottimizzazioni con TensorRT o quantizzazione int8 porterebbero il totale sotto 1 secondo.

---

### 5.7 Probabilità di Successo

| Scenario | Probabilità | Note |
|----------|-------------|------|
| Output indistinguibile dal fotografo in blind test | **60-70%** | Con 150+ coppie diverse e meta-training Reptile corretto |
| Output preferito dal fotografo >50% dei casi | **50-60%** | Dipende dalla complessità degli edits locali; la grid 16×16 cattura zone principali ma non sub-zone fini come 32×32 |
| Output "meglio" per consistenza secondo giudici terzi | **55-65%** | Il modello applica lo stile medio senza varianza inter-sessione |
| Fallimento completo | **< 5%** | Solo con training set troppo omogeneo |

**Il fattore più predittivo del successo non è l'architettura: è la diversità del training set.** 150 coppie ben diversificate (ritratti, paesaggi, still life, low light, golden hour) producono un modello molto più robusto di 200 coppie tutte dello stesso tipo di scena. La scelta di MobileNetV3-Small al posto di EfficientNet-B4 comporta una leggera riduzione delle probabilità di successo (~5 punti percentuali) compensata da tempi di adattamento più brevi e minore rischio di overfitting nel regime few-shot.

---
## 6. Pipeline Matematiche Complete

### 6.1 Notazione

| Simbolo | Definizione |
|---------|-------------|
| $\mathbf{R} \in \mathbb{Z}^{H_{raw}\times W_{raw}}$ | Lettura grezza del sensore (valori interi, bit depth $b$) |
| $b_{dark},\ b_{sat}$ | Livello di nero e di saturazione del sensore (da metadati EXIF) |
| $R_{norm} \in [0,1]^{H_{raw}\times W_{raw}}$ | Segnale RAW linearizzato e normalizzato |
| $c(i,j) \in \{R, G_1, G_2, B\}$ | Funzione di maschera CFA (Color Filter Array, pattern Bayer) |
| $\hat{\mathbf{E}}^{full} \in [0,1]^{H_{raw}\times W_{raw}\times 3}$ | Immagine RGB lineare dopo demosaicatura |
| $\mathbf{w}_{wb} = (w_R, w_G, w_B)$ | Guadagni di bilanciamento del bianco (da metadati EXIF) |
| $\mathbf{M}_{cam\to XYZ} \in \mathbb{R}^{3\times 3}$ | Matrice di camera (Camera RGB → CIE XYZ, da metadati DNG) |
| $\mathbf{M}_{XYZ\to sRGB} \in \mathbb{R}^{3\times 3}$ | Matrice primarie sRGB (costante, da standard IEC 61966-2-1) |
| $\gamma_{sRGB}(\cdot)$ | Funzione di trasferimento sRGB (gamma encoding, EOTF) |
| $\gamma_{sRGB}^{-1}(\cdot)$ | Inversa della EOTF sRGB (linearizzazione) |
| $s \in (0,1]$ | Fattore di scala per downsampling al training |
| $L_{train}$ | Lato lungo target per training ($L_{train} = 512$ px, risoluzione $512\times384$) |
| $(H_s, W_s)$ | Dimensioni ridotte: $H_s = \lfloor s\cdot H_0\rceil$, $W_s = \lfloor s\cdot W_0\rceil$ |
| $\mathbf{X} \in [0,1]^{H_s\times W_s\times 3}$ | Tensore sRGB dopo downsampling |
| $\hat{\mathbf{X}} \in \mathbb{R}^{B\times 3\times H_s\times W_s}$ | Tensore normalizzato ImageNet, input al modello (fp16) |
| $I \in [0,1]^{H \times W \times 3}$ | Immagine RGB normalizzata (dimensioni generiche) |
| $I_{Lab} \in \mathbb{R}^{H \times W \times 3}$ | Immagine nello spazio CIE L\*a\*b\* |
| $I^{src}, I^{tgt}, I^{pred}$ | Immagine sorgente, target (ground truth), predetta |
| $G \in \mathbb{R}^{12 \times H_g \times W_g \times L_b}$ | Bilateral grid |
| $\alpha \in [0,1]^{H \times W}$ | Confidence mask spaziale |
| $\mathbf{s} \in \mathbb{R}^{256}$ | Style prototype del fotografo |
| $\boldsymbol{\delta}_i \in \mathbb{R}^{192}$ | Edit delta della coppia $i$-esima |
| $K = 20$ | Numero di coppie nel subset fisso per il cross-attention |
| $P_k \in \mathbb{R}^{B \times C_k \times \lfloor H_s/2^k\rceil \times \lfloor W_s/2^k\rceil}$ | Feature map alla scala $k$ (risoluzione dipende dall'input) |
| $T = \lfloor H_s/32\rceil \cdot \lfloor W_s/32\rceil$ | Numero di token in $P_5$ (varia con la risoluzione) |
| $\phi_l(\cdot)$ | Feature map al layer $l$ di VGG16 (frozen) |
| $\mathbf{G}_l(I)$ | Matrice di Gram delle feature al layer $l$ |
| $\mathcal{L}(\cdot, \cdot)$ | Funzione di loss generica |
| $\lambda_\bullet$ | Peso (iperparametro) del termine di loss corrispondente |
| $\Delta E_{00}(\cdot,\cdot)$ | Distanza cromatica CIEDE2000 tra due colori Lab |
| $C^* = \sqrt{(a^*)^2 + (b^*)^2}$ | Chroma (saturazione) nel colore Lab |
| $h^* = \text{atan2}(b^*, a^*)$ | Angolo di hue nel piano $(a^*, b^*)$ |
| $\sigma(\cdot)$ | Funzione sigmoide $\sigma(x) = (1+e^{-x})^{-1}$ |
| $\delta(\cdot)$ | Funzione ReLU $\delta(x) = \max(0,x)$ |
| $\odot$ | Prodotto di Hadamard (elemento per elemento) |
| $[\cdot;\cdot]$ | Concatenazione lungo la dimensione dei canali |
| $\|\cdot\|_F$ | Norma di Frobenius |
| $\lfloor x \rceil$ | Arrotondamento all'intero più vicino |

---

### 6.2 Pipeline di Pre-elaborazione RAW: da Sensore a Tensore Normalizzato

Prima che qualsiasi componente neurale elabori l'immagine, i dati grezzi del sensore devono essere trasformati in un tensore di valori linearmente proporzionali all'irradianza della scena. Questa pipeline è deterministica, differenziabile quasi ovunque, e — crucialmente — **resolution-agnostic**: tutte le operazioni sono formulate in termini di coordinate normalizzate o operazioni locali, indipendentemente dalla risoluzione fisica del sensore.

Sia $\mathcal{S}$ il sensore con dimensioni fisiche $(H_{raw} \times W_{raw})$ pixel, pattern di Color Filter Array $\mathcal{P}$ (tipicamente Bayer RGGB), e bit depth $b$ (tipicamente 10–14 bit). Ogni sensore produce un'immagine in scala di grigi $\mathbf{R} \in \{0, 1, \ldots, 2^b-1\}^{H_{raw} \times W_{raw}}$ dove ogni pixel misura l'irradianza attraverso un singolo filtro cromatico.

---

#### 6.2.1 Lettura e Linearizzazione del Segnale RAW

Il sensore introduce due artefatti sistematici che devono essere corretti prima di ogni altra operazione: il **livello di nero** (dark current) e il **livello di saturazione** (full well capacity).

Sia $\mathbf{R} \in \mathbb{Z}^{H_{raw}\times W_{raw}}$ la lettura grezza del sensore. I parametri $b_{dark} \in \mathbb{R}$ (livello di nero, tipicamente 512–1024 per sensori a 14 bit) e $b_{sat} \in \mathbb{R}$ (livello di saturazione, tipicamente $2^{14}-1 = 16383$) sono metadata presenti nell'header del file RAW (DNG, CR3, NEF, ARW, ecc.).

La **normalizzazione lineare** è:

$$R_{norm}(i,j) = \frac{\mathbf{R}(i,j) - b_{dark}}{b_{sat} - b_{dark}} \in [0, 1]$$

con clipping implicito: $R_{norm}(i,j) \leftarrow \max(0,\, \min(1,\, R_{norm}(i,j)))$. Dopo questa operazione, $R_{norm}(i,j) = 0$ rappresenta assenza di luce (ombra pura) e $R_{norm}(i,j) = 1$ rappresenta saturazione del sensore (blow-out).

**Proprietà fondamentale.** Il segnale $R_{norm}(i,j)$ è **linearmente proporzionale all'irradianza della scena** $E(i,j)$:

$$R_{norm}(i,j) \approx k \cdot E(i,j) \cdot t_{exp}$$

dove $k$ è la responsività del sensore e $t_{exp}$ è il tempo di esposizione. Questa linearità è la proprietà che distingue i dati RAW dai formati già processati come JPEG — e la ragione per cui il color grading operando su RAW ha accesso alla piena gamma dinamica della scena.

---

#### 6.2.2 Stima e Sottrazione del Rumore (Modello Poisson-Gaussiano)

Il segnale $R_{norm}$ è corrotto da rumore di due nature:

- **Rumore di shot (Poisson):** proporzionale al segnale
- **Rumore di lettura (Gaussiano):** indipendente dal segnale

Il modello combinato **Poisson-Gaussiano** per la varianza del rumore al pixel $(i,j)$ è:

$$\sigma^2_{noise}(i,j) = \underbrace{\alpha \cdot R_{norm}(i,j)}_{\text{shot noise (Poisson)}} + \underbrace{\sigma^2_{read}}_{\text{read noise (Gaussian)}}$$

con $\alpha$ (gain factor, fornito dai metadati EXIF come funzione dell'ISO) e $\sigma^2_{read}$ (varianza di lettura, caratteristica del sensore).

La normalizzazione per Signal-to-Noise Ratio atteso è:

$$\tilde{R}_{norm}(i,j) = \frac{R_{norm}(i,j)}{\sqrt{\sigma^2_{noise}(i,j)} + \varepsilon}$$

Questa operazione è differenziabile rispetto a $R_{norm}$ e rende la distribuzione del segnale più uniforme tra immagini ad alto e basso ISO.

---

#### 6.2.3 Pattern Bayer e Demosaicatura

Il Color Filter Array (CFA) Bayer di un sensore a colori standard dispone i filtri in una griglia $2\times 2$ ripetuta:

$$\text{Pattern RGGB}: \quad \begin{pmatrix} R & G \\ G & B \end{pmatrix} \text{ ripetuto su } (H_{raw}/2) \times (W_{raw}/2) \text{ blocchi}$$

La maschera di selezione del filtro per il pattern RGGB è la funzione $c: \mathbb{Z}^2 \to \{R, G_1, G_2, B\}$:

$$c(i,j) = \begin{cases} R & i \equiv 0 \pmod 2,\ j \equiv 0 \pmod 2 \\ G_1 & i \equiv 0 \pmod 2,\ j \equiv 1 \pmod 2 \\ G_2 & i \equiv 1 \pmod 2,\ j \equiv 0 \pmod 2 \\ B & i \equiv 1 \pmod 2,\ j \equiv 1 \pmod 2 \end{cases}$$

La **demosaicatura** (demosaicing) stima i valori mancanti degli altri due canali per ogni pixel, producendo un'immagine a tre canali.

**Algoritmo AHD (Adaptive Homogeneity-Directed).** L'algoritmo AHD procede in tre fasi:

**Fase 1 — Interpolazione iniziale del verde.** Il canale verde $G$ ha doppia densità rispetto a R e B ed è quindi interpolato per primo:

$$\hat{G}(i,j) = \frac{1}{4}\bigl[R_{norm}(i,j-1) + R_{norm}(i,j+1) + R_{norm}(i-1,j) + R_{norm}(i+1,j)\bigr] \quad \text{se } c(i,j) \in \{R, B\}$$

**Fase 2 — Interpolazione di R e B tramite differenze.** AHD utilizza la differenza $D_c(i,j) = E^{c}(i,j) - G(i,j)$:

$$\hat{D}_R(i,j) = \text{Interp2D}\!\left(\{R_{norm}(i',j') - \hat{G}(i',j') : c(i',j')=R\},\ (i,j)\right)$$

$$\hat{D}_B(i,j) = \text{Interp2D}\!\left(\{R_{norm}(i',j') - \hat{G}(i',j') : c(i',j')=B\},\ (i,j)\right)$$

I valori finali sono:

$$\hat{E}^{full}_R(i,j) = \hat{D}_R(i,j) + \hat{G}(i,j), \quad \hat{E}^{full}_B(i,j) = \hat{D}_B(i,j) + \hat{G}(i,j)$$

**Fase 3 — Selezione adattiva della direzione:**

$$\hat{E}^{full}_c(i,j) = \begin{cases} \hat{E}^{H}_c(i,j) & \text{se } H(i,j) \leq V(i,j) \\ \hat{E}^{V}_c(i,j) & \text{altrimenti} \end{cases}$$

dove $H(i,j)$ e $V(i,j)$ misurano la variazione locale nelle direzioni orizzontale e verticale. L'output è $\hat{\mathbf{E}}^{full} \in [0,1]^{H_{raw}\times W_{raw}\times 3}$.

---

#### 6.2.4 Correzione Lens: Vignettatura e Aberrazione Cromatica

**Correzione della vignettatura.** Il profilo radiale normalizzato è:

$$r(i,j) = \frac{1}{r_{max}}\sqrt{\left(i - \frac{H_{raw}}{2}\right)^2 + \left(j - \frac{W_{raw}}{2}\right)^2}, \quad r_{max} = \sqrt{\left(\frac{H_{raw}}{2}\right)^2 + \left(\frac{W_{raw}}{2}\right)^2}$$

Il profilo di vignettatura $v(r) = 1 + k_2 r^2 + k_4 r^4 + k_6 r^6$ (con $k_{2k} < 0$ tipicamente). La correzione è:

$$\hat{E}^{vgn}_c(i,j) = \frac{\hat{E}^{full}_c(i,j)}{v(r(i,j))}, \quad c \in \{R,G,B\}$$

Le coordinate normalizzate rendono il profilo **resolution-agnostic**: si applica identicamente a qualsiasi risoluzione fisica.

**Correzione dell'aberrazione cromatica laterale.** Remappatura radiale invertita per canale:

$$\hat{E}^{ca}_c(i,j) = \hat{E}^{vgn}_c\!\left(\rho_c(i,j)\right), \quad c \in \{R,B\}$$

dove $\rho_c(i,j) = (1 + \Delta r_c \cdot r^2(i,j)) \cdot (i,j)$ con $|\Delta r_c| < 0.01$, implementata tramite interpolazione bilineare.

---

#### 6.2.5 Bilanciamento del Bianco (White Balance)

Sia $\mathbf{w}_{wb} = (w_R, w_G, w_B) \in \mathbb{R}^3_{>0}$ il vettore di guadagni dai metadati EXIF. La correzione è:

$$\hat{E}^{wb}_c(i,j) = w_c \cdot \hat{E}^{ca}_c(i,j), \quad c \in \{R,G,B\}$$

con $w_G = 1$ per convenzione e clipping a $[0,1]$ dopo la moltiplicazione.

---

#### 6.2.6 Conversione dallo Spazio Colore del Sensore a XYZ

La **camera matrix** $\mathbf{M}_{cam\to XYZ} \in \mathbb{R}^{3\times 3}$ dai metadati DNG converte da Camera RGB a CIE XYZ D50:

$$\begin{pmatrix} X \\ Y \\ Z \end{pmatrix}_{D50} = \mathbf{M}_{cam\to XYZ} \cdot \begin{pmatrix} \hat{E}^{wb}_R \\ \hat{E}^{wb}_G \\ \hat{E}^{wb}_B \end{pmatrix}$$

Per convertire da XYZ D50 a XYZ D65:

$$\begin{pmatrix} X \\ Y \\ Z \end{pmatrix}_{D65} = \mathbf{M}_{Bradford} \cdot \begin{pmatrix} X \\ Y \\ Z \end{pmatrix}_{D50}$$

---

#### 6.2.7 Conversione XYZ → sRGB Lineare

$$\begin{pmatrix} R_{lin} \\ G_{lin} \\ B_{lin} \end{pmatrix} = \mathbf{M}_{XYZ\to sRGB} \cdot \begin{pmatrix} X_{D65} \\ Y_{D65} \\ Z_{D65} \end{pmatrix}, \quad \mathbf{M}_{XYZ\to sRGB} = \begin{pmatrix} 3.2406 & -1.5372 & -0.4986 \\ -0.9689 & 1.8758 & 0.0415 \\ 0.0557 & -0.2040 & 1.0570 \end{pmatrix}$$

Clipping al range valido: $R_{lin}, G_{lin}, B_{lin} \leftarrow \text{clip}(\cdot, 0, 1)$.

---

#### 6.2.8 Gamma Encoding: sRGB Lineare → sRGB Standard

La **funzione di trasferimento sRGB** per ogni canale $c$ è:

$$I^{sRGB}_c = \gamma_{sRGB}(I^{lin}_c) = \begin{cases} 12.92 \cdot I^{lin}_c & I^{lin}_c \leq 0.0031308 \\ 1.055 \cdot (I^{lin}_c)^{1/2.4} - 0.055 & I^{lin}_c > 0.0031308 \end{cases}$$

**Inversione (OETF):**

$$I^{lin}_c = \gamma_{sRGB}^{-1}(I^{sRGB}_c) = \begin{cases} I^{sRGB}_c / 12.92 & I^{sRGB}_c \leq 0.04045 \\ \left(\frac{I^{sRGB}_c + 0.055}{1.055}\right)^{2.4} & I^{sRGB}_c > 0.04045 \end{cases}$$

---

#### 6.2.9 Riduzione di Risoluzione Adattiva per il Training

La risoluzione di training è $512 \times 384$ pixel (lato lungo $L_{train} = 512$), scelta per garantire batch size 4–6 con fp16 e gradient checkpointing su RTX 3080 (10 GB VRAM). Questa risoluzione è sufficiente a preservare le relazioni cromatiche globali e le texture di media frequenza rilevanti per il color grading, mantenendo un margine di VRAM per il backward pass.

**Definizione del fattore di scala.** Sia $(H_0, W_0)$ la dimensione nativa con $H_0 \leq W_0$. Il fattore di scala uniforme è:

$$s = \frac{L_{train}}{\max(H_0, W_0)} = \frac{512}{\max(H_0, W_0)} \in (0, 1]$$

Le dimensioni ridotte sono:

$$H_s = \lfloor s \cdot H_0 \rceil, \quad W_s = \lfloor s \cdot W_0 \rceil$$

**Nota sulle dimensioni di training.** Con $L_{train} = 512$ e aspect ratio $4:3$ tipico di fotocamere moderne, si ottiene $H_s = 384$, $W_s = 512$, da cui $T(H_s, W_s) = \lfloor 384/32 \rfloor \cdot \lfloor 512/32 \rfloor = 12 \cdot 16 = 192$ token in $P_5$ — un numero molto gestibile per il cross-attention con $K=20$.

**Downsampling con anti-aliasing.** Il filtro di Lanczos con parametro $a = 3$ (standard fotografico) è:

$$L(x) = \begin{cases} \text{sinc}(x)\,\text{sinc}(x/a) & |x| < a \\ 0 & |x| \geq a \end{cases}, \quad \text{sinc}(x) = \frac{\sin(\pi x)}{\pi x}$$

La funzione di ricampionamento per la coordinata output $(i_s, j_s)$ è:

$$I^{sRGB}_c(i_s, j_s) = \sum_{i'} \sum_{j'} I^{sRGB}_c(i', j') \cdot L\!\left(\frac{i'}{s} - i_s\right) \cdot L\!\left(\frac{j'}{s} - j_s\right)$$

**Coerenza della pair $(I^{src}_s, I^{tgt}_s)$.** Il target JPEG deve essere ricampionato con lo **stesso identico** $s$, le stesse dimensioni $(H_s, W_s)$ e lo stesso filtro Lanczos.

---

#### 6.2.10 Normalizzazione al Tensore di Ingresso del Modello

L'output dell'intera pipeline pre-neural è il tensore $\mathbf{X} = I^{sRGB}_s \in [0,1]^{H_s \times W_s \times 3}$, normalizzato con le statistiche di ImageNet:

$$\hat{X}_c(i,j) = \frac{X_c(i,j) - \mu_c^{IN}}{\sigma_c^{IN}}, \quad \mathbf{\mu}^{IN} = (0.485, 0.456, 0.406), \quad \boldsymbol{\sigma}^{IN} = (0.229, 0.224, 0.225)$$

Il tensore finale $\hat{\mathbf{X}} \in \mathbb{R}^{B \times 3 \times H_s \times W_s}$ (formato PyTorch, canali prima) viene immediatamente convertito in fp16 prima di entrare nel CNN stem.

---

#### 6.2.11 Riepilogo della Pipeline RAW → Tensore e Invarianza alla Risoluzione

La pipeline completa è:

$$\mathbf{R} \xrightarrow{\text{(1) linearize}} R_{norm} \xrightarrow{\text{(2) noise}} \tilde{R}_{norm} \xrightarrow{\text{(3) demosaic}} \hat{\mathbf{E}}^{full} \xrightarrow{\text{(4) lens corr.}} \hat{\mathbf{E}}^{ca} \xrightarrow{\text{(5) WB}} \hat{\mathbf{E}}^{wb} \xrightarrow{\text{(6) cam matrix}} \mathbf{XYZ} \xrightarrow{\text{(7) sRGB lin.}} \mathbf{I}^{lin} \xrightarrow{\text{(8) gamma}} \mathbf{I}^{sRGB} \xrightarrow{\text{(9) downsample}} \mathbf{X} \xrightarrow{\text{(10) normalize}} \hat{\mathbf{X}}$$

**Teorema 6 (Invarianza alla Risoluzione della Pipeline).** Sia $f_\theta: \mathbb{R}^{B\times 3\times H\times W} \to \mathbb{R}^{B\times 3\times H\times W}$ il modello HybridStyleNet. Il modello è resolution-agnostic nel senso seguente: per qualsiasi coppia di risoluzioni $(H_1, W_1)$ e $(H_2, W_2)$ con $H_1/W_1 = H_2/W_2$ (stesso aspect ratio), la predizione a risoluzione $H_1\times W_1$ è uguale alla predizione a risoluzione $H_2\times W_2$ ricampionata a $H_1\times W_1$, a meno di errori di ricampionamento $O(1/\min(H_1,H_2))$.

*Argomentazione.*

(a) Le **convoluzioni** in MobileNetV3-Small operano con kernel locali: la loro risposta dipende solo dai pattern locali, non dalla risoluzione assoluta.

(b) Il **Swin Transformer con RoPE** codifica solo le distanze relative tra token, non le posizioni assolute. Cambiare la risoluzione dell'input cambia il numero di token $T = H_s W_s / (32^2)$, ma non la semantica delle rappresentazioni.

(c) La **bilateral grid** è parametrizzata in coordinate normalizzate $x_g(j) = \frac{j}{W-1}(W_g-1)$: la stessa trasformazione cromatica viene applicata alla stessa posizione relativa indipendentemente dalla risoluzione assoluta.

(d) La **confidence mask** e il **blending finale** sono operazioni pixel-wise e di interpolazione bilineare, invarianti per costruzione.

*Implicazione pratica.* Il modello viene addestrato su crop $512\times 384$ per efficienza di memoria, ma all'inferenza opera su immagini a risoluzione piena ($6048\times 4024$ o qualsiasi altra dimensione), senza alcuna modifica architetturale. $\square$

---

### 6.3 Bilateral Grid e Slicing

#### 6.3.1 Struttura della Grid e Motivazione

La bilateral grid è la struttura dati fondamentale del rendering differenziabile. Sia $G \in \mathbb{R}^{12 \times H_g \times W_g \times L_b}$ una griglia tridimensionale dove:

- $H_g \times W_g$ è la risoluzione spaziale (globale: $8 \times 8$; locale: $16 \times 16$)
- $L_b = 8$ è il numero di bin lungo la dimensione della luminanza
- Il fattore 12 rappresenta i coefficienti di una trasformazione affine $3 \times 3 + 3$

Ogni cella $(x, y, l)$ della griglia contiene:

$$G(x,y,l) = \bigl[\mathbf{A}(x,y,l),\ \mathbf{b}(x,y,l)\bigr] \in \mathbb{R}^{12}$$

con $\mathbf{A}(x,y,l) \in \mathbb{R}^{3\times 3}$ matrice di trasformazione cromatica e $\mathbf{b}(x,y,l) \in \mathbb{R}^3$ bias additivo.

**Scelta della risoluzione $16\times16$ per la grid locale.** La grid locale a $16\times16$ è il compromesso selezionato tra la grid globale $8\times8$ e la grid fine $32\times32$. Essa garantisce una granularità spaziale sufficiente a distinguere le principali zone semantiche di una foto (cielo, volto, mezzitoni, ombre) senza il costo di memoria e tempo computazionale di $32\times32$. In termini pratici, ogni cella di una grid $16\times16$ copre $\frac{H}{16}\times\frac{W}{16}$ pixel — su un'immagine $3000\times2000$, circa $187\times125$ pixel per cella, ovvero zone ben distinte di soggetto, sfondo e bordi.

**Intuizione geometrica.** La grid discretizza lo spazio $(x, y, g)$ dove $x,y$ sono le coordinate spaziali e $g$ è la luminanza del pixel. A ogni punto di questo spazio tridimensionale è associata una trasformazione affine diversa: pixel in posizioni diverse e/o con luminanze diverse ricevono trattamenti cromatici distinti.

#### 6.3.2 Guida di Luminanza e Coordinate nella Grid

Per ogni pixel $(i,j)$ dell'immagine sorgente $I_{src}$, la guida di luminanza è:

$$g(i,j) = 0.299\, I_R(i,j) + 0.587\, I_G(i,j) + 0.114\, I_B(i,j) \in [0,1]$$

Le coordinate nella griglia:

$$x_g(j) = \frac{j}{W-1}(W_g - 1) \in [0, W_g-1], \quad y_g(i) = \frac{i}{H-1}(H_g - 1) \in [0, H_g-1], \quad l_g(i,j) = g(i,j)\cdot(L_b - 1) \in [0, L_b-1]$$

#### 6.3.3 Interpolazione Trilineare

$$[\mathbf{A}_{ij},\,\mathbf{b}_{ij}] = \sum_{p \in \{0,1\}} \sum_{q \in \{0,1\}} \sum_{r \in \{0,1\}} w_{pqr}(i,j)\cdot G\!\bigl(\lfloor x_g \rfloor + p,\, \lfloor y_g \rfloor + q,\, \lfloor l_g \rfloor + r\bigr)$$

con pesi trilineari $w_{pqr}(i,j) = w_p^x \cdot w_q^y \cdot w_r^l$ e $w_0^x = 1 - (x_g - \lfloor x_g \rfloor)$, $w_1^x = x_g - \lfloor x_g \rfloor$ (analogamente per $y, l$). I pesi sommano a 1. L'interpolazione trilineare è differenziabile quasi ovunque — condizione sufficiente per il training con SGD.

#### 6.3.4 Applicazione della Trasformazione Affine

$$I'(i,j) = \mathbf{A}_{ij}\cdot I(i,j) + \mathbf{b}_{ij}$$

**Proprietà edge-aware.** Due pixel con luminanza molto simile ricevono quasi la stessa trasformazione cromatica indipendentemente dalla loro distanza spaziale. Viceversa, due pixel vicini ma con luminanza molto diversa ricevono trasformazioni potenzialmente molto diverse — evitando l'alone (halo) ai bordi.

---

### 6.4 Forward Pass Completo e Resolution-Agnostic

Il forward pass descrive la trasformazione end-to-end $\mathbf{R} \mapsto I^{pred}$, che unisce la pipeline RAW (sezione 6.2) con l'elaborazione neurale. Tutte le dimensioni intermedie sono espresse in termini simbolici di $(H, W)$.

#### Propagazione delle dimensioni

Sia $I^{src} = \hat{\mathbf{X}} \in \mathbb{R}^{B \times 3 \times H \times W}$ il tensore in ingresso (in fp16). Le dimensioni alle varie scale sono:

| Stage | Tensore | Dimensione | Stride cumulativo |
|-------|---------|-----------|-------------------|
| Input | $I^{src}$ | $B \times 3 \times H \times W$ | $1$ |
| CNN stage 1 | $P_1$ | $B \times 16 \times H_1 \times W_1$ | $2$ |
| CNN stage 2 | $P_2$ | $B \times 24 \times H_2 \times W_2$ | $4$ |
| CNN stage 3 | $P_3$ | $B \times 48 \times H_3 \times W_3$ | $8$ |
| Swin stage 4 | $P_4$ | $B \times 96 \times H_4 \times W_4$ | $16$ |
| Swin stage 5 | $P_5$ | $B \times 192 \times H_5 \times W_5$ | $32$ |

dove $H_k = \lfloor H/2^k \rfloor$ e $W_k = \lfloor W/2^k \rfloor$. Il numero di token in $P_5$ è:

$$T(H,W) = H_5 \cdot W_5 = \left\lfloor \frac{H}{32} \right\rfloor \cdot \left\lfloor \frac{W}{32} \right\rfloor$$

A titolo illustrativo: $T(512, 384) = 16 \cdot 12 = 192$ (training crop), $T(3000, 2000) \approx 93 \cdot 62 = 5{,}766$ (risoluzione professionale tipica), $T(6048, 4024) \approx 189 \cdot 125 = 23{,}625$ (sensore 24 MP a piena risoluzione).

#### Step 0 — Pipeline RAW → Tensore (sezione 6.2)

$$\mathbf{R} \in \mathbb{Z}^{H_{raw} \times W_{raw}} \xrightarrow{\text{§6.2.1–6.2.9}} I^{src} \in \mathbb{R}^{B \times 3 \times H \times W}\ \text{(fp16)}$$

con $H = 384$, $W = 512$ durante il training; $s = 1$ (nessun downsampling) durante l'inferenza a risoluzione piena.

#### Step 1 — CNN Stem: MobileNetV3-Small Stage 1–3 (fp16)

$$P_3 = \mathcal{F}^{(3)} \circ \mathcal{F}^{(2)} \circ \mathcal{F}^{(1)}(I^{src}) \in \mathbb{R}^{B \times 48 \times H_3 \times W_3}\ \text{(fp16)}$$

Ogni stage $\mathcal{F}^{(k)}$ è una composizione di blocchi bneck con stride 2 (depthwise separable conv + SE + BN + Hard-Swish), producendo uno stride cumulativo $2^k$. Il receptive field effettivo al termine dello stage 3 è $\approx 20$–$30$ pixel. **Gradient checkpointing** attivo: i tensori intermedi vengono ricalcolati durante il backward pass.

#### Step 2 — Swin Transformer Stage 4–5 con RoPE (fp16)

$$P_4 = \text{SwinStage}_4(P_3;\, \text{RoPE}) \in \mathbb{R}^{B \times 96 \times H_4 \times W_4}\ \text{(fp16)}$$

$$P_5 = \text{SwinStage}_5(P_4;\, \text{RoPE}) \in \mathbb{R}^{B \times 192 \times H_5 \times W_5}\ \text{(fp16)}$$

Ogni stage Swin applica W-MSA alternato a SW-MSA con finestre $M \times M = 7 \times 7$ e **4 teste di attenzione**. La complessità è $O(T(H,W) \cdot M^2 \cdot d)$ — lineare in $(H,W)$. Con RoPE il prodotto $\mathbf{q}_m^T \mathbf{k}_n = g(\mathbf{q}, \mathbf{k}, m-n)$ dipende solo dalla distanza relativa: il modello generalizza a qualsiasi risoluzione senza riaddestrare. Zero-padding simmetrico garantisce finestre complete quando $H_5$ o $W_5$ non sono multipli di $M$.

#### Step 3 — Style Prototype (calcolato una sola volta per fotografo, poi cached)

L'encoder $\text{Enc}: \mathbb{R}^{B\times 3\times H\times W} \to \mathbb{R}^{192}$ (CNN stem + GAP) è risoluzione-agnostico:

$$\text{GAP}(P_3) = \frac{1}{H_3 W_3}\sum_{i=1}^{H_3}\sum_{j=1}^{W_3} P_3(\cdot,\cdot,i,j) \in \mathbb{R}^{B \times 48} \xrightarrow{\text{Linear}} \mathbb{R}^{192}$$

Le edit delta e il prototype (in fp16):

$$\boldsymbol{\delta}_i = \text{Enc}(I_i^{tgt}) - \text{Enc}(I_i^{src}) \in \mathbb{R}^{192}, \quad i = 1,\ldots,N$$

$$\mathbf{s} = \text{SetTransformer}_{4\text{ teste}}\!\left(\{\boldsymbol{\delta}_i\}_{i=1}^N\right) \in \mathbb{R}^{256}$$

#### Step 4 — Cross-Attention Contestuale (K=20, fp16)

Sia $\mathbf{Z} = \text{Flatten}(P_5) \in \mathbb{R}^{T(H,W) \times 192}$. Le chiavi e valori del subset fisso ($K=20$ coppie, cached):

$$\mathbf{K}_{tr} \in \mathbb{R}^{20 \times 192}, \qquad \mathbf{V}_{tr} \in \mathbb{R}^{20 \times 192}$$

Le proiezioni lineari:

$$\mathbf{Q} = \mathbf{Z}\mathbf{W}^Q \in \mathbb{R}^{T(H,W) \times 192}, \quad \mathbf{K} = \mathbf{K}_{tr}\mathbf{W}^K \in \mathbb{R}^{20 \times 192}, \quad \mathbf{V} = \mathbf{V}_{tr}\mathbf{W}^V \in \mathbb{R}^{20 \times 192}$$

$$\mathbf{C} = \text{Softmax}\!\!\left(\frac{\mathbf{Q}\mathbf{K}^T}{\sqrt{192}}\right)\!\mathbf{V} \in \mathbb{R}^{T(H,W) \times 192}$$

La matrice di attenzione $\mathbf{A} \in \mathbb{R}^{T \times 20}$ è compatta e computazionalmente leggera grazie a $K=20$ fissato. Il tensore $\mathbf{C}$ viene risagomato a $\mathbb{R}^{B \times 192 \times H_5 \times W_5}$ e proiettato a 128 canali con $\text{Conv}_{1\times1}$ (per la compatibilità con SPADE):

$$\mathbf{C}_3 = \text{BilinearUp}\!\left(\text{Conv}_{1\times1}^{192\to128}(\mathbf{C}),\; (H_3, W_3)\right) \in \mathbb{R}^{B \times 128 \times H_3 \times W_3}$$

$$\mathbf{C}_4 = \text{BilinearUp}\!\left(\text{Conv}_{1\times1}^{192\to128}(\mathbf{C}),\; (H_4, W_4)\right) \in \mathbb{R}^{B \times 128 \times H_4 \times W_4}$$

**Upsampling bilineare.** L'operazione $\text{BilinearUp}(\mathbf{F}, (H', W'))$ per la posizione di output $(i', j')$ calcola la posizione corrispondente nella feature map:

$$u = i' \cdot \frac{H_{in} - 1}{H' - 1} \in [0, H_{in}-1], \qquad v = j' \cdot \frac{W_{in} - 1}{W' - 1} \in [0, W_{in}-1]$$

$$\text{BilinearUp}(\mathbf{F})_{c,i',j'} = (1-\Delta u)(1-\Delta v)\,F_{c,\lfloor u\rfloor,\lfloor v\rfloor} + \Delta u(1-\Delta v)\,F_{c,\lfloor u\rfloor+1,\lfloor v\rfloor} + (1-\Delta u)\Delta v\,F_{c,\lfloor u\rfloor,\lfloor v\rfloor+1} + \Delta u\,\Delta v\,F_{c,\lfloor u\rfloor+1,\lfloor v\rfloor+1}$$

con $\Delta u = u - \lfloor u\rfloor \in [0,1)$. Le coordinate normalizzate garantiscono che l'upsampling sia indipendente dalla risoluzione assoluta.

#### Step 5 — Bilateral Grid Predictions

Le bilateral grid hanno risoluzione spaziale **fissa** indipendentemente dalla risoluzione dell'input.

$$G_{global} = \text{GlobalBranch}\!\left(\text{AdaIN}(P_5,\, \mathbf{s})\right) \in \mathbb{R}^{B \times 12 \times 8 \times 8 \times L_b}$$

$$\mathbf{f}_{global} = \frac{1}{H_5 W_5}\sum_{i,j}\text{AdaIN}(P_5, \mathbf{s})_{i,j} \in \mathbb{R}^{B \times 192}$$

$$G_{global} = \text{reshape}\!\left(\mathbf{W}_{gb,2}\,\delta(\mathbf{W}_{gb,1}\,\mathbf{f}_{global})\right) \in \mathbb{R}^{B \times 12 \times 8 \times 8 \times L_b}$$

$$\mathbf{x}_4 = \text{SPADEResBlock}(P_4,\, \mathbf{C}_4) \in \mathbb{R}^{B \times 96 \times H_4 \times W_4}$$

$$\mathbf{x}_3 = \text{SPADEResBlock}\!\left(\text{BilinearUp}(\mathbf{x}_4, (H_3, W_3)) + P_3,\, \mathbf{C}_3\right) \in \mathbb{R}^{B \times 96 \times H_3 \times W_3}$$

$$G_{local} = \text{reshape}\!\left(\text{Conv}_{1\times 1}\!\left(\text{AdaptiveAvgPool}(\mathbf{x}_3,\; (16, 16))\right)\right) \in \mathbb{R}^{B \times 12 \times 16 \times 16 \times L_b}$$

La confidence mask:

$$\alpha = \sigma\!\left(\text{Conv}_{1\times 1}\!\left(\delta\!\left(\text{Conv}_{3\times 3}([\tilde{P}_4;\, P_3])\right)\right)\right) \in [0,1]^{B \times 1 \times H_3 \times W_3}$$

$$\alpha^{full} = \text{BilinearUp}(\alpha,\; (H, W)) \in [0,1]^{B \times 1 \times H \times W}$$

con $\tilde{P}_4 = \text{BilinearUp}(P_4, (H_3, W_3))$.

#### Step 6 — Bilateral Slicing Globale a Risoluzione Piena

Per ogni pixel $(i,j) \in \{0,\ldots,H-1\} \times \{0,\ldots,W-1\}$, guida di luminanza e coordinate nella grid globale $8\times 8\times L_b$:

$$g(i,j) = 0.299\,I^{src}_R(i,j) + 0.587\,I^{src}_G(i,j) + 0.114\,I^{src}_B(i,j)$$

$$x_g^{glob}(j) = \frac{j}{W-1}\cdot 7, \quad y_g^{glob}(i) = \frac{i}{H-1}\cdot 7, \quad l_g(i,j) = g(i,j)\cdot(L_b - 1)$$

$$\left[\mathbf{A}^{glob}_{ij},\,\mathbf{b}^{glob}_{ij}\right] = \text{TrilinearInterp}\!\left(G_{global},\; x_g^{glob}(j),\; y_g^{glob}(i),\; l_g(i,j)\right)$$

$$I_g(i,j) = \mathbf{A}^{glob}_{ij}\cdot I^{src}(i,j) + \mathbf{b}^{glob}_{ij}$$

#### Step 7 — Bilateral Slicing Locale a Risoluzione Piena

Analogamente per la grid locale $16\times 16\times L_b$:

$$x_g^{loc}(j) = \frac{j}{W-1}\cdot 15, \quad y_g^{loc}(i) = \frac{i}{H-1}\cdot 15$$

$$\left[\mathbf{A}^{loc}_{ij},\,\mathbf{b}^{loc}_{ij}\right] = \text{TrilinearInterp}\!\left(G_{local},\; x_g^{loc}(j),\; y_g^{loc}(i),\; l_g(i,j)\right)$$

$$I_l(i,j) = \mathbf{A}^{loc}_{ij}\cdot I_g(i,j) + \mathbf{b}^{loc}_{ij}$$

La composizione $I_l = \text{BilSlice}_{local}(\text{BilSlice}_{global}(I^{src}))$ realizza una pipeline di due trasformazioni affini dipendenti dalla scena: la prima stabilisce il colore globale (mood, WB, cast), la seconda raffina per zone semantiche (skin, cielo, ombre, mezzitoni).

#### Step 8 — Blending con Confidence Mask

$$I_{out}(i,j) = \alpha^{full}(i,j)\cdot I_l(i,j) + \bigl(1 - \alpha^{full}(i,j)\bigr)\cdot I_g(i,j)$$

#### Step 9 — Clipping e Gamma Re-encoding

$$I^{pred}_{lin}(i,j) = \text{clip}\!\left(I_{out}(i,j),\; 0,\; 1\right)$$

$$I^{pred}(i,j) = \gamma_{sRGB}\!\left(I^{pred}_{lin}(i,j)\right)$$

Il clipping introduce non-differenziabilità dove $I_{out} \notin [0,1]$: questo insieme ha misura quasi zero dopo convergenza, e il gradiente è approssimato a zero (straight-through estimator implicito). Il reencoding gamma viene eseguito in fp32 per precisione (operazione scalare, costo trascurabile).

---

#### Riepilogo: Invarianza alla Risoluzione nel Forward Pass

| Componente | Dipende da $(H,W)$? | Perché è resolution-agnostic | Precisione |
|------------|---------------------|-------------------------------|------------|
| Pipeline RAW (§6.2) | ✅ produce $(H,W)$ | Coordinate normalizzate, Lanczos adattivo | fp32 |
| CNN stem MobileNetV3 | Input $(H,W)$, output $(H_3,W_3)$ | Conv locali, stride fissato | fp16 |
| Swin + RoPE (4 teste) | Token $T(H,W)$ variabile | RoPE dipende solo da distanza relativa | fp16 |
| GAP → Enc | ❌ output fisso $\mathbb{R}^{192}$ | Media su $(H_3,W_3)$ qualsiasi | fp16 |
| Cross-attn ($K=20$) | Token $T(H,W)$ variabile, chiavi fisse | $K$ fissato, chiavi cached | fp16 |
| $G_{global}$ | ❌ output fisso $8\times 8\times L_b$ | GAP elimina dimensioni spaziali | fp16 |
| $G_{local}$ | ❌ output fisso $16\times 16\times L_b$ | AdaptiveAvgPool a target fissato | fp16 |
| Bilateral slicing | Opera a $(H,W)$ qualsiasi | Coordinate normalizzate in $[0,1]^2$ | fp32 |
| Confidence mask | Opera a $(H,W)$ qualsiasi | BilinearUp da $(H_3,W_3)$ a $(H,W)$ | fp16 |
| Gamma re-encoding | Opera a $(H,W)$ qualsiasi | Operazione scalare | fp32 |

---

### 6.5 Funzione di Loss Composita: Color-Aesthetic Loss

#### 6.5.0 Motivazione e Inadeguatezza delle Loss Standard

La funzione di loss è il componente più critico del sistema. Le loss standard hanno difetti fondamentali per il task di color grading fotografico.

La **Mean Squared Error (MSE)** in RGB è non uniforme percettivamente, favorisce output "media" sfocata e pesa ugualmente errori in ombre e alte luci. La **MAE (L1)** ha gli stessi problemi percettivi. Nessuna delle due cattura la distribuzione spettrale globale dell'immagine né la sua struttura semantica multi-scala.

La **Color-Aesthetic Loss** proposta è:

$$\boxed{\mathcal{L} = \lambda_{\Delta E}\,\mathcal{L}_{\Delta E} + \lambda_{hist}\,\mathcal{L}_{hist} + \lambda_{perc}\,\mathcal{L}_{perc} + \lambda_{style}\,\mathcal{L}_{style} + \lambda_{cos}\,\mathcal{L}_{cos} + \lambda_{chroma}\,\mathcal{L}_{chroma} + \lambda_{id}\,\mathcal{L}_{id}}$$

con pesi $\lambda_{\Delta E} = 0.5,\ \lambda_{hist} = 0.3,\ \lambda_{perc} = 0.4,\ \lambda_{style} = 0.2,\ \lambda_{cos} = 0.15,\ \lambda_{chroma} = 0.2,\ \lambda_{id} = 0.5$.

**Nota su fp16 e calcolo della loss.** Il calcolo della loss viene eseguito in fp32 (i tensori di input vengono promossi da fp16 a fp32 prima di entrare nel grafo della loss), per evitare underflow nei termini logaritmici e nelle radici quadrate di CIEDE2000. Questo è lo standard per l'uso di PyTorch AMP con loss complesse.

---

#### 6.5.1 ΔE Loss (CIEDE2000) — Accuratezza Cromatica Percettiva

**Motivazione.** L'accuratezza cromatica percettiva misura quanto due colori differiscono nel percetto visivo umano. La misura standard industriale è CIEDE2000 ($\Delta E_{00}$), adottata dall'ICC per la gestione del colore professionale.

**Conversione RGB → CIE Lab.** La pipeline di conversione è:

$$I_{RGB} \xrightarrow{\text{linearize}} I_{lin} = \begin{cases} I_{RGB}/12.92 & I_{RGB} \leq 0.04045 \\ \left(\frac{I_{RGB}+0.055}{1.055}\right)^{2.4} & I_{RGB} > 0.04045 \end{cases}$$

$$\begin{pmatrix} X \\ Y \\ Z \end{pmatrix} = \mathbf{M}_{sRGB \to XYZ} \begin{pmatrix} I_R^{lin} \\ I_G^{lin} \\ I_B^{lin} \end{pmatrix}, \quad \mathbf{M}_{sRGB \to XYZ} = \begin{pmatrix} 0.4124 & 0.3576 & 0.1805 \\ 0.2126 & 0.7152 & 0.0722 \\ 0.0193 & 0.1192 & 0.9505 \end{pmatrix}$$

$$f(t) = \begin{cases} t^{1/3} & t > (6/29)^3 \\ \frac{1}{3}(29/6)^2\,t + 4/29 & t \leq (6/29)^3 \end{cases}$$

$$L^* = 116\,f(Y/Y_n) - 16, \quad a^* = 500\left(f(X/X_n) - f(Y/Y_n)\right), \quad b^* = 200\left(f(Y/Y_n) - f(Z/Z_n)\right)$$

con $(X_n, Y_n, Z_n) = (95.047, 100.000, 108.883)$ per D65.

**Formula CIEDE2000 completa.** Dati due colori $\mathbf{c}_1 = (L_1^*, a_1^*, b_1^*)$ e $\mathbf{c}_2 = (L_2^*, a_2^*, b_2^*)$:

*Passo 1:*
$$C_k^* = \sqrt{(a_k^*)^2 + (b_k^*)^2}, \quad \bar{C}^* = \frac{C_1^* + C_2^*}{2}$$

*Passo 2:*
$$G = 0.5\left(1 - \sqrt{\frac{(\bar{C}^*)^7}{(\bar{C}^*)^7 + 25^7}}\right), \quad a_k' = a_k^*(1 + G), \quad C_k' = \sqrt{(a_k')^2 + (b_k^*)^2}$$

*Passo 3:*
$$h_k' = \text{atan2}(b_k^*, a_k') \mod 2\pi$$

*Passo 4:*
$$\Delta L' = L_2^* - L_1^*, \quad \Delta C' = C_2' - C_1', \quad \Delta h' = \begin{cases} h_2' - h_1' & |h_2' - h_1'| \leq \pi \\ h_2' - h_1' + 2\pi & h_2' - h_1' < -\pi \\ h_2' - h_1' - 2\pi & h_2' - h_1' > \pi \end{cases}$$
$$\Delta H' = 2\sqrt{C_1' C_2'}\,\sin(\Delta h'/2)$$

*Passo 5:*
$$\bar{L}' = \frac{L_1^*+L_2^*}{2}, \quad \bar{C}' = \frac{C_1'+C_2'}{2}, \quad \bar{h}' = \begin{cases} \frac{h_1'+h_2'}{2} & |h_1'-h_2'| \leq \pi \\ \frac{h_1'+h_2'+2\pi}{2} & |h_1'-h_2'| > \pi,\ h_1'+h_2' < 2\pi \\ \frac{h_1'+h_2'-2\pi}{2} & \text{altrimenti} \end{cases}$$

*Passo 6:*
$$T = 1 - 0.17\cos(\bar{h}'-30°) + 0.24\cos(2\bar{h}') + 0.32\cos(3\bar{h}'+6°) - 0.20\cos(4\bar{h}'-63°)$$
$$S_L = 1 + 0.015\frac{(\bar{L}'-50)^2}{\sqrt{20+(\bar{L}'-50)^2}}, \quad S_C = 1 + 0.045\bar{C}', \quad S_H = 1 + 0.015\bar{C}'T$$

*Passo 7:*
$$R_C = 2\sqrt{\frac{(\bar{C}')^7}{(\bar{C}')^7 + 25^7}}, \quad \Delta\theta = 30°\exp\!\left(-\left(\frac{\bar{h}'-275°}{25°}\right)^2\right), \quad R_T = -R_C\sin(2\Delta\theta)$$

*Passo 8:*
$$\Delta E_{00}(\mathbf{c}_1, \mathbf{c}_2) = \sqrt{\left(\frac{\Delta L'}{S_L}\right)^2 + \left(\frac{\Delta C'}{S_C}\right)^2 + \left(\frac{\Delta H'}{S_H}\right)^2 + R_T\frac{\Delta C'}{S_C}\frac{\Delta H'}{S_H}}$$

**Loss:**

$$\mathcal{L}_{\Delta E} = \frac{1}{HW}\sum_{i=1}^{H}\sum_{j=1}^{W} \Delta E_{00}\!\left(I^{pred}_{Lab}(i,j),\ I^{tgt}_{Lab}(i,j)\right)$$

Con $\varepsilon$-smoothing $C_k' = \sqrt{(a_k')^2 + (b_k^*)^2 + \varepsilon}$ per differenziabilità.

---

#### 6.5.2 Color Histogram Loss — Distribuzione Spettrale Globale

**Soft histogram differenziabile.** Per il canale $c \in \{L^*, a^*, b^*\}$ con $B = 64$ bin:

$$h_c^{pred}(k) = \frac{\sum_{i,j} \exp\!\left(-\frac{\left(I^{pred}_c(i,j) - \mu_k\right)^2}{2\sigma_{bin}^2}\right)}{\sum_{k'}\sum_{i,j} \exp\!\left(-\frac{\left(I^{pred}_c(i,j) - \mu_{k'}\right)^2}{2\sigma_{bin}^2}\right)}$$

con $\sigma_{bin} = \frac{1}{2B}$.

**Earth Mover's Distance (EMD):**

$$\mathcal{L}_{hist} = \frac{1}{3}\sum_{c \in \{L^*, a^*, b^*\}} \sum_{k=1}^{B}\left|\text{CDF}_c^{pred}(k) - \text{CDF}_c^{tgt}(k)\right|$$

Per distribuzioni 1D, EMD = $W_1$ = norma L1 tra le CDF. $\mathcal{L}_{hist}$ è invariante a permutazioni spaziali dei pixel — complementare a $\mathcal{L}_{\Delta E}$.

---

#### 6.5.3 Perceptual Loss — Similarità Semantica Multi-Scala

**VGG16 con 2 layer (invece di VGG19 con 4).** Per contenere il costo computazionale in fp16, si usano le feature di VGG16 (frozen, pre-addestrata su ImageNet) a soli 2 layer: relu2\_2 e relu3\_3.

| Layer | $C_l$ | $H_l \times W_l$ (su input $H\times W$) | Cattura |
|-------|-------|----------------------------------------|---------|
| relu2\_2 | 128 | $H/2 \times W/2$ | Texture, pattern semplici, struttura a media scala |
| relu3\_3 | 256 | $H/4 \times W/4$ | Strutture, pattern complessi, parti semantiche |

La perceptual loss è:

$$\mathcal{L}_{perc} = \sum_{l \in \{2,3\}} w_l\cdot\frac{1}{C_l H_l W_l}\left\|\phi_l(I^{pred}) - \phi_l(I^{tgt})\right\|_F^2$$

con pesi $\mathbf{w} = [1.0, 0.75]$ (layer più profondo pesa meno perché meno preciso spazialmente).

**Motivazione della scelta VGG16 a 2 layer.** VGG16 ha meno parametri rispetto a VGG19 e produce feature equivalenti fino al blocco 3, sufficiente per catturare struttura e texture rilevanti per il color grading. La riduzione a 2 layer taglia il costo computazionale della perceptual loss di circa il 60% rispetto alla versione originale (VGG19 con 4 layer), senza significativa perdita di qualità del segnale di gradiente per questo task.

**Perché i pesi VGG sono congelati.** Aggiornare i pesi di VGG16 durante il training farebbe collassare $\phi_l$ su una rappresentazione banale che minimizza $\mathcal{L}_{perc}$ ma perde il significato percettivo.

**Gradient flow:**

$$\frac{\partial\mathcal{L}_{perc}}{\partial I^{pred}} = \sum_{l\in\{2,3\}} \frac{2w_l}{C_l H_l W_l} J_{\phi_l}(I^{pred})^T \!\left(\phi_l(I^{pred}) - \phi_l(I^{tgt})\right)$$

---

#### 6.5.4 Style Loss (Gram Matrix) — Correlazioni Cromatico-Texturali

Matrice di Gram normalizzata su VGG16 (gli stessi 2 layer di $\mathcal{L}_{perc}$):

$$\mathbf{G}_l(I) = \frac{1}{C_l H_l W_l}\,\mathbf{F}_l\,\mathbf{F}_l^T \in \mathbb{R}^{C_l\times C_l}$$

$$\mathcal{L}_{style} = \frac{1}{2}\sum_{l \in \{2,3\}}\left\|\mathbf{G}_l(I^{pred}) - \mathbf{G}_l(I^{tgt})\right\|_F^2$$

Il fattore $1/2$ normalizza rispetto al numero di layer. Huang & Bethge (2017) dimostrano che le statistiche di secondo ordine delle feature maps encodano lo stile visivo: la style loss garantisce che anche le correlazioni inter-canale siano allineate al target, complementando il controllo di primo ordine operato da AdaIN.

---

#### 6.5.5 Cosine Similarity Loss — Direzione Cromatica

Sia $\mathbf{v}(i,j) = (a^*(i,j),\, b^*(i,j)) \in \mathbb{R}^2$ il vettore cromatico nel piano $(a^*, b^*)$:

$$\mathcal{L}_{cos} = 1 - \frac{1}{HW}\sum_{i=1}^{H}\sum_{j=1}^{W} \frac{\mathbf{v}^{pred}(i,j)^\top\,\mathbf{v}^{tgt}(i,j)}{\max\!\left(\left\|\mathbf{v}^{pred}(i,j)\right\|_2,\,\varepsilon\right)\cdot\max\!\left(\left\|\mathbf{v}^{tgt}(i,j)\right\|_2,\,\varepsilon\right)}$$

con $\varepsilon = 10^{-8}$. $\mathcal{L}_{cos} = 0$ quando ogni pixel ha esattamente lo stesso hue del target. Isola l'errore di hue, complementando $\mathcal{L}_{\Delta E}$ che combina errori di hue, saturazione e luminanza.

---

#### 6.5.6 Chroma Consistency Loss — Saturazione e Hue Circolare

**Saturazione:**

$$\mathcal{L}_{sat} = \frac{1}{HW}\sum_{i,j}\left|\sqrt{(a^{*,pred})^2 + (b^{*,pred})^2 + \varepsilon} - \sqrt{(a^{*,tgt})^2 + (b^{*,tgt})^2 + \varepsilon}\right|$$

**Hue circolare:**

$$h^*(i,j) = \text{atan2}(b^*(i,j),\, a^*(i,j)) \in (-\pi, \pi]$$

$$d_{circ}(h_1, h_2) = \arccos\!\left(\cos(h_1 - h_2)\right) \in [0, \pi]$$

$$\mathcal{L}_{hue} = \frac{1}{HW}\sum_{i,j} d_{circ}(h^{*,pred}(i,j),\, h^{*,tgt}(i,j))$$

$$\mathcal{L}_{chroma} = \mathcal{L}_{sat} + 0.5\cdot\mathcal{L}_{hue}$$

---

#### 6.5.7 Identity Loss — Prevenzione dell'Overediting

Con probabilità $p_{id} = 0.2$ ad ogni mini-batch, $I^{tgt} \leftarrow I^{src}$:

$$\mathcal{L}_{id} = \frac{1}{HW}\sum_{i,j}\left\|I^{pred}(i,j) - I^{src}(i,j)\right\|_1$$

Vincola implicitamente i coefficienti della bilateral grid verso l'identità ($\mathbf{A}_{ij} \to \mathbf{I}_{3\times3}$, $\mathbf{b}_{ij} \to \mathbf{0}$) e funge da regolarizzatore geometrico che previene divergenze numeriche.

---

#### 6.5.8 Style Consistency Loss — Coerenza Inter-Immagine

Per una coppia di immagini nel mini-batch, la similarity di contenuto è:

$$w_{ab} = \frac{\text{Enc}(I_a^{src}) \cdot \text{Enc}(I_b^{src})}{\|\text{Enc}(I_a^{src})\|_2\,\|\text{Enc}(I_b^{src})\|_2}$$

Se $w_{ab} > \tau_{cons} = 0.7$:

$$\mathcal{L}_{cons}^{(a,b)} = w_{ab} \cdot \left[\,\mathcal{L}_{hist}(I_a^{pred}, I_b^{pred}) + \frac{1}{2}\sum_{l \in \{2,3\}}\left\|\mathbf{G}_l(I_a^{pred}) - \mathbf{G}_l(I_b^{pred})\right\|_F^2\right]$$

(il fattore $1/2$ al posto di $1/4$ riflette l'uso di 2 invece di 4 layer).

$$\mathcal{L}_{cons} = \frac{2}{B(B-1)}\sum_{a < b} \mathcal{L}_{cons}^{(a,b)} \cdot \mathbf{1}[w_{ab} > \tau_{cons}]$$

Attivata solo nella fase 3B (epoche 11–30).

#### 6.5.9 Differenziabilità

**Teorema 3 (Differenziabilità della Color-Aesthetic Loss).** La funzione $\mathcal{L}: \mathbb{R}^{H\times W\times 3} \to \mathbb{R}_{\geq 0}$ è differenziabile quasi ovunque rispetto a $I^{pred}$.

*Dimostrazione per componenti:*

$\mathcal{L}_{\Delta E}$: con $\varepsilon$-smoothing su tutte le radici quadrate, differenziabile q.o.

$\mathcal{L}_{hist}$: il kernel gaussiano è $C^\infty$; la norma L1 delle CDF è differenziabile q.o.

$\mathcal{L}_{perc}$: composizione di convoluzione e ReLU (su VGG16 frozen); differenziabile q.o.

$\mathcal{L}_{style}$: la Gram matrix è un prodotto matriciale (ovunque differenziabile). Differenziabile ovunque.

$\mathcal{L}_{cos}$: con $\max(\|v\|, \varepsilon)$; differenziabile q.o.

$\mathcal{L}_{chroma}$: $C^* = \sqrt{\cdot + \varepsilon}$ ovunque differenziabile; $d_{circ}$ differenziabile q.o.

Somma pesata di funzioni differenziabili q.o. è differenziabile q.o. $\square$

#### 6.5.10 Non Convessità e Implicazioni

**Teorema 4 (Non Convessità).** $\mathcal{L}$ è non convessa in $I^{pred}$.

*Prova per $\mathcal{L}_{style}$:* La Gram matrix è quadratica in $\phi_l(I)$ (non lineare in $I$); il termine $\|\mathbf{G}_l(I^{pred}) - \mathbf{G}_l(I^{tgt})\|_F^2$ è quartico in $\phi_l(I^{pred})$ con Hessiano a autovalori di segno misto. $\square$

Strategie di mitigazione: (1) inizializzazione da $\theta_{meta}$ già vicina a un buon bacino; (2) AdamW con momentum; (3) curriculum progressivo delle loss.

#### 6.5.11 Convergenza

**Teorema 5.** Sia $\mathcal{L}$ $L$-smooth e lower-bounded da $\mathcal{L}^*$. AdamW con $\eta \leq 1/L$ e gradient clipping (max\_norm = 1.0) converge a un punto critico:

$$\min_{t=1,\ldots,K} \mathbb{E}\left[\left\|\nabla_\theta\mathcal{L}(\theta_t)\right\|^2\right] \leq \frac{2(\mathcal{L}(\theta_0) - \mathcal{L}^*)}{\eta K}$$

In $K = O(1/\epsilon^2)$ passi si raggiunge $\|\nabla\mathcal{L}\| \leq \epsilon$. $\square$

---

### 6.6 Curriculum dei Pesi della Loss

| Epoca | $\lambda_{\Delta E}$ | $\lambda_{hist}$ | $\lambda_{perc}$ | $\lambda_{style}$ | $\lambda_{cos}$ | $\lambda_{chroma}$ | $\lambda_{id}$ |
|-------|---------------------|-----------------|-----------------|------------------|-----------------|---------------------|----------------|
| 1–5 | 0.6 | 0.4 | 0.0 | 0.0 | 0.0 | 0.0 | 0.5 |
| 6–10 | 0.5 | 0.3 | 0.2 | 0.1 | 0.0 | 0.1 | 0.5 |
| 11–20 | 0.5 | 0.3 | 0.4 | 0.2 | 0.15 | 0.2 | 0.5 |
| 21+ | 0.5 | 0.3 | 0.4 | 0.2 | 0.15 | 0.2 | 0.5 |

**Motivazione.** Nelle prime 5 epoche si usano solo $\mathcal{L}_{\Delta E}$ e $\mathcal{L}_{hist}$: forniscono gradiente stabile e diretto. I termini basati su VGG16 ($\mathcal{L}_{perc}$, $\mathcal{L}_{style}$) vengono attivati solo dopo convergenza cromatica di base. $\mathcal{L}_{cos}$ e $\mathcal{L}_{chroma}$ entrano nella fase finale per raffinare la direzionalità cromatica.

---

### 6.7 Tabella Riassuntiva delle Proprietà Matematiche

| Termine | Spazio | Differenziabile | Convessa | Invariante a permutazioni spaziali | Penalizza |
|---------|--------|-----------------|----------|-------------------------------------|-----------|
| $\mathcal{L}_{\Delta E}$ | Lab | ✅ q.o. | ❌ | ❌ | Errore cromatico percettivo pixel-wise |
| $\mathcal{L}_{hist}$ | Lab (CDF) | ✅ q.o. | ❌ | ✅ | Distribuzione globale dei colori |
| $\mathcal{L}_{perc}$ | Feature VGG16 (2L) | ✅ q.o. | ❌ | ❌ | Struttura semantica bi-scala |
| $\mathcal{L}_{style}$ | Gram VGG16 (2L) | ✅ | ❌ | ✅ | Correlazioni inter-canale (stile) |
| $\mathcal{L}_{cos}$ | $(a^*,b^*)$ | ✅ q.o. | ❌ | ❌ | Errore di hue (direzione cromatica) |
| $\mathcal{L}_{chroma}$ | $(C^*, h^*)$ | ✅ q.o. | ❌ | ❌ | Saturazione e hue circolare |
| $\mathcal{L}_{id}$ | RGB | ✅ | ✅ | ❌ | Overediting (identità) |
| $\mathcal{L}$ totale | — | ✅ q.o. | ❌ | — | Combinazione pesata curriculum |

---
## 7. Dataset Disponibili

### 7.1 MIT-Adobe FiveK

**Link**: https://data.csail.mit.edu/graphics/fivek/

**Struttura**:
```
fivek/
├── raw/              # 5000 DNG files
├── tiff/             # Pre-processed TIFF
├── expert_A/         # Retouched by expert A
├── expert_B/         # Retouched by expert B
├── expert_C/         # Retouched by expert C
├── expert_D/         # Retouched by expert D
└── expert_E/         # Retouched by expert E
```

**Caratteristiche**:
- 5,000 RAW images (DNG format)
- 5 expert photographers con stili distinti:
  - A: Dramatic, high contrast
  - B: Natural, subtle
  - C: Vibrant colors
  - D: Warm tones
  - E: Cool, desaturated
- Risoluzione: ~3-5 MP

**Usage per la tesi**:
1. **Meta-training Reptile**: Experts A, B, C (3 tasks reali × 1000 coppie + task sintetici da interpolazione)
2. **Validation**: Expert D
3. **Test**: Expert E

---

### 7.2 PPR10K

**Caratteristiche**:
- 10,000 portrait images con coppie before/after
- Human region masks
- Focus: Skin tones, color grading su ritratti

**Usage**: Ablation study su ritratti vs scene generali.

---

### 7.3 Creazione Dataset Custom

Per il fotografo target, il dataset viene costruito esportando coppie (originale, editato) direttamente dal software di post-produzione (Lightroom, Capture One, Darktable). Non sono necessari metadata di editing: il sistema richiede solo le immagini. Il formato consigliato è sRGB TIFF 16-bit o RAW demosaicato.

**Requisiti minimi:**

- Numero coppie: $N \geq 50$ con meta-learning Reptile; $N \in [100, 200]$ nella configurazione ottimale
- Diversità delle scene: copertura dello spazio visivo del fotografo — ritratti, paesaggi, still life, bassa luce, luce naturale e artificiale
- Consistenza temporale: tutte le coppie devono essere state editate nello stesso "periodo stilistico"

**Nota sul numero minimo.** Con $N = 50$ e Reptile da $\theta_{meta}$, il modello ha già visto migliaia di coppie durante il meta-training e deve solo adattare lo stile generale. Senza meta-learning, $N = 50$ è insufficiente per qualsiasi architettura di complessità comparabile.

---

### 7.4 Data Augmentation Strategy

La data augmentation è critica nel regime few-shot perché moltiplica effettivamente il numero di esempi.

**Regola fondamentale:** qualsiasi trasformazione applicata a $I^{src}$ deve essere applicata identicamente a $I^{tgt}$, in modo da preservare la relazione di editing.

**Trasformazioni geometriche** (identiche su entrambe le immagini): flip orizzontale con probabilità $p = 0.5$; random crop su regione $[0.7H, H] \times [0.7W, W]$ con aspect ratio preservato; rotazione uniforme $\theta \sim \mathcal{U}(-5°, 5°)$.

**Perturbazioni di acquisizione** (solo su $I^{src}$, con probabilità $p = 0.3$): rescaling dell'esposizione con fattore $\gamma \sim \mathcal{U}(0.9, 1.1)$; rumore gaussiano $\mathcal{N}(0, \sigma^2)$ con $\sigma \sim \mathcal{U}(0, 0.01)$.

**Perturbazioni non ammesse:** qualsiasi trasformazione del colore su $I^{tgt}$; trasformazioni geometriche diverse tra $I^{src}$ e $I^{tgt}$.

**Fattore di moltiplicazione effettivo.** Con flip (×2), 3 scale di crop (×3) e perturbazione di esposizione (×2): $2 \times 3 \times 2 = 12$ coppie aumentate per coppia originale — un dataset da $N=100$ genera circa 1200 campioni effettivi per epoca.

---

## 8. Strategie di Training

### 8.1 Panoramica: Training a Tre Fasi

| Fase | Dataset | Obiettivo | Durata stimata |
|------|---------|-----------|----------------|
| **1. Pre-training** | FiveK (tutti i fotografi, mixed) | Imparare la grammatica del color grading | ~8h su RTX 3080 (fp16) |
| **2. Meta-training Reptile** | FiveK (per-photographer) + task sintetici | $\theta_{meta}$: inizializzazione per adattamento rapido | ~12h su RTX 3080 (fp16) |
| **3. Few-shot adaptation** | 100–200 coppie del fotografo target | Personalizzazione al singolo stile | ~2h su RTX 3080 (fp16) |

La progressione è strettamente ordinata: la fase 2 richiede un punto di partenza $\theta_0$ già vicino a una trasformazione fotografica plausibile (fase 1), e la fase 3 richiede $\theta_{meta}$ come inizializzazione.

**fp16 mixed precision in tutte le fasi.** Si usa PyTorch AMP con GradScaler in tutte e tre le fasi. Il GradScaler inizia con scala $S = 2^{10}$ e la aggiusta automaticamente (scale up/down) in base alla frequenza di overflow dei gradienti. La loss viene sempre calcolata in fp32 (upcast automatico).

---

### 8.2 Fase 1: Pre-Training su FiveK Mixed

**Obiettivo.** Il modello impara la struttura generale di una trasformazione fotografica — cosa significa "migliorare" un'immagine in modo fotograficamente credibile, indipendentemente dallo stile del fotografo specifico.

**Configurazione.** Dataset: unione dei target di tutti e 5 i fotografi FiveK ($\mathcal{D}_{pre} = \bigcup_{k \in \{A,\ldots,E\}} \{(I_i^{src}, I_i^{tgt,k})\}_{i=1}^{1000}$, totale 5000 coppie). Il conditioning sullo stile ($\mathbf{s}$) è disabilitato. Loss semplificata:

$$\mathcal{L}_{pre} = \mathcal{L}_{\Delta E} + 0.5\,\mathcal{L}_{perc}$$

**Ottimizzazione.** AdamW con $\eta_1 = 10^{-4}$, $\lambda_{wd} = 10^{-4}$, batch size $B_1 = 6$ (fp16, $512\times384$, RTX 3080), per 50 epoche. Learning rate warmup lineare da 0 a $\eta_1$ nelle prime 2 epoche, poi cosine decay. Gradient checkpointing attivo sull'encoder.

---

### 8.3 Fase 2: Meta-Training Reptile con Task Augmentation

**Inizializzazione.** Il modello parte da $\theta_0 = \theta_{pre}$. Set Transformer e cross-attention vengono attivati. Loss: Color-Aesthetic Loss con curriculum dalla colonna "6–10" come punto di partenza.

**Iperparametri Reptile:** $\alpha = 10^{-3}$ (inner lr), $\epsilon = 10^{-2}$ (meta step size), $M = 2$ (task per batch), $K_s = 15$ (coppie support), $K_q = 5$ (coppie query), $k = 5$ (inner steps). Totale: 10000 iterazioni meta.

**Task sampling.** Ad ogni iterazione, i $M = 2$ task vengono campionati da $p_{aug}$: con probabilità $0.5$ si usa un task reale (fotografo FiveK), con probabilità $0.5$ un task sintetico con $\lambda \sim \mathcal{U}(0.1, 0.9)$ e coppia di fotografi scelta uniformemente tra le $\binom{5}{2} = 10$ coppie possibili.

**Aggiornamento Reptile:**

$$\theta \leftarrow \theta + \frac{\epsilon}{M}\sum_{m=1}^{M}(\tilde{\theta}_{\mathcal{T}_m} - \theta), \quad \tilde{\theta}_{\mathcal{T}_m} = \mathcal{U}_\alpha^k(\theta, \mathcal{D}_{\mathcal{T}_m}^{sup})$$

**Gradient clipping:** norma globale $c = 1.0$ applicata dopo ogni inner loop, per stabilità in fp16.

**Stabilità fp16 con Reptile.** A differenza di MAML, Reptile non richiede la computazione di Hessiani e si adatta naturalmente a fp16 con GradScaler. L'aggiornamento $\tilde{\theta}_\mathcal{T} - \theta$ viene calcolato in fp32 (differenza di due tensori fp16 promossi) per evitare errori di cancellazione numerica.

---

### 8.4 Fase 3: Few-Shot Adaptation (Freeze-Then-Unfreeze)

**Inizializzazione:** $\theta_0 = \theta_{meta}$.

**Fase 3A** (epoche 1–10): parametri liberi $\Theta_{free}^A = \Theta_{slow} \cup \Theta_{adapt}$; $\Theta_{freeze}$ congelati. Loss curriculum dalla colonna "1–5" per le prime 5 epoche, poi "6–10". AdamW con $\eta_A = 5\times10^{-5}$, $\lambda_{wd} = 2\times10^{-3}$, fp16 con GradScaler.

**Fase 3B** (epoche 11–30): tutti i parametri liberi. Loss curriculum dalla colonna "11–20" in poi. AdamW con $\eta_B = 2.5\times10^{-5}$, cosine annealing:

$$\eta(t) = \eta_B\cdot\frac{1}{2}\!\left(1 + \cos\!\left(\frac{\pi(t-10)}{20}\right)\right), \quad t \in [10, 30]$$

**Early stopping.** Holdout $\mathcal{D}_\phi^{val}$ ($20\%$ delle coppie), patience $p = 5$ epoche.

---

### 8.5 Regularization

**Decoupled weight decay (AdamW):**

$$\theta_{t+1} = \theta_t - \eta\!\left(\frac{\hat{m}_t}{\sqrt{\hat{v}_t}+\varepsilon}\right) - \eta\,\lambda_{wd}\,\theta_t$$

**Identity loss** (sezione 6.5.7): con probabilità $p_{id} = 0.2$ per mini-batch.

**Gradient clipping:** norma globale $c = 1.0$.

**Data augmentation** (sezione 7.4): moltiplicazione effettiva $\times 12$.

**Gradient checkpointing** sull'encoder: riduce il picco di VRAM del 40–50% al costo di +30% nel tempo di training.

---

## 9. Valutazione e Metriche

### 9.1 Metriche Quantitative

#### 9.1.1 CIEDE2000 (ΔE₀₀)

$$\overline{\Delta E}_{00} = \frac{1}{|\mathcal{D}_{test}|} \sum_{(I^{src}, I^{tgt}) \in \mathcal{D}_{test}} \frac{1}{HW} \sum_{i=1}^{H}\sum_{j=1}^{W} \Delta E_{00}(I^{pred}_{Lab}(i,j),\, I^{tgt}_{Lab}(i,j))$$

**Target:** $\overline{\Delta E}_{00} < 5$ (accettabile), $< 2$ (eccellente).

---

#### 9.1.2 SSIM — Structural Similarity Index

Per due finestre $\mathbf{x}, \mathbf{y} \in \mathbb{R}^{N}$:

$$\text{SSIM}(\mathbf{x}, \mathbf{y}) = \frac{(2\mu_x \mu_y + c_1)(2\sigma_{xy} + c_2)}{(\mu_x^2 + \mu_y^2 + c_1)(\sigma_x^2 + \sigma_y^2 + c_2)}$$

con $c_1 = (0.01)^2$, $c_2 = (0.03)^2$. SSIM si decompone in:

$$\text{SSIM} = \underbrace{\frac{2\mu_x\mu_y + c_1}{\mu_x^2 + \mu_y^2 + c_1}}_{\text{luminance}} \cdot \underbrace{\frac{2\sigma_x\sigma_y + c_2}{\sigma_x^2 + \sigma_y^2 + c_2}}_{\text{contrast}} \cdot \underbrace{\frac{\sigma_{xy} + c_3}{\sigma_x\sigma_y + c_3}}_{\text{structure}}$$

La **MS-SSIM** su $M = 5$ scale è:

$$\text{MS-SSIM}(I^{pred}, I^{tgt}) = \prod_{j=1}^{M} \left[\text{SSIM}^{(j)}\right]^{\omega_j}, \quad \boldsymbol{\omega} = [0.0448,\, 0.2856,\, 0.3001,\, 0.2363,\, 0.1333]$$

Per il color grading si applica SSIM sulla luminanza $L^*$ (struttura) piuttosto che sui canali RGB (colore):

$$\text{SSIM}_{struct} = \text{SSIM}(I^{pred}_{L^*},\, I^{tgt}_{L^*})$$

**Target:** $\text{SSIM}_{struct} > 0.95$.

---

#### 9.1.3 LPIPS — Learned Perceptual Image Patch Similarity

LPIPS (Zhang et al., CVPR 2018) usa pesi calibrati tramite esperimenti psicofisici su giudizi umani. Sia $\hat{\phi}_l(I) = \phi_l(I) / \|\phi_l(I)\|_2$ la feature map normalizzata per canale:

$$\text{LPIPS}(I^{pred}, I^{tgt}) = \sum_{l} \frac{1}{H_l W_l} \sum_{h,w} \left\| \mathbf{w}_l \odot \left(\hat{\phi}_l^{hw}(I^{pred}) - \hat{\phi}_l^{hw}(I^{tgt})\right) \right\|_2^2$$

con $\mathbf{w}_l \in \mathbb{R}^{C_l}$ pesi per canale appresi su JND (Just Noticeable Difference).

**Target:** $\text{LPIPS} < 0.1$.

---

#### 9.1.4 NIMA — Neural Image Assessment

NIMA predice la distribuzione dei voti estetici $\mathbf{p}(I) = \text{Softmax}(\text{CNN}_{NIMA}(I)) \in \Delta^{10}$:

$$\mu_{NIMA}(I) = \sum_{s=1}^{10} s \cdot p_s(I), \quad \sigma_{NIMA}(I) = \sqrt{\sum_{s=1}^{10}(s - \mu_{NIMA})^2 \cdot p_s(I)}$$

$$\Delta\mu_{NIMA} = \mu_{NIMA}(I^{pred}) - \mu_{NIMA}(I^{src})$$

**Target:** $\Delta\mu_{NIMA} > 0$.

---

#### 9.1.5 Tabella Riassuntiva

| Metrica | Formula | Target | Cosa misura |
|---------|---------|--------|-------------|
| $\overline{\Delta E}_{00}$ | Media pixel-wise CIEDE2000 | $< 5$ (good), $< 2$ (excellent) | Accuratezza cromatica percettiva |
| $\text{SSIM}_{struct}$ | SSIM su canale $L^*$ | $> 0.95$ | Preservazione struttura |
| $\text{LPIPS}$ | Distanza feature pesata | $< 0.1$ | Similarità percettiva calibrata |
| $\Delta\mu_{NIMA}$ | $\mu_{NIMA}(I^{pred}) - \mu_{NIMA}(I^{src})$ | $> 0$ | Miglioramento estetico assoluto |

---

### 9.2 Metriche Qualitative

#### Human Evaluation

**Protocol**:
1. 50 immagini test, 3 metodi (baseline HDRNet, fine-tuned DPE, HybridStyleNet)
2. A/B Test con 10 valutatori (incluso il fotografo target)
3. Domanda: "Quale predizione è più fedele allo stile del fotografo?"
4. **Metric**: Preference rate (%) per ogni metodo

**Expected result**: HybridStyleNet > 60% preference vs baselines

---

### 9.3 Ablation Studies

| Ablation | Configurazione | Misura il contributo di |
|----------|---------------|------------------------|
| **A0 — Baseline** | MobileNetV3-Small + BilGrid 8×8×8 + 16×16×8, no conditioning | Lower bound |
| **A1 — No Swin** | MobileNetV3-Small puro (stage 1-5 CNN), no Transformer | Context globale (tramonto, meteo) |
| **A2 — No Local Branch** | Solo Global BilGrid 8×8×8 | Edits locali (skin, sky, shadow) |
| **A3 — No Cross-Attention** | Style prototype via media semplice invece di Set Transformer + cross-attn | In-context style conditioning |
| **A4 — No Reptile** | Random init invece di $\theta_{meta}$ | Valore del meta-training |
| **A5 — No Task Augmentation** | Reptile su 5 task fissi FiveK (no sintetici) | Task augmentation per meta-overfitting |
| **A6 — No SPADE (→ AdaIN)** | AdaIN anche nel local branch | Conditioning spazialmente variabile |
| **A7 — No Consistency Loss** | Loss senza $\mathcal{L}_{consistency}$ | Coerenza stilistica inter-immagine |
| **A8 — 50 vs 100 vs 200 coppie** | Varia N del training set | Sample efficiency |
| **A9 — Full Model** | Tutto attivo | — |

**Expected ranking** per ΔE↓: A9 < A3 < A1 < A4 < A2 < A6 < A7 < A5 < A0

Il gap maggiore atteso è tra A1 (no Swin) e A9: la rimozione del context globale dovrebbe produrre il degradamento più marcato su scene con forte dipendenza contestuale. Il gap tra A4 (no Reptile) e A9 misura il valore dell'inizializzazione meta rispetto al training da zero.

---

### 9.4 Comparison con State-of-Art

| Method | ΔE ↓ | SSIM ↑ | LPIPS ↓ | Time (ms) ↓ |
|--------|------|--------|---------|-------------|
| HDRNet (generic) | 6.5 | 0.96 | 0.12 | **15** |
| DPE fine-tuned | 7.2 | 0.91 | 0.15 | 50 |
| CSRNet conditioned | 5.8 | 0.97 | 0.10 | 15 |
| **HybridStyleNet (ours)** | **4.5** | **0.97** | **0.09** | **22** |

Il ΔE atteso di 4.5 (leggermente peggiore del 4.2 del progetto originale con EfficientNet-B4 e grid 32×32) riflette le ottimizzazioni operate: MobileNetV3-Small ha minore capacità rappresentativa, e la grid 16×16 offre meno granularità spaziale della 32×32. Il vantaggio in termini di tempo di inferenza (~22ms vs ~25ms su RTX 3080 con fp16) compensa ampiamente.

---

## 10. Roadmap Implementativa

### Timeline (15 settimane)

| Fase | Settimane | Tasks | Deliverable |
|------|-----------|-------|-------------|
| **Setup & Dataset** | 1-2 | Download FiveK, preprocessing fp16, data loaders, crop $512\times384$ resolution-agnostic | Dataset pipeline ready |
| **Baseline** | 3 | MobileNetV3-Small + BilGrid senza conditioning (A0) | Benchmark lower bound |
| **CNN + Swin Encoder** | 4-5 | MobileNetV3-Small stem + Swin stage 4-5 con RoPE, 4 teste, gradient checkpointing | Encoder con context globale |
| **Bilateral Grid Branches** | 6-7 | Global (8×8×8, AdaIN) + Local (16×16×8, SPADE 128ch) + Confidence Mask | HybridStyleNet senza meta |
| **Set Transformer + Cross-Attn** | 8-9 | Style Prototype (4 teste) + Cross-Attention (K=20 subset fisso) | Conditioning completo |
| **Meta-Training Reptile** | 10-11 | Reptile ($M=2$ task) + task augmentation su FiveK, fp16 | $\theta_{meta}$ checkpoint |
| **Experiments & Ablations** | 12-13 | A0–A9 ablations + comparison baselines + metriche | Tutti i risultati |
| **Custom Dataset** | 14 | Few-shot adaptation su fotografo reale + human evaluation | Real-world demo |
| **Writing** | 15 | Thesis draft + presentazione | Submission ready |

---

## 11. Contributi Originali

### Contributo Scientifico

1. **Architettura CNN + Swin Transformer ibrida per photographer-specific color grading su GPU consumer** ⭐⭐⭐ **CONTRIBUTO PRINCIPALE**
   - Primo lavoro che motiva e risolve il problema del context globale nel color grading fotografico con un encoder ibrido CNN + Swin su GPU consumer (RTX 3080, 10 GB VRAM)
   - MobileNetV3-Small + Swin con 4 teste di attenzione + RoPE: complessità $O(n)$, generalizzazione a risoluzioni non viste, compatibilità nativa fp16
   - CNN stem preserva l'inductive bias locale critico per il few-shot regime con footprint ridotto

2. **Set Transformer + Cross-Attention con subset fisso K=20 per in-context style conditioning** ⭐⭐⭐
   - Il Set Transformer (4 teste) aggrega le edit delta in modo robusto agli outlier
   - Il cross-attention con subset fisso K=20 seleziona dinamicamente quale edit è più rilevante per la scena specifica — in-context learning con costo computazionale costante rispetto alla dimensione del training set
   - Sub-style automatico: stile "tramonto" vs stile "ritratto" dallo stesso training set

3. **Reptile con task augmentation per meta-overfitting su pochi fotografi**
   - Con 5 fotografi FiveK, MAML overfita e richiede Hessiani incompatibili con fp16; Reptile (primo ordine) è nativo fp16 e 4× più veloce per iterazione
   - L'interpolazione di stili in Lab genera task sintetici continui, risolvendo il meta-overfitting
   - Prima applicazione di Reptile con task augmentation a color grading photographer-specific

4. **SPADE (128 canali) nel local branch con bilateral grid 16×16×8**
   - La grid $16\times16$ è il compromesso ottimale tra granularità semantica e budget di memoria fp16 su GPU consumer
   - SPADE con 128 canali: conditioning spazialmente variabile con costo ridotto del 75% rispetto a 256 canali, mantenendo distinzione tra skin tones, cielo e ombre
   - Motivato teoricamente dalla natura localmente differenziata del color grading professionale

5. **Benchmark few-shot photographer-specific con fp16 su GPU consumer**
   - Protocollo interamente eseguibile su RTX 3080 (10 GB VRAM): batch $512\times384$, fp16, gradient checkpointing
   - Ablation suite completa (A0–A9) per misurare contributo isolato di ogni componente

### Contributo Pratico

1. **Few-shot pratico su GPU consumer**
   - 50-200 coppie (sole immagini, nessun metadata di editing)
   - ~2 ore di adattamento su RTX 3080 con fp16 vs giorni/cloud per soluzioni commerciali
   - Accessibile a fotografi individuali con hardware standard

2. **Inference a piena risoluzione su qualsiasi fotocamera**
   - ~2.2s su RTX 3080 per immagini $\approx 3000\times2000$ (24 MP) in fp16
   - Scala a risoluzioni maggiori (36, 45, 60 MP) senza modifiche architetturali grazie alla resolution-agnostic property + RoPE
   - Ampiamente entro il budget di 10s; ottimizzabile sotto 1s con TensorRT

3. **Open-source release**
   - Codice completo su GitHub (HybridStyleNet + training pipeline fp16 + evaluation)
   - $\theta_{meta}$ pre-trained su FiveK scaricabile: adattamento in 2 ore senza riaddestrare da zero
   - Requisiti hardware documentati: RTX 3080 (10 GB VRAM) o equivalente

---

## Conclusione

Questa documentazione delinea la tesi magistrale su **photographer-specific color grading via deep learning**, con un approccio rigorosamente **end-to-end** e interamente ottimizzato per GPU consumer con **fp16 mixed precision**.

**Key Takeaways**:
1. Il problema chiave non era l'architettura generica — era la mancanza di **context semantico globale** nelle CNN, risolto con l'encoder ibrido MobileNetV3-Small + Swin Transformer (4 teste, RoPE)
2. Il secondo problema era l'aggregazione robusta dello stile — risolto con Set Transformer (4 teste) + Cross-Attention con subset fisso K=20
3. Il terzo problema era il meta-overfitting e l'incompatibilità fp16 di MAML — risolto con Reptile (primo ordine, nativo fp16) + task augmentation in Lab space
4. Le ottimizzazioni hardware (fp16, gradient checkpointing, batch $512\times384$, grid 16×16, SPADE 128ch, K=20) rendono l'intero sistema eseguibile su RTX 3080 (10 GB VRAM) senza sacrificare la coerenza matematica dell'approccio
5. Tutti i contributi sono originali rispetto allo stato dell'arte e misurabili tramite gli ablation studies A0–A9
