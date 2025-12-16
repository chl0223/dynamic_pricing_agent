# Dynamic Pricing Agent

**Summary:** An advanced, machine learning-driven system designed for a competitive marketplace. Our agent employs an 8-Segment XGBoost model for personalized demand estimation, integrates a dynamic inventory-pressure multiplier for capacity management, and utilizes an Adaptive Strategy Selector (Meta-Agent) to dynamically counter opponent pricing strategies for maximal revenue.

* **Project:** Competitive Algorithmic Pricing Challenge
* **Group Members:** Alice Lee, Pin-Hsuan Lai, Pin-Yeh Lai

---

## Part 1: Personalized Pricing Under Capacity Constraints

**Goal:** Design a capacity-constrained pricing system by estimating demand and setting revenue-maximizing personalized prices while managing limited inventory.

### 1. Demand Estimation Strategy

Initial analysis confirmed demand varied significantly by $\text{Covariate1}$ and $\text{Covariate2}$, making segmentation essential.

* **Model Selection:** We developed an 8-Segment XGBoost model, splitting customers based on median cuts of the three covariates.
* **Performance:** The segmented approach ($\sim 3.23\text{M}$ actual revenue) significantly outperformed the strongest global model (LightGBM, $\sim 2.49\text{M}$), validating that segmentation was key to capturing customer variation.
* **Pricing Mechanism:** We used vectorized computation to predict purchase probabilities across 100 prices for each segment, selecting the price with the highest expected revenue.

### 2. Capacity-Constrained Price Optimization

To prevent early stockouts and reduce leftover inventory, we integrated an inventory adjustment mechanism into the optimal price selection.

* **Dynamic Inventory-Pressure Multiplier:** This multiplier adjusts the optimal price based on current inventory relative to the time until replenishment.
    * **Low Inventory:** Price is increased (up to **4%** per shortage unit) to delay stockouts.
    * **High Inventory:** Price is decreased (up to **2%** per extra unit) to aggressively capture orders.
* **Result:** This combined approach provided the best balance, outperforming simpler, rigid pricing variants that ignored inventory constraints.

---

## Part 2: Pricing Under Competition (Duopoly)

**Goal:** Adapt the pricing strategy to maximize total revenue in a round-robin tournament against other agents, where inventory resets every 20 customers.

### 1. Adaptive Strategy Selector (Meta-Agent)

The core innovation is an adaptive framework that identifies opponent behavior before selecting the optimal counter-strategy. This was crucial for avoiding the low-profit outcomes of DP vs. DP price wars ($\sim 3\text{K}-4\text{K}$).

* **Detection Phase:** In the first 80 steps, the agent collects opponent quotes and calculates two price stability indicators: Price Standard Deviation (`price_std`) and Small Change Ratio (`frac_small_move`).
* **Classification and Optimal Deployment:**
    * **Against Static/Predictable Opponent (`price_std` < 15$):** Deploys the DP Agent (designed for stable revenue).
    * **Against Dynamic/Reactive Opponent:** Deploys the Multiplier Agent (designed to compete effectively against volatile pricing).

### 2. Strategic Insights from Simulation

Local simulation helped derive a "payoff matrix" confirming the strategy was dependent on the competitor type:

| Strategy Matchup | Avg Revenue (Each Agent) | Key Insight |
| :--- | :--- | :--- |
| Multiplier vs. DP | DP wins ($\sim 10\text{K}$); Multiplier gets ($\sim 7\text{K}$) | Multiplier Agent is profitable even when losing. |
| DP vs. DP | $\sim 3\text{K}-4\text{K}$ | Vicious Price War—must be avoided. |

### 3. Sub-Agent Deployment Strategy

| Sub-Agent | Best Used Against | Core Strategy Components |
| :--- | :--- | :--- |
| **DP Agent** | Predictable Competitors | Integrates an offline DP policy, a Greedy Optimization (vectorized price search), and a dynamic multiplier based on sales success rate. |
| **Multiplier Agent** | Volatile/Reactive Competitors | Combines XGBoost with a heuristic centered on Market Saturation, adjusting the quote based on joint inventory and time pressure. |

---

## 🛠️ Tools and Technologies

| Category | Tools & Libraries | Key Application |
| :--- | :--- | :--- |
| **Language** | Python | Primary implementation language. |
| **Machine Learning** | XGBoost, LightGBM | Demand estimation and feature-based segmentation. |
| **Optimization** | NumPy | Used for **vectorized computations** (e.g., price search), crucial for meeting the $\frac{1}{2}$ second execution speed constraint. |
| **Version Control** | Git | Team collaboration and submission management. |

---

## Conclusion

The project's success is attributed to the Adaptive Strategy Selector, proving that algorithmic complexity must be balanced with practical market understanding. The ability to correctly classify opponents and deploy the optimal counter-strategy was the key differentiator in the competitive environment.
