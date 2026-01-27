import streamlit as st
import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from typing import Dict, Tuple
import sys

# Configuration de la page
st.set_page_config(
    page_title="Modèle Principal-Agent",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Style CSS personnalisé
st.markdown("""
<style>
/* Fond principal */
.stApp {
  background: linear-gradient(135deg, #0f2433 0%, #1b2b3a 60%, #eaf3f8 100%);
  color-scheme: light;
}

/* Sidebar */
[data-testid="stSidebar"] {
  background: linear-gradient(180deg, #0b1a26 0%, #12232f 100%);
  border-right: 1px solid rgba(255,255,255,0.06);
  box-shadow: 0 6px 18px rgba(11,26,38,0.6);
  color: #e9f2f7;
}

/* Titres */
h1, h2, h3 {
  color: #ffffff !important;
  font-weight: 600 !important;
  text-shadow: 0 2px 6px rgba(11,26,38,0.6);
  letter-spacing: 0.2px;
}

/* Texte général */
p, li, span, label {
  color: #e9f2f7 !important;
  line-height: 1.5;
}

/* Métriques */
[data-testid="stMetricValue"] {
  color: #ffffff !important;
  font-size: 1.75rem !important;
  font-weight: 700 !important;
}
[data-testid="stMetricLabel"] {
  color: #bcd6e6 !important;
  font-weight: 500 !important;
}

/* Boutons globaux */
.stButton > button {
  background: linear-gradient(180deg, #6a8fb0 0%, #5b7fa0 100%) !important;
  color: #ffffff !important;
  border: 1px solid rgba(255,255,255,0.06) !important;
  font-weight: 600 !important;
  border-radius: 10px !important;
  padding: 0.55rem 1.1rem !important;
  box-shadow: 0 6px 18px rgba(91,127,160,0.12) !important;
  transition: transform 0.18s ease, box-shadow 0.18s ease, color 0.12s ease, background 0.12s ease !important;
}
.stButton > button:hover {
  transform: translateY(-3px) !important;
  box-shadow: 0 10px 30px rgba(91,127,160,0.18) !important;
}

/* --- Modifications demandées : sidebar - boutons et textes en noir sur fond clair --- */

/* Texte et labels dans la sidebar (par défaut lisible sur fond sombre) */
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] .stMarkdown p,
[data-testid="stSidebar"] .caption,
[data-testid="stSidebar"] span,
[data-testid="stSidebar"] li {
  color: #e9f2f7 !important;
  transition: color 0.18s ease !important;
}

/* Inputs / selects / textareas dans la sidebar : fond clair et texte noir */
[data-testid="stSidebar"] input,
[data-testid="stSidebar"] textarea,
[data-testid="stSidebar"] select,
[data-testid="stSidebar"] .stNumberInput input {
  background-color: #ffffff !important;
  color: #071014 !important;
  border: 1px solid rgba(11,26,38,0.08) !important;
  border-radius: 8px !important;
  padding: 8px !important;
  transition: box-shadow 0.12s ease, border-color 0.12s ease, color 0.12s ease, background 0.12s ease !important;
}

/* Placeholder plus doux */
[data-testid="stSidebar"] input::placeholder,
[data-testid="stSidebar"] textarea::placeholder {
  color: rgba(7,16,20,0.45) !important;
}

/* Focus des inputs : texte noir et bordure d'accent ardoise bleu */
[data-testid="stSidebar"] input:focus,
[data-testid="stSidebar"] textarea:focus,
[data-testid="stSidebar"] select:focus,
[data-testid="stSidebar"] .stNumberInput input:focus {
  box-shadow: 0 6px 18px rgba(106,143,176,0.12) !important;
  border-color: rgba(106,143,176,0.36) !important;
  color: #071014 !important;
  outline: none !important;
}

/* Boutons spécifiques à la sidebar : fond clair, texte noir par défaut */
[data-testid="stSidebar"] .stButton > button {
  background: linear-gradient(180deg, #ffffff 0%, #f3f6f9 100%) !important;
  color: #071014 !important;
  border: 1px solid rgba(11,26,38,0.08) !important;
  box-shadow: none !important;
  font-weight: 600 !important;
  border-radius: 10px !important;
  padding: 0.5rem 1rem !important;
  transition: background 0.18s ease, color 0.12s ease, transform 0.12s ease, box-shadow 0.12s ease !important;
}

/* Hover des boutons sidebar */
[data-testid="stSidebar"] .stButton > button:hover {
  background: linear-gradient(180deg, #f0f4f8 0%, #e6eef5 100%) !important;
  transform: translateY(-2px) !important;
  box-shadow: 0 8px 20px rgba(11,26,38,0.06) !important;
}

/* Active / focus : texte bien noir et léger outline pour indiquer sélection */
[data-testid="stSidebar"] .stButton > button:active,
[data-testid="stSidebar"] .stButton > button:focus,
[data-testid="stSidebar"] .stButton > button[aria-pressed="true"] {
  background: linear-gradient(180deg, #e6eef5 0%, #dbeaf2 100%) !important;
  color: #071014 !important;
  outline: 2px solid rgba(106,143,176,0.18) !important;
  box-shadow: 0 8px 24px rgba(106,143,176,0.10) !important;
}

/* Onglets / sections sélectionnées dans la sidebar : texte noir pendant la sélection */
[data-testid="stSidebar"] [data-baseweb="tab"][aria-selected="true"],
[data-testid="stSidebar"] [aria-pressed="true"] {
  background: linear-gradient(180deg, #ffffff 0%, #f6fbff 100%) !important;
  color: #071014 !important;
  font-weight: 700 !important;
  border: 1px solid rgba(11,26,38,0.06) !important;
  transition: background 0.18s ease, color 0.12s ease !important;
}

/* Si un champ est cliqué, forcer le label adjacent à devenir noir pour lisibilité */
[data-testid="stSidebar"] input:focus + label,
[data-testid="stSidebar"] textarea:focus + label,
[data-testid="stSidebar"] select:focus + label {
  color: #071014 !important;
}

/* Retour à l'état normal : transitions douces pour que le texte redevienne clair après blur */
[data-testid="stSidebar"] input,
[data-testid="stSidebar"] textarea,
[data-testid="stSidebar"] select,
[data-testid="stSidebar"] .stButton > button,
[data-testid="stSidebar"] label {
  transition: color 0.18s ease, background 0.18s ease, border-color 0.18s ease !important;
}

/* --- Fin des modifications sidebar --- */
            
/* Onglets (globaux) */
.stTabs [data-baseweb="tab-list"] {
  gap: 10px;
  background-color: transparent;
  padding: 8px;
  border-radius: 12px;
}
.stTabs [data-baseweb="tab"] {
  background-color: var(--bg-surface);
  color: #000000; /* texte noir demandé */
  border-radius: 8px;
  padding: 8px 18px;
  font-weight: 500;
  border: 1px solid var(--border-subtle);
  transition: transform .18s ease, box-shadow .18s ease, color .12s ease, background .12s ease;
}
.stTabs [data-baseweb="tab"]:hover {
  transform: translateY(-3px);
  box-shadow: 0 10px 30px rgba(91,127,160,0.18);
  color: #000000;
}
.stTabs [data-baseweb="tab"]:focus-visible {
  outline: 3px solid var(--focus-ring);
  outline-offset: 3px;
  border-color: rgba(106,143,176,0.45);
}
.stTabs [data-baseweb="tab"][aria-selected="true"] {
  background: linear-gradient(180deg, #ffffff 0%, #f0f7fb 100%);
  color: #000000;
  font-weight: 700;
  box-shadow: 0 8px 20px rgba(11,26,38,0.6);
}

/* Cartes et conteneurs */
div[data-testid="stExpander"], .stContainer {
  background-color: var(--bg-surface);
  border: 1px solid var(--border-subtle);
  border-radius: 12px;
  padding: 12px;
}

/* Inputs globaux (plus ciblé) */
input[type="text"], input[type="email"], input[type="search"], .stNumberInput input, .stSlider, textarea {
  background-color: var(--bg-surface);
  color: var(--text-light);
  border: 1px solid rgba(255,255,255,0.04);
  border-radius: 8px;
  padding: 8px;
  transition: box-shadow .12s ease, border-color .12s ease;
}
input:focus-visible, textarea:focus-visible {
  box-shadow: 0 0 0 4px rgba(106,143,176,0.08);
  border-color: rgba(106,143,176,0.6);
}

/* Tableaux */
table {
  background-color: transparent;
  border-collapse: collapse;
  width: 100%;
}
table th {
  background-color: var(--bg-surface-2);
  color: #ffffff;
  font-weight: 600;
  border-bottom: 1px solid rgba(255,255,255,0.04);
  padding: 10px;
  text-align: left;
}
table td {
  background-color: transparent;
  color: #e6f1f6;
  border-bottom: 1px solid rgba(255,255,255,0.02);
  padding: 10px;
}
/* Alternance de lignes pour lisibilité */
table tbody tr:nth-child(even) td {
  background-color: rgba(255,255,255,0.01);
}

/* Petite adaptation responsive */
@media (max-width: 600px) {
  .stTabs [data-baseweb="tab"] {
    padding: 6px 10px;
    font-size: 14px;
  }
  table th, table td { padding: 8px; font-size: 13px; }
}
/* Messages d'info/warning/error/success */
.stAlert {
  background-color: rgba(255,255,255,0.02) !important;
  border: 1px solid rgba(255,255,255,0.04) !important;
  border-radius: 10px !important;
  color: #eaf6fb !important;
  padding: 10px !important;
}

/* Captions */
.caption {
  color: #a9c3d6 !important;
  font-style: italic !important;
}

/* Code blocks */
code {
  background-color: rgba(235,245,250,0.9) !important;
  color: #0b1a26 !important;
  border: 1px solid rgba(11,26,38,0.06) !important;
  padding: 4px 8px !important;
  border-radius: 6px !important;
  font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, "Roboto Mono", monospace;
}

/* Spinner */
.stSpinner > div {
  border-top-color: #6a8fb0 !important;
}

/* Scrollbar personnalisée */
::-webkit-scrollbar {
  width: 10px;
  height: 10px;
}
::-webkit-scrollbar-track {
  background: transparent;
}
::-webkit-scrollbar-thumb {
  background: rgba(11,26,38,0.12);
  border-radius: 6px;
}
::-webkit-scrollbar-thumb:hover {
  background: rgba(11,26,38,0.18);
}
</style>
""", unsafe_allow_html=True)

# Classe du modèle 
class PrincipalAgentModel:
    """
    Modèle Principal-Agent avec aléa moral
    """
    
    def __init__(self, q_bar=100, q=20, pi_H=0.79, pi_B=0.30, 
                 d_H=5, d_B=2, u_r=10, gamma=0.5):
        """Initialisation des paramètres du modèle"""
        # Vérifications des paramètres
        assert q_bar > q, "Le profit en cas de succès doit être > profit en cas d'échec"
        assert 0 < pi_B < pi_H < 1, "Condition: 0 < pi_B < pi_H < 1"
        assert d_H > d_B >= 0, "Le coût de l'effort élevé doit être > effort faible"
        assert 0 < gamma <= 1, "Coefficient d'aversion au risque: 0 < gamma ≤ 1"
        
        self.q_bar = q_bar
        self.q = q
        self.pi_H = pi_H
        self.pi_B = pi_B
        self.d_H = d_H
        self.d_B = d_B
        self.u_r = u_r
        self.gamma = gamma
        
    def utility_agent(self, w: float) -> float:
        """Fonction d'utilité de l'agent: u(w) = w^gamma"""
        return w**self.gamma if w >= 0 else -np.inf
    
    def expected_utility_agent(self, w_bar: float, w: float, effort: str) -> float:
        """Utilité espérée de l'agent: E[u(w(y))] - d_e"""
        pi = self.pi_H if effort == 'H' else self.pi_B
        d = self.d_H if effort == 'H' else self.d_B
        
        return pi * self.utility_agent(w_bar) + (1 - pi) * self.utility_agent(w) - d
    
    def expected_profit_principal(self, w_bar: float, w: float, effort: str) -> float:
        """Profit espéré du principal: E[q(y) - w(y)]"""
        pi = self.pi_H if effort == 'H' else self.pi_B
        
        return pi * (self.q_bar - w_bar) + (1 - pi) * (self.q - w)


def solve_first_best(model: PrincipalAgentModel, target_effort: str = 'H') -> Dict:
    """Résolution du cas first-best (effort observable)"""
    d = model.d_H if target_effort == 'H' else model.d_B
    pi = model.pi_H if target_effort == 'H' else model.pi_B

    # Salaire optimal: w_bar = w (assurance complète)
    w_optimal = (model.u_r + d)**(1/model.gamma)
    w_bar_optimal = w_optimal

    # Profit espéré du principal
    profit_principal = model.expected_profit_principal(
        w_bar_optimal, w_optimal, target_effort
    )

    # Utilité de l'agent
    utility_agent = model.expected_utility_agent(
        w_bar_optimal, w_optimal, target_effort
    )

    return {
        'effort': target_effort,
        'w_bar': w_bar_optimal,
        'w': w_optimal,
        'profit_principal': profit_principal,
        'utility_agent': utility_agent,
        'type': 'First-Best',
        'pi': pi
    }


def solve_second_best(model: PrincipalAgentModel) -> Dict:
    """Résolution du cas second-best (aléa moral - effort non observable)"""
    
    def objective(x):
        """Fonction à minimiser: -Profit du Principal"""
        w_bar, w = x
        if w_bar < 0 or w < 0:
            return 1e10
        return -model.expected_profit_principal(w_bar, w, 'H')
    
    def constraint_participation(x):
        """Contrainte de Participation (CP): EU_A(H) - u_r ≥ 0"""
        w_bar, w = x
        return model.expected_utility_agent(w_bar, w, 'H') - model.u_r
    
    def constraint_incentive(x):
        """Contrainte d'Incitation (CI): EU_A(H) - EU_A(B) ≥ 0"""
        w_bar, w = x
        eu_H = model.expected_utility_agent(w_bar, w, 'H')
        eu_B = model.expected_utility_agent(w_bar, w, 'B')
        return eu_H - eu_B
    
    # Contraintes
    constraints = [
        {'type': 'ineq', 'fun': constraint_participation},
        {'type': 'ineq', 'fun': constraint_incentive}
    ]
    
    # Point initial
    w_init = (model.u_r + model.d_H)**(1/model.gamma)
    x0 = [w_init * 1.5, w_init * 0.8]
    
    # Optimisation
    result = minimize(
        objective,
        x0,
        method='SLSQP',
        bounds=[(0, None), (0, None)],
        constraints=constraints,
        options={'ftol': 1e-9, 'maxiter': 1000}
    )
    
    w_bar_opt, w_opt = result.x
    
    # Calcul des résultats
    profit_principal = model.expected_profit_principal(w_bar_opt, w_opt, 'H')
    utility_agent_H = model.expected_utility_agent(w_bar_opt, w_opt, 'H')
    utility_agent_B = model.expected_utility_agent(w_bar_opt, w_opt, 'B')
    
    # Vérification des contraintes
    cp_slack = utility_agent_H - model.u_r
    ci_slack = utility_agent_H - utility_agent_B
    
    return {
        'effort': 'H',
        'w_bar': w_bar_opt,
        'w': w_opt,
        'profit_principal': profit_principal,
        'utility_agent': utility_agent_H,
        'utility_agent_B': utility_agent_B,
        'type': 'Second-Best',
        'cp_slack': cp_slack,
        'ci_slack': ci_slack,
        'success': result.success
    }

def verify_perfect_bayesian_equilibrium(model: PrincipalAgentModel, contract: Dict) -> Dict:
    """
    Vérifie si un contrat donné constitue un Équilibre Bayésien Parfait (PBE).

    Critères vérifiés :
      - Participation de l'agent (CP)
      - Contrainte d'incitation (CI)
      - Absence de déviation profitable pour le principal

    Args:
        model: instance de PrincipalAgentModel
        contract: dict contenant au moins 'w_bar' et 'w'

    Returns:
        dict avec indicateurs et valeurs numériques utiles pour l'interface
    """
    # Récupération des salaires
    w_bar = float(contract.get('w_bar', 0.0))
    w = float(contract.get('w', 0.0))

    # Utilités attendues
    eu_H = model.expected_utility_agent(w_bar, w, 'H')
    eu_B = model.expected_utility_agent(w_bar, w, 'B')
    eu_reject = model.u_r

    # Participation
    agent_accepts = max(eu_H, eu_B) >= eu_reject

    # Effort optimal si accepte
    if agent_accepts:
        optimal_effort = 'H' if eu_H >= eu_B else 'B'
        ic_satisfied = (eu_H >= eu_B)
    else:
        optimal_effort = None
        ic_satisfied = False

    # Profit du principal dans l'état anticipé
    profit_equilibrium = model.expected_profit_principal(
        w_bar, w, optimal_effort if optimal_effort is not None else 'B'
    )

    # Déviations testées
    w_B_deviation = (model.u_r + model.d_B)**(1.0 / model.gamma)
    profit_deviation_B = model.expected_profit_principal(w_B_deviation, w_B_deviation, 'B')
    profit_no_contract = 0.0

    no_profitable_deviation = (profit_equilibrium >= profit_deviation_B) and (profit_equilibrium >= profit_no_contract)

    is_pbe = agent_accepts and ic_satisfied and no_profitable_deviation

    return {
        'is_pbe': is_pbe,
        'agent_accepts': agent_accepts,
        'optimal_effort': optimal_effort,
        'ic_satisfied': ic_satisfied,
        'no_profitable_deviation': no_profitable_deviation,
        'equilibrium_profit': profit_equilibrium,
        'eu_H': eu_H,
        'eu_B': eu_B,
        'profit_deviation_B': profit_deviation_B,
        'w_bar': w_bar,
        'w': w
    }


def sensitivity_analysis(
    model: PrincipalAgentModel,
    gamma_range: Tuple[float, float] = (0.3, 1.0),
    gamma_steps: int = 8,
    pi_B_min: float = 0.2,
    pi_B_max_offset: float = 0.05,
    pi_B_steps: int = 8
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Analyse de sensibilité du coût d'agence par rapport à γ (aversion au risque)
    et π_B (probabilité de succès avec effort faible).

    Args:
        model: instance de PrincipalAgentModel (sera modifiée temporairement)
        gamma_range: (min, max) pour γ
        gamma_steps: nombre de points pour γ
        pi_B_min: borne inférieure pour π_B
        pi_B_max_offset: offset pour garantir π_B < π_H (π_B_max = min(0.7, π_H - offset))
        pi_B_steps: nombre de points pour π_B

    Returns:
        gammas, agency_costs_gamma, pi_Bs_effectifs, agency_costs_pi
        (tableaux numpy prêts à tracer)
    """
    # Sauvegarde des paramètres originaux
    original_gamma = model.gamma
    original_pi_B = model.pi_B

    # Grille pour gamma
    gammas = np.linspace(gamma_range[0], gamma_range[1], gamma_steps)
    agency_costs_gamma = np.zeros_like(gammas)

    for i, g in enumerate(gammas):
        model.gamma = float(g)
        fb = solve_first_best(model, 'H')
        sb = solve_second_best(model)
        agency_costs_gamma[i] = fb['profit_principal'] - sb['profit_principal']

    # Grille pour pi_B (s'assurer qu'elle reste < pi_H)
    pi_B_max = min(0.7, model.pi_H - pi_B_max_offset)
    if pi_B_max <= pi_B_min:
        # Si l'intervalle invalide, on renvoie un tableau vide pour pi_B
        pi_Bs_effectifs = np.array([])
        agency_costs_pi = np.array([])
    else:
        pi_Bs = np.linspace(pi_B_min, pi_B_max, pi_B_steps)
        agency_costs_pi_list: list = []
        pi_Bs_effectifs_list: list = []
        for pi_b in pi_Bs:
            if pi_b < model.pi_H:
                model.pi_B = float(pi_b)
                fb = solve_first_best(model, 'H')
                sb = solve_second_best(model)
                agency_costs_pi_list.append(fb['profit_principal'] - sb['profit_principal'])
                pi_Bs_effectifs_list.append(pi_b)
        pi_Bs_effectifs = np.array(pi_Bs_effectifs_list)
        agency_costs_pi = np.array(agency_costs_pi_list)

    # Restauration des paramètres originaux
    model.gamma = original_gamma
    model.pi_B = original_pi_B

    return gammas, agency_costs_gamma, pi_Bs_effectifs, agency_costs_pi

# ============= INTERFACE STREAMLIT =============

def main():
    st.title(" Modèle Principal-Agent avec Aléa Moral")
    st.markdown("---")
    
    # Sidebar pour les paramètres
    st.sidebar.header(" Paramètres du Modèle")
    
    st.sidebar.subheader("Profits")
    q_bar = st.sidebar.number_input(
        "q̄ (Profit si succès)", 
        min_value=1.0, 
        max_value=1000.0, 
        value=600.0, 
        step=10.0,
        help="Profit du principal en cas de succès"
    )
    
    q = st.sidebar.number_input(
        "q (Profit si échec)", 
        min_value=0.0, 
        max_value=float(q_bar-1), 
        value=100.0, 
        step=10.0,
        help="Profit du principal en cas d'échec"
    )
    
    st.sidebar.subheader("Probabilités de succès")
    pi_H = st.sidebar.slider(
        "π_H (Effort élevé)", 
        min_value=0.1, 
        max_value=0.99, 
        value=0.79, 
        step=0.01,
        help="Probabilité de succès avec effort élevé"
    )
    
    pi_B = st.sidebar.slider(
        "π_B (Effort faible)", 
        min_value=0.01, 
        max_value=float(pi_H-0.01), 
        value=0.3, 
        step=0.01,
        help="Probabilité de succès avec effort faible"
    )
    
    st.sidebar.subheader("Coûts d'effort")
    d_H = st.sidebar.number_input(
        "d_H (Désutilité effort élevé)", 
        min_value=0.1, 
        max_value=100.0, 
        value=5.0, 
        step=0.5,
        help="Coût de l'effort élevé pour l'agent"
    )
    
    d_B = st.sidebar.number_input(
        "d_B (Désutilité effort faible)", 
        min_value=0.0, 
        max_value=float(d_H-0.1), 
        value=2.0, 
        step=0.5,
        help="Coût de l'effort faible pour l'agent"
    )
    
    st.sidebar.subheader("Préférences de l'agent")
    u_r = st.sidebar.number_input(
        "u_r (Utilité de réserve)", 
        min_value=0.0, 
        max_value=100.0, 
        value=10.0, 
        step=1.0,
        help="Utilité que l'agent peut obtenir ailleurs"
    )
    
    gamma = st.sidebar.slider(
        "γ (Aversion au risque)", 
        min_value=0.1, 
        max_value=1.0, 
        value=0.5, 
        step=0.05,
        help="0 < γ ≤ 1. Plus γ est faible, plus l'agent est averse au risque"
    )
    
    # Bouton de calcul
    st.sidebar.markdown("---")
    calculate = st.sidebar.button(" Calculer les solutions", type="primary", use_container_width=True)
    
    # Validation des paramètres
    params_valid = True
    error_messages = []
    
    if q_bar <= q:
        params_valid = False
        error_messages.append("q̄ doit être > q")
    
    if pi_B >= pi_H:
        params_valid = False
        error_messages.append(" π_B doit être < π_H")
    
    if d_H <= d_B:
        params_valid = False
        error_messages.append(" d_H doit être > d_B")
    
    if not params_valid:
        st.error("Paramètres invalides :")
        for msg in error_messages:
            st.write(msg)
        return
    
    # Créer le modèle
    try:
        model = PrincipalAgentModel(
            q_bar=q_bar,
            q=q,
            pi_H=pi_H,
            pi_B=pi_B,
            d_H=d_H,
            d_B=d_B,
            u_r=u_r,
            gamma=gamma
        )
    except AssertionError as e:
        st.error(f"Erreur dans les paramètres : {str(e)}")
        return
    
    # Affichage des paramètres actuels
    with st.expander("Résumé des paramètres", expanded=False):
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Profit succès (q̄)", f"{q_bar:.0f}")
            st.metric("Profit échec (q)", f"{q:.0f}")
        with col2:
            st.metric("Proba succès effort H (π_H)", f"{pi_H:.2f}")
            st.metric("Proba succès effort B (π_B)", f"{pi_B:.2f}")
        with col3:
            st.metric("Coût effort H (d_H)", f"{d_H:.1f}")
            st.metric("Coût effort B (d_B)", f"{d_B:.1f}")
        
        col4, col5 = st.columns(2)
        with col4:
            st.metric("Utilité de réserve (u_r)", f"{u_r:.1f}")
        with col5:
            st.metric("Aversion au risque (γ)", f"{gamma:.2f}")
    
    if calculate or 'results_calculated' not in st.session_state:
        # Calculs
        with st.spinner("Calcul en cours..."):
            fb = solve_first_best(model, 'H')
            sb = solve_second_best(model)
            pbe = verify_perfect_bayesian_equilibrium(model, sb)
            agency_cost = fb['profit_principal'] - sb['profit_principal']
            
            # Stocker dans session_state
            st.session_state['fb'] = fb
            st.session_state['sb'] = sb
            st.session_state['pbe'] = pbe
            st.session_state['agency_cost'] = agency_cost
            st.session_state['model'] = model
            st.session_state['results_calculated'] = True
    
    if 'results_calculated' in st.session_state:
        fb = st.session_state['fb']
        sb = st.session_state['sb']
        pbe = st.session_state['pbe']
        agency_cost = st.session_state['agency_cost']
        
        # Onglets pour les résultats
        tab1, tab2, tab3, tab4 = st.tabs([
            " Résultats Principaux", 
            " Analyse Détaillée", 
            " Équilibre Bayésien",
            " Analyse de Sensibilité"
        ])
        
        with tab1:
            st.header("Résultats Principaux")
            
            # Comparaison FB vs SB
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.subheader("First-Best")
                st.info("**Information Symétrique**\n\nEffort observable")
                st.metric("Salaire (w̄ = w)", f"{fb['w_bar']:.2f}")
                st.metric("Profit Principal", f"{fb['profit_principal']:.2f}")
                st.metric("Utilité Agent", f"{fb['utility_agent']:.2f}")
                st.caption("Assurance complète")
            
            with col2:
                st.subheader("Second-Best")
                st.warning("**Aléa Moral**\n\nEffort non observable")
                st.metric("Salaire succès (w̄)", f"{sb['w_bar']:.2f}")
                st.metric("Salaire échec (w)", f"{sb['w']:.2f}")
                st.metric("Écart (w̄ - w)", f"{sb['w_bar'] - sb['w']:.2f}")
                st.metric("Profit Principal", f"{sb['profit_principal']:.2f}")
                st.caption("Risque pour l'agent")
            
            with col3:
                st.subheader("Coût d'Agence")
                st.error("**Perte d'Efficience**")
                st.metric(
                    "Coût d'agence", 
                    f"{agency_cost:.2f}",
                    delta=f"-{100 * agency_cost / fb['profit_principal']:.1f}%",
                    delta_color="inverse"
                )
                st.caption("Perte due à l'asymétrie d'information")
            
            st.markdown("---")
            
            # Graphique comparatif
            fig, ax = plt.subplots(1, 2, figsize=(12, 5), facecolor='#1a1a1a')
            
            # Graphique 1: Salaires
            categories = ['Succès\n(w̄)', 'Échec\n(w)']
            fb_wages = [fb['w_bar'], fb['w']]
            sb_wages = [sb['w_bar'], sb['w']]
            
            x = np.arange(len(categories))
            width = 0.35
            
            ax[0].bar(x - width/2, fb_wages, width, label='First-Best', 
                     color='#ffffff', alpha=0.9, edgecolor='#cccccc', linewidth=2)
            ax[0].bar(x + width/2, sb_wages, width, label='Second-Best', 
                     color='#666666', alpha=0.9, edgecolor='#999999', linewidth=2)
            ax[0].set_ylabel('Salaire', fontsize=12, color='white', fontweight='bold')
            ax[0].set_title('Comparaison des Salaires', fontsize=13, fontweight='bold', color='white', pad=15)
            ax[0].set_xticks(x)
            ax[0].set_xticklabels(categories, color='white', fontsize=11)
            ax[0].tick_params(colors='white')
            ax[0].legend(facecolor='#2d2d2d', edgecolor='#505050', labelcolor='white', fontsize=10)
            ax[0].grid(axis='y', alpha=0.2, color='#505050', linestyle='--')
            ax[0].set_facecolor('#1a1a1a')
            ax[0].spines['bottom'].set_color('#505050')
            ax[0].spines['top'].set_color('#505050')
            ax[0].spines['left'].set_color('#505050')
            ax[0].spines['right'].set_color('#505050')
            
            # Graphique 2: Profits
            regimes = ['First-Best', 'Second-Best']
            profits = [fb['profit_principal'], sb['profit_principal']]
            colors = ['#ffffff', '#666666']
            
            bars = ax[1].bar(regimes, profits, color=colors, alpha=0.9, 
                           edgecolor=['#cccccc', '#999999'], linewidth=2)
            ax[1].set_ylabel('Profit du Principal', fontsize=12, color='white', fontweight='bold')
            ax[1].set_title('Comparaison des Profits', fontsize=13, fontweight='bold', color='white', pad=15)
            ax[1].tick_params(colors='white')
            ax[1].set_xticklabels(regimes, color='white', fontsize=11)
            ax[1].grid(axis='y', alpha=0.2, color='#505050', linestyle='--')
            ax[1].set_facecolor('#1a1a1a')
            ax[1].spines['bottom'].set_color('#505050')
            ax[1].spines['top'].set_color('#505050')
            ax[1].spines['left'].set_color('#505050')
            ax[1].spines['right'].set_color('#505050')
            
            # Annoter la différence
            ax[1].annotate(
                f'Coût d\'agence\n{agency_cost:.2f}',
                xy=(0.5, sb['profit_principal'] + (fb['profit_principal'] - sb['profit_principal'])/2),
                ha='center',
                fontsize=11,
                fontweight='bold',
                color='black',
                bbox=dict(boxstyle='round,pad=0.8', facecolor='#ffffff', 
                         edgecolor='#cccccc', linewidth=2, alpha=0.95)
            )
            
            plt.tight_layout()
            st.pyplot(fig)
        
        with tab2:
            st.header("Analyse Détaillée")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader(" First-Best (Information Symétrique)")
                st.write(f"""
                **Caractéristiques :**
                - Effort imposé : **{fb['effort']}**
                - Probabilité de succès : **π = {fb['pi']:.2%}**
                - Assurance complète : **w̄ = w = {fb['w_bar']:.2f}**
                
                **Résultats :**
                - Profit Principal : **{fb['profit_principal']:.2f}**
                - Utilité Agent : **{fb['utility_agent']:.2f}** (= u_r)
                
                **Interprétation :**
                Le principal peut observer l'effort, donc il assure complètement l'agent 
                (salaire constant) tout en imposant l'effort optimal.
                """)
            
            with col2:
                st.subheader(" Second-Best (Aléa Moral)")
                st.write(f"""
                **Caractéristiques :**
                - Effort induit : **{sb['effort']}**
                - Différenciation des salaires : **w̄ - w = {sb['w_bar'] - sb['w']:.2f}**
                
                **Résultats :**
                - Profit Principal : **{sb['profit_principal']:.2f}**
                - Utilité Agent (effort H) : **{sb['utility_agent']:.2f}**
                - Utilité Agent (effort B) : **{sb['utility_agent_B']:.2f}**
                
                **Contraintes :**
                - CP (Participation) : **{sb['cp_slack']:.4f}** {' Saturée' if abs(sb['cp_slack']) < 0.01 else '✅ Satisfaite'}
                - CI (Incitation) : **{sb['ci_slack']:.4f}** {' Saturée' if abs(sb['ci_slack']) < 0.01 else '✅ Satisfaite'}
                
                **Interprétation :**
                Le principal doit créer une différence de salaire pour inciter l'agent 
                à choisir l'effort élevé, ce qui expose l'agent au risque.
                """)
            
            st.markdown("---")
            
            # Tableau comparatif détaillé
            st.subheader("Tableau Comparatif Détaillé")
            
            comparison_data = {
                'Indicateur': [
                    'Salaire en cas de succès (w̄)',
                    'Salaire en cas d\'échec (w)',
                    'Écart de salaire (w̄ - w)',
                    'Profit Principal',
                    'Utilité Agent',
                    'Assurance de l\'agent'
                ],
                'First-Best': [
                    f"{fb['w_bar']:.2f}",
                    f"{fb['w']:.2f}",
                    f"{fb['w_bar'] - fb['w']:.2f}",
                    f"{fb['profit_principal']:.2f}",
                    f"{fb['utility_agent']:.2f}",
                    "Complète "
                ],
                'Second-Best': [
                    f"{sb['w_bar']:.2f}",
                    f"{sb['w']:.2f}",
                    f"{sb['w_bar'] - sb['w']:.2f}",
                    f"{sb['profit_principal']:.2f}",
                    f"{sb['utility_agent']:.2f}",
                    "Partielle "
                ]
            }
            
            st.table(comparison_data)
        
        with tab3:
            st.header("Vérification de l'Équilibre Bayésien Parfait")
            
            if pbe['is_pbe']:
                st.success(" Le contrat Second-Best constitue un Équilibre Bayésien Parfait (PBE)")
            else:
                st.error(" Le contrat ne constitue PAS un PBE")
            
            st.markdown("---")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader(" Rationalité de l'Agent")
                st.write(f"""
                **Utilités espérées :**
                - Si accepte et choisit H : **{pbe['eu_H']:.4f}**
                - Si accepte et choisit B : **{pbe['eu_B']:.4f}**
                - Si refuse : **{model.u_r:.4f}**
                
                **Décision :**
                - Acceptation : **{'OUI' if pbe['agent_accepts'] else ' NON'}**
                - Effort optimal : **{pbe['optimal_effort']}**
                - CI satisfaite : **{'OUI' if pbe['ic_satisfied'] else ' NON'}**
                """)
            
            with col2:
                st.subheader(" Optimalité du Principal")
                st.write(f"""
                **Profits :**
                - À l'équilibre : **{pbe['equilibrium_profit']:.4f}**
                - Si déviation vers B : **{pbe['profit_deviation_B']:.4f}**
                - Si pas de contrat : **0.00**
                
                **Conclusion :**
                - Pas de déviation profitable : **{' OUI' if pbe['no_profitable_deviation'] else ' NON'}**
                """)
            
            st.markdown("---")
            
            st.subheader("Croyances Bayésiennes")
            st.write(f"""
            **Croyances a priori du Principal :**
            - P(e=H | contrat proposé) = **1.00** (anticipé correctement)
            - P(e=B | contrat proposé) = **0.00**
            
            **Mise à jour après observation :**
            - Si succès observé : P(e=H | Succès) = **{model.pi_H / model.pi_H:.2f}** = 1.00
            - Si échec observé : P(e=H | Échec) = **{(1-model.pi_H) / (1-model.pi_H):.2f}** = 1.00
            
            Les croyances sont cohérentes avec la stratégie d'équilibre.
            """)
            
            if pbe['is_pbe']:
                st.info("""
                ** Conditions du PBE vérifiées :**
                1. L'agent accepte le contrat et choisit rationnellement e=H
                2. Les croyances du principal sont cohérentes (bayésiennes)
                3. Aucun joueur ne peut dévier de manière profitable
                """)
        
        with tab4:
            st.header(" Analyse de Sensibilité")
            st.write("Impact des paramètres sur le coût d'agence")
            
            with st.spinner("Calcul de l'analyse de sensibilité..."):
                gammas, ac_gamma, pi_Bs, ac_pi = sensitivity_analysis(model)
            
            # Graphiques de sensibilité
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5), facecolor='#1a1a1a')
            
            # Graphique 1: Impact de gamma
            ax1.plot(gammas, ac_gamma, 'o-', linewidth=3, markersize=10, 
                    color='#ffffff', markerfacecolor='#cccccc', markeredgecolor='#ffffff', 
                    markeredgewidth=2, label='Coût d\'agence')
            ax1.fill_between(gammas, 0, ac_gamma, alpha=0.2, color='#ffffff')
            ax1.set_xlabel('Coefficient d\'aversion au risque (γ)', fontsize=12, 
                          color='white', fontweight='bold')
            ax1.set_ylabel('Coût d\'agence', fontsize=12, color='white', fontweight='bold')
            ax1.set_title('Impact de l\'aversion au risque', fontsize=13, 
                         fontweight='bold', color='white', pad=15)
            ax1.grid(True, alpha=0.2, color='#505050', linestyle='--')
            ax1.axhline(y=0, color='#888888', linestyle='-', alpha=0.5, linewidth=1.5)
            ax1.axvline(x=gamma, color='#ffffff', linestyle='--', alpha=0.7, 
                       linewidth=2, label=f'γ actuel = {gamma:.2f}')
            ax1.tick_params(colors='white')
            ax1.legend(facecolor='#2d2d2d', edgecolor='#505050', labelcolor='white', fontsize=10)
            ax1.set_facecolor('#1a1a1a')
            for spine in ax1.spines.values():
                spine.set_color('#505050')
                spine.set_linewidth(1.5)
            
            # Graphique 2: Impact de pi_B
            ax2.plot(pi_Bs, ac_pi, 'o-', linewidth=3, markersize=10, 
                    color='#cccccc', markerfacecolor='#666666', markeredgecolor='#cccccc', 
                    markeredgewidth=2, label='Coût d\'agence')
            ax2.fill_between(pi_Bs, 0, ac_pi, alpha=0.2, color='#cccccc')
            ax2.set_xlabel('Probabilité de succès avec effort faible (π_B)', 
                          fontsize=12, color='white', fontweight='bold')
            ax2.set_ylabel('Coût d\'agence', fontsize=12, color='white', fontweight='bold')
            ax2.set_title('Impact de la difficulté de surveillance', fontsize=13, 
                         fontweight='bold', color='white', pad=15)
            ax2.grid(True, alpha=0.2, color='#505050', linestyle='--')
            ax2.axhline(y=0, color='#888888', linestyle='-', alpha=0.5, linewidth=1.5)
            ax2.axvline(x=pi_B, color='#cccccc', linestyle='--', alpha=0.7, 
                       linewidth=2, label=f'π_B actuel = {pi_B:.2f}')
            ax2.tick_params(colors='white')
            ax2.legend(facecolor='#2d2d2d', edgecolor='#505050', labelcolor='white', fontsize=10)
            ax2.set_facecolor('#1a1a1a')
            for spine in ax2.spines.values():
                spine.set_color('#505050')
                spine.set_linewidth(1.5)
            
            plt.tight_layout()
            st.pyplot(fig)
            
            st.markdown("---")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader(" Interprétation - Aversion au risque (γ)")
                st.write("""
                **Effet de γ sur le coût d'agence :**
                - γ proche de 1 : Agent **neutre au risque** → Coût d'agence **faible**
                - γ proche de 0 : Agent **très averse au risque** → Coût d'agence **élevé**
                
                **Explication :**
                Plus l'agent est averse au risque, plus il est coûteux de l'inciter 
                à fournir l'effort élevé car il faut une prime de risque importante.
                """)
            
            with col2:
                st.subheader("Interprétation - π_B")
                st.write("""
                **Effet de π_B sur le coût d'agence :**
                - π_B élevé : Efforts **difficiles à distinguer** → Coût d'agence **élevé**
                - π_B faible : Efforts **facilement distinguables** → Coût d'agence **faible**
                
                **Explication :**
                Plus π_B est proche de π_H, plus il est difficile d'inciter l'agent 
                car la différence de performance entre les efforts est faible.
                """)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #b0b0b0; padding: 20px;'>
        <p style='font-size: 1.1rem; font-weight: 600; color: #ffffff; margin-bottom: 10px;'>
             Modèle Principal-Agent avec Aléa Moral
        </p>
        <p style='font-size: 0.95rem; color: #909090;'>
            Interface interactive développée avec Streamlit
        </p>
        <p style='font-size: 0.85rem; color: #707070; margin-top: 10px;'>
            Thème élégant noir & blanc • Design minimaliste
        </p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
