import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from typing import Dict, Tuple, List

# Configuration pour les graphiques
plt.style.use('seaborn-v0_8-darkgrid')
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 10

#definition class et modele
class PrincipalAgentModel:
    """
    Modèle Principal-Agent avec aléa moral
    
    Notations conformes au document:
    - q_bar: profit en cas de succès
    - q: profit en cas d'échec
    - w_bar: salaire en cas de succès
    - w: salaire en cas d'échec
    - pi_H: probabilité de succès avec effort élevé H
    - pi_B: probabilité de succès avec effort faible B
    - d_H: désutilité de l'effort élevé
    - d_B: désutilité de l'effort faible
    - u_r: utilité de réserve de l'agent
    """
    
    def __init__(self, q_bar=100, q=20, pi_H=0.8, pi_B=0.4, 
                 d_H=15, d_B=5, u_r=10, gamma=0.5):
        """
        Initialisation des paramètres du modèle
        
        Args:
            q_bar: profit en cas de succès
            q: profit en cas d'échec
            pi_H: probabilité de succès avec effort élevé
            pi_B: probabilité de succès avec effort faible
            d_H: désutilité effort élevé
            d_B: désutilité effort faible
            u_r: utilité de réserve
            gamma: coefficient d'aversion au risque (0 < gamma ≤ 1)
        """
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
        """
        Fonction d'utilité de l'agent: u(w) = w^gamma
        
        Args:
            w: salaire
        Returns:
            Utilité de l'agent
        """
        return w**self.gamma if w >= 0 else -np.inf
    
    def expected_utility_agent(self, w_bar: float, w: float, effort: str) -> float:
        """
        Utilité espérée de l'agent: E[u(w(y))] - d_e
        
        Args:
            w_bar: salaire en cas de succès
            w: salaire en cas d'échec
            effort: 'H' (élevé) ou 'B' (faible)
        Returns:
            Utilité espérée de l'agent
        """
        pi = self.pi_H if effort == 'H' else self.pi_B
        d = self.d_H if effort == 'H' else self.d_B
        
        return pi * self.utility_agent(w_bar) + (1 - pi) * self.utility_agent(w) - d
    
    def expected_profit_principal(self, w_bar: float, w: float, effort: str) -> float:
        """
        Profit espéré du principal: E[q(y) - w(y)]
        
        Args:
            w_bar: salaire en cas de succès
            w: salaire en cas d'échec
            effort: 'H' ou 'B'
        Returns:
            Profit espéré du principal
        """
        pi = self.pi_H if effort == 'H' else self.pi_B
        
        return pi * (self.q_bar - w_bar) + (1 - pi) * (self.q - w)
    
# first best fonction
def solve_first_best(model: PrincipalAgentModel, target_effort: str = 'H', verbose=True) -> Dict:
    """
    Résolution du cas first-best (effort observable)

    Le principal peut imposer directement l'effort et assurer complètement
    l'agent (w_bar = w) car il est averse au risque.

    Args:
        model: instance du modèle
        target_effort: 'H' ou 'B' - effort à imposer
    Returns:
        dict avec contrat optimal et profits
    """
    print("\n" + "-"*50)
    print("Résolution du first best (Information Symétrique)")
    print("-"*50)

    d = model.d_H if target_effort == 'H' else model.d_B
    pi = model.pi_H if target_effort == 'H' else model.pi_B

    # Salaire optimal: w_bar = w (assurance complète)
    # Contrainte de participation saturée: u(w) - d = u_r
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

    print(f"\nEffort imposé: {target_effort}")
    print(f"Probabilité de succès: π_{target_effort} = {pi:.2f}")
    print(f"\nContrat optimal (assurance complète):")
    print(f"  w̄ (succès) = w (échec) = {w_optimal:.2f}")
    print(f"\nRésultats:")
    print(f"  Profit espéré Principal: {profit_principal:.2f}")
    print(f"  Utilité Agent: {utility_agent:.2f} (= u_r = {model.u_r:.2f})")

    # Comparaison des efforts
    if target_effort == 'H':
        w_B = (model.u_r + model.d_B)**(1/model.gamma)
        profit_B = model.expected_profit_principal(w_B, w_B, 'B')

        print(f"\nComparaison:")
        print(f"  Si effort H imposé: Profit = {profit_principal:.2f}")
        print(f"  Si effort B imposé: Profit = {profit_B:.2f}")

        if profit_principal > profit_B:
            print(f" Le principal préfère imposer l'effort H")
        else:
            print(f" Le principal préfère imposer l'effort B")

    return {
        'effort': target_effort,
        'w_bar': w_bar_optimal,
        'w': w_optimal,
        'profit_principal': profit_principal,
        'utility_agent': utility_agent,
        'type': 'First-Best'
    }

# second best fonction
def solve_second_best(model: PrincipalAgentModel, verbose=True) -> Dict:
    """
    Résolution du cas second-best (aléa moral - effort non observable)
    
    Le principal doit satisfaire:
    1. Contrainte de Participation (CP): EU_A(H) ≥ u_r
    2. Contrainte d'Incitation (CI): EU_A(H) ≥ EU_A(B)
    
    Args:
        model: instance du modèle
    Returns:
        dict avec contrat optimal et profits
    """
    print("\n" + "-"*40)
    print("Résolution du second best (Aléa Moral)")
    print("-"*40)

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
    
    if not result.success:
        print("Attention: L'optimisation n'a pas convergé!")
        print(f"Message: {result.message}")
    
    w_bar_opt, w_opt = result.x
    
    # Calcul des résultats
    profit_principal = model.expected_profit_principal(w_bar_opt, w_opt, 'H')
    utility_agent_H = model.expected_utility_agent(w_bar_opt, w_opt, 'H')
    utility_agent_B = model.expected_utility_agent(w_bar_opt, w_opt, 'B')
    
    # Vérification des contraintes
    cp_slack = utility_agent_H - model.u_r
    ci_slack = utility_agent_H - utility_agent_B
    
    print(f"\nContrat optimal:")
    print(f"  w̄ (succès) = {w_bar_opt:.2f}")
    print(f"  w (échec)  = {w_opt:.2f}")
    print(f"  Écart w̄ - w = {w_bar_opt - w_opt:.2f}")
    
    print(f"\nRésultats:")
    print(f"  Profit espéré Principal: {profit_principal:.2f}")
    print(f"  Utilité Agent (si e=H): {utility_agent_H:.2f}")
    print(f"  Utilité Agent (si e=B): {utility_agent_B:.2f}")
    
    print(f"\nVérification des contraintes:")
    print(f"  CP (≥ 0): {cp_slack:.4f} {' contrainte saturée' if abs(cp_slack) < 0.01 else '✓'}")
    print(f"  CI (≥ 0): {ci_slack:.4f} {' contrainte saturée' if abs(ci_slack) < 0.01 else '✓'}")

    print(f"\nInterprétation:")
    if w_bar_opt > w_opt:
        print(f"  L'agent est incité à fournir l'effort élevé H")
        print(f"  Mais il supporte du risque (w̄ ≠ w)")
    
    return {
        'effort': 'H',
        'w_bar': w_bar_opt,
        'w': w_opt,
        'profit_principal': profit_principal,
        'utility_agent': utility_agent_H,
        'type': 'Second-Best',
        'cp_slack': cp_slack,
        'ci_slack': ci_slack
    }

#fonction PBE
def verify_perfect_bayesian_equilibrium(model: PrincipalAgentModel, 
                                       contract: Dict) -> Dict:
    """
    Vérifie que la solution constitue un Équilibre Bayésien Parfait (PBE)
    
    Un PBE requiert:
    1. Optimalité séquentielle: chaque joueur optimise à chaque nœud de décision
    2. Croyances bayésiennes: cohérentes avec les stratégies d'équilibre
    3. Pas de déviation profitable
    
    Args:
        model: instance du modèle
        contract: dictionnaire avec w_bar et w
    Returns:
        dict avec résultats de vérification
    """
    print("\n" + "-"*50)
    print("Vérification de l'équilibre bayésien parfait (PBE)")
    print("-"*50)
    
    w_bar = contract['w_bar']
    w = contract['w']
    
    # ==== 1. RATIONALITÉ SÉQUENTIELLE DE L'AGENT ====
    print("\n1. Vérification de la rationalité de l'Agent:")
    
    # Utilités pour chaque effort
    eu_H = model.expected_utility_agent(w_bar, w, 'H')
    eu_B = model.expected_utility_agent(w_bar, w, 'B')
    eu_reject = model.u_r  # Utilité si refus
    
    print(f"   EU_A(accepter, e=H) = {eu_H:.4f}")
    print(f"   EU_A(accepter, e=B) = {eu_B:.4f}")
    print(f"   EU_A(refuser)       = {eu_reject:.4f}")
    
    # Décision de participation
    accept_contract = max(eu_H, eu_B) >= eu_reject
    print(f"\n   → L'agent {'Accepte' if accept_contract else 'Refuse'} le contrat")
    
    # Choix d'effort optimal
    if accept_contract:
        optimal_effort = 'H' if eu_H >= eu_B else 'B'
        print(f"  Si accepté, effort optimal: {optimal_effort}")
        
        ic_satisfied = eu_H >= eu_B
        print(f"  Contrainte d'incitation: {'SATISFAITE' if ic_satisfied else 'VIOLÉE'}")
    
    # ==== 2. CROYANCES DU PRINCIPAL ====
    print("\n2. Croyances du Principal (après observation du résultat):")
    
    # Probabilités a priori
    print(f"   Croyances a priori (avant observation):")
    print(f"     P(e=H | contrat proposé) = 1 (anticipé)")
    print(f"     P(e=B | contrat proposé) = 0")
    
    # Mise à jour bayésienne après observation du succès
    if accept_contract and optimal_effort == 'H':
        # Probabilité de succès observé sachant e=H
        p_success_given_H = model.pi_H
        # Probabilité de succès observé sachant e=B
        p_success_given_B = model.pi_B
        
        # Règle de Bayes: P(H|S) = P(S|H)*P(H) / P(S)
        # Si le principal anticipe correctement e=H
        prob_H_given_success = (p_success_given_H * 1.0) / p_success_given_H
        prob_H_given_failure = ((1-p_success_given_H) * 1.0) / (1-p_success_given_H)
        
        print(f"\n   Mise à jour bayésienne:")
        print(f"     P(e=H | Succès observé) = {prob_H_given_success:.4f}")
        print(f"     P(e=H | Échec observé)  = {prob_H_given_failure:.4f}")
        print(f"     Croyances cohérentes avec stratégie d'équilibre")
    
    # ==== 3. OPTIMALITÉ DU PRINCIPAL ====
    print("\n3. Vérification de l'optimalité du Principal:")
    
    profit_equilibrium = model.expected_profit_principal(w_bar, w, optimal_effort if accept_contract else 'B')
    print(f"   Profit à l'équilibre: {profit_equilibrium:.4f}")
    
    # Test de déviations possibles
    print(f"\n   Test de déviations:")
    
    # Déviation 1: offrir un contrat qui induit e=B
    w_B_deviation = (model.u_r + model.d_B)**(1/model.gamma)
    profit_deviation_B = model.expected_profit_principal(w_B_deviation, w_B_deviation, 'B')
    print(f"     Déviation vers e=B: Profit = {profit_deviation_B:.4f}")
    
    # Déviation 2: ne pas proposer de contrat
    profit_no_contract = 0
    print(f"     Ne pas contracter: Profit = {profit_no_contract:.4f}")
    
    no_profitable_deviation = (profit_equilibrium >= profit_deviation_B) and (profit_equilibrium >= profit_no_contract)
    print(f"\n   Pas de déviation profitable: {'OUI' if no_profitable_deviation else 'NON'}")
    
    # ==== 4. TEST DE ROBUSTESSE ====
    print("\n4. Tests de robustesse:")
    
    # Test 1: Si le principal augmente w_bar
    w_bar_test = w_bar * 1.1
    eu_H_test = model.expected_utility_agent(w_bar_test, w, 'H')
    eu_B_test = model.expected_utility_agent(w_bar_test, w, 'B')
    profit_test = model.expected_profit_principal(w_bar_test, w, 'H')
    
    print(f"   Si w̄ augmente de 10%:")
    print(f"     CI reste satisfaite: {eu_H_test >= eu_B_test}")
    print(f"     Profit Principal: {profit_test:.4f} {'<' if profit_test < profit_equilibrium else '≥'} {profit_equilibrium:.4f}")
    
    # Test 2: Si le principal diminue w_bar
    w_bar_test2 = w_bar * 0.9
    eu_H_test2 = model.expected_utility_agent(w_bar_test2, w, 'H')
    eu_B_test2 = model.expected_utility_agent(w_bar_test2, w, 'B')
    
    print(f"\n   Si w̄ diminue de 10%:")
    print(f"     CI satisfaite: {eu_H_test2 >= eu_B_test2}")
    print(f"     CP satisfaite: {eu_H_test2 >= model.u_r}")
    
    # ==== CONCLUSION ====
    print("\n" + "-"*50)
    print("Conclusion sur l'Équilibre Bayésien Parfait:")
    print("-"*50)
    
    is_pbe = accept_contract and ic_satisfied and no_profitable_deviation
    
    if is_pbe:
        print("PBE vérifié")    
        print("\nLe contrat proposé constitue un PBE car:")
        print("  1. L'agent accepte et choisit rationnellement e=H")
        print("  2. Les croyances du principal sont cohérentes")
        print("  3. Aucun joueur ne peut dévier de manière profitable")
    else:
        print("Pas de PBE")
    
    return {
        'is_pbe': is_pbe,
        'agent_accepts': accept_contract,
        'optimal_effort': optimal_effort if accept_contract else None,
        'ic_satisfied': ic_satisfied if accept_contract else False,
        'no_profitable_deviation': no_profitable_deviation,
        'equilibrium_profit': profit_equilibrium,
        'eu_H': eu_H,
        'eu_B': eu_B
    }

#analyse de sensibilité
def compare_regimes(model: PrincipalAgentModel) -> Tuple[Dict, Dict, float]:
    """
    Compare les résultats First-Best vs Second-Best
    
    Args:
        model: instance du modèle
    Returns:
        tuple (first_best, second_best, agency_cost)
    """
    print("\n" + "="*50)
    print("Comparaison des régimes First-Best vs Second-Best")
    print("="*50)
    
    fb = solve_first_best(model, 'H')
    sb = solve_second_best(model)
    
    # Coût d'agence
    agency_cost = fb['profit_principal'] - sb['profit_principal']
    
    print(f"\nCoût d'agence (perte due à l'asymétrie d'information):")
    print(f"  Profit FB - Profit SB = {agency_cost:.2f}")
    print(f"  Perte relative: {100 * agency_cost / fb['profit_principal']:.2f}%")
    
    print(f"\nÉcart de salaire:")
    print(f"  FB: w̄ - w = {fb['w_bar'] - fb['w']:.2f} (assurance complète)")
    print(f"  SB: w̄ - w = {sb['w_bar'] - sb['w']:.2f} (incitation)")
    
    return fb, sb, agency_cost
def sensitivity_analysis(model: PrincipalAgentModel):
    """
    Analyse de sensibilité: impact des paramètres sur le coût d'agence
    
    Args:
        model: instance du modèle
    Returns:
        figure matplotlib
    """
    print("\n" + "="*40)
    print("Analyse de sensibilité du coût d'agence")
    print("="*40)
    
    # Sauvegarder paramètres originaux
    original_gamma = model.gamma
    original_pi_B = model.pi_B
    
    # Analyse 1: Impact de l'aversion au risque (gamma)
    gammas = np.linspace(0.3, 1.0, 8)
    agency_costs_gamma = []
    
    print("\n1. Impact de l'aversion au risque (γ):")
    for g in gammas:
        model.gamma = g
        fb = solve_first_best(model, 'H')
        sb = solve_second_best(model)
        ac = fb['profit_principal'] - sb['profit_principal']
        agency_costs_gamma.append(ac)
        print(f"   γ = {g:.2f}: Coût d'agence = {ac:.2f}")
    
    # Analyse 2: Impact de pi_B
    model.gamma = original_gamma
    pi_Bs = np.linspace(0.2, 0.7, 8)
    agency_costs_pi = []
    
    print("\n2. Impact de π_B (effort faible):")
    for pi_b in pi_Bs:
        if pi_b < model.pi_H:
            model.pi_B = pi_b
            fb = solve_first_best(model, 'H')
            sb = solve_second_best(model)
            ac = fb['profit_principal'] - sb['profit_principal']
            agency_costs_pi.append(ac)
            print(f"   π_B = {pi_b:.2f}: Coût d'agence = {ac:.2f}")
    
    # Restaurer paramètres
    model.gamma = original_gamma
    model.pi_B = original_pi_B
    
    # Graphiques
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    ax1.plot(gammas, agency_costs_gamma, 'b-o', linewidth=2, markersize=8)
    ax1.set_xlabel('Coefficient d\'aversion au risque (γ)', fontsize=12)
    ax1.set_ylabel('Coût d\'agence', fontsize=12)
    ax1.set_title('Impact de l\'aversion au risque', fontsize=13, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.axhline(y=0, color='r', linestyle='--', alpha=0.5)
    
    ax2.plot(pi_Bs[:len(agency_costs_pi)], agency_costs_pi, 'g-o', linewidth=2, markersize=8)
    ax2.set_xlabel('Probabilité de succès avec effort faible (π_B)', fontsize=12)
    ax2.set_ylabel('Coût d\'agence', fontsize=12)
    ax2.set_title('Impact de la difficulté de surveillance', fontsize=13, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.axhline(y=0, color='r', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    
    return fig

#exemple
# 1. Création du modèle
model = PrincipalAgentModel(
    q_bar=600,    # Profit si succès
    q=100,         # Profit si échec
    pi_H=0.7,     # pi_B=0.7 Proba succès si effort faible  pi_H=0.7 donne le PBE
    pi_B=0.3,     # pi_B=0.3 Proba succès si effort faible  pi_B=0.3 donne le PBE
    d_H=5,        # Coût effort élevé
    d_B=2,        # Coût effort faible
    u_r=10,       # Utilité de réserve
    gamma=0.5     # Aversion au risque
)

# 2. Résoudre First-Best
fb = solve_first_best(model, 'H')

# 3. Résoudre Second-Best
sb = solve_second_best(model)

# 4. Vérifier l'équilibre bayésien
pbe_results = verify_perfect_bayesian_equilibrium(model, sb)

# 5. Comparer les régimes
fb, sb, agency_cost = compare_regimes(model)

# 6. Analyse de sensibilité
fig = sensitivity_analysis(model)
plt.show()
