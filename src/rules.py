def classify_access_risk_from_input(row):
    participant_count = float(row.get("participant_count", 0))
    active_provider_count = float(row.get("active_provider_count", 1))
    utilisation_rate = float(row.get("utilisation_rate", 0))
    average_support_budget = float(row.get("average_support_budget", 0))

    if active_provider_count <= 0:
        active_provider_count = 1

    participant_to_provider_ratio = participant_count / active_provider_count
    utilisation_gap = 100 - utilisation_rate

    risk_score = 0

    # Risk from high demand compared with provider supply
    if participant_to_provider_ratio >= 80:
        risk_score += 3
    elif participant_to_provider_ratio >= 40:
        risk_score += 2
    elif participant_to_provider_ratio >= 20:
        risk_score += 1

    # Risk from low plan utilisation
    if utilisation_gap >= 50:
        risk_score += 3
    elif utilisation_gap >= 30:
        risk_score += 2
    elif utilisation_gap >= 20:
        risk_score += 1

    # Risk from low number of active providers
    if active_provider_count <= 5:
        risk_score += 2
    elif active_provider_count <= 15:
        risk_score += 1

    # Risk from high support budget, used as a proxy for complex needs
    if average_support_budget >= 60000:
        risk_score += 1

    if risk_score >= 5:
        return "High Risk"
    elif risk_score >= 2:
        return "Medium Risk"
    else:
        return "Low Risk"


def explain_access_risk(row):
    reasons = []

    participant_count = float(row.get("participant_count", 0))
    active_provider_count = float(row.get("active_provider_count", 1))
    utilisation_rate = float(row.get("utilisation_rate", 0))
    average_support_budget = float(row.get("average_support_budget", 0))

    if active_provider_count <= 0:
        active_provider_count = 1

    participant_to_provider_ratio = participant_count / active_provider_count
    utilisation_gap = 100 - utilisation_rate

    if participant_to_provider_ratio >= 80:
        reasons.append("the participant-to-provider ratio is very high")
    elif participant_to_provider_ratio >= 40:
        reasons.append("the participant-to-provider ratio is moderately high")
    elif participant_to_provider_ratio < 10:
        reasons.append("the participant-to-provider ratio is low, suggesting better provider availability")

    if utilisation_gap >= 50:
        reasons.append("the utilisation gap is very large, meaning the plan budget may not be used effectively")
    elif utilisation_gap >= 30:
        reasons.append("there is a moderate utilisation gap")
    elif utilisation_gap < 20:
        reasons.append("plan utilisation is relatively strong")

    if active_provider_count <= 5:
        reasons.append("there are very few active providers")
    elif active_provider_count >= 50:
        reasons.append("there is relatively strong provider availability")

    if average_support_budget >= 60000:
        reasons.append("the average support budget is high, suggesting more complex support needs")

    if not reasons:
        reasons.append("the input values do not show major access-risk signals")

    return "This risk level was assigned because " + "; ".join(reasons) + "."


def rule_based_recommendation(row):
    recommendations = []

    disability = str(row.get("disability_type", "")).lower()
    utilisation_rate = float(row.get("utilisation_rate", 0))
    participant_count = float(row.get("participant_count", 0))
    active_provider_count = float(row.get("active_provider_count", 1))
    average_support_budget = float(row.get("average_support_budget", 0))

    if active_provider_count <= 0:
        active_provider_count = 1

    participant_to_provider_ratio = participant_count / active_provider_count
    utilisation_gap = 100 - utilisation_rate

    if utilisation_gap >= 30:
        recommendations.append("Support coordination or plan management review")

    if participant_to_provider_ratio >= 40:
        recommendations.append("Provider availability review")

    if "autism" in disability or "development" in disability:
        recommendations.append("Capacity Building - Improved Daily Living")

    if "intellectual" in disability:
        recommendations.append("Capacity Building - Social and Community Participation")

    if "psychosocial" in disability:
        recommendations.append("Psychosocial recovery and community participation support")

    if "physical" in disability or "mobility" in disability:
        recommendations.append("Assistive technology or transport support")

    if "hearing" in disability:
        recommendations.append("Assistive technology and communication support")

    if "visual" in disability or "vision" in disability:
        recommendations.append("Assistive technology and orientation support")

    if "brain" in disability:
        recommendations.append("Capacity Building and daily living support review")

    if "neurological" in disability:
        recommendations.append("Therapy, assistive technology, and daily living support review")

    if average_support_budget >= 60000:
        recommendations.append("High-intensity support review")

    if not recommendations:
        recommendations.append("Maintain current support planning and continue monitoring utilisation")

    return recommendations