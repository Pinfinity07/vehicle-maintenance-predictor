"""LangGraph agentic workflow for conversational fleet management analysis."""

import json
import os
import re
from typing import TypedDict

from langgraph.graph import StateGraph, END
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage

from model_utils import predict, VEHICLE_MODELS, FUEL_TYPES, TRANSMISSION_TYPES, OWNER_TYPES
from rag_utils import get_rag, get_modifier_rag


# ---------------------------------------------------------------------------
# Required features for ML prediction
# ---------------------------------------------------------------------------

REQUIRED_FEATURES = {
    "Vehicle_Model": {"type": "categorical", "options": VEHICLE_MODELS, "question": "What type of vehicle is it? (Truck, Van, Bus, SUV, Sedan)"},
    "Mileage": {"type": "numeric", "range": "5000-150000 km", "question": "What is the current mileage in km?"},
    "Maintenance_History": {"type": "categorical", "options": ["Poor", "Average", "Good"], "question": "How would you rate the overall maintenance history? (Poor, Average, or Good)"},
    "Reported_Issues": {"type": "numeric", "range": "0-10", "question": "How many issues have been reported recently?"},
    "Vehicle_Age": {"type": "numeric", "range": "1-20 years", "question": "How old is the vehicle (in years)?"},
    "Fuel_Type": {"type": "categorical", "options": FUEL_TYPES, "question": "What fuel type does it use? (Electric, Diesel, Petrol, Hybrid)"},
    "Transmission_Type": {"type": "categorical", "options": TRANSMISSION_TYPES, "question": "Is it Automatic or Manual transmission?"},
    "Engine_Size": {"type": "numeric", "range": "1000-3500 cc", "question": "What is the engine size in cc?"},
    "Odometer_Reading": {"type": "numeric", "range": "5000-250000 km", "question": "What is the current odometer reading in km?"},
    "Owner_Type": {"type": "categorical", "options": OWNER_TYPES, "question": "Is this the First, Second, or Third owner?"},
    "Insurance_Premium": {"type": "numeric", "range": "8000-35000", "question": "What is the annual insurance premium?"},
    "Service_History": {"type": "numeric", "range": "0-20", "question": "How many times has the vehicle been serviced?"},
    "Accident_History": {"type": "numeric", "range": "0-5", "question": "How many accidents has the vehicle been in?"},
    "Fuel_Efficiency": {"type": "numeric", "range": "8-22 km/l", "question": "What is the fuel efficiency (km per liter)?"},
    "Tire_Condition": {"type": "categorical", "options": ["Worn Out", "Good", "New"], "question": "What is the current tire condition? (Worn Out, Good, or New)"},
    "Brake_Condition": {"type": "categorical", "options": ["Worn Out", "Good", "New"], "question": "What is the current brake condition? (Worn Out, Good, or New)"},
    "Battery_Status": {"type": "categorical", "options": ["Weak", "Good", "Strong"], "question": "What is the battery status? (Weak, Good, or Strong)"},
    "Last_Service_Date_days": {"type": "numeric", "range": "30-900 days", "question": "How many days ago was the last service?"},
    "Warranty_Expiry_Date_days": {"type": "numeric", "range": "-400 to 800 days", "question": "How many days until warranty expires? (negative if already expired)"},
}


# ---------------------------------------------------------------------------
# State definition
# ---------------------------------------------------------------------------

class FleetState(TypedDict, total=False):
    user_message: str             # Latest user message
    messages: list[dict]          # Chat history [{role, content}]
    collected_features: dict      # Features extracted so far
    new_features: dict            # Features extracted in this turn
    extra_info: list[str]         # Extra info user mentioned (not model features)
    intent: str                   # "question" | "info" | "mixed" | "offtopic"
    risk_modifiers: list[str]     # Retrieved risk modifier docs
    prediction: dict              # ML prediction result
    risk_level: str
    guidelines: list[str]         # Retrieved maintenance guidelines
    report: str                   # Final generated report
    response: str                 # Final text response to user
    phase: str                    # "collecting" | "analyzing" | "done"
    api_key: str
    error: str


# ---------------------------------------------------------------------------
# LLM helper
# ---------------------------------------------------------------------------

def _get_llm(api_key: str | None = None):
    key = api_key or os.environ.get("GROQ_API_KEY", "")
    if not key:
        return None
    return ChatGroq(
        model="llama-3.3-70b-versatile",
        api_key=key,
        temperature=0.3,
        max_tokens=2048,
    )


# ---------------------------------------------------------------------------
# Feature extraction prompt
# ---------------------------------------------------------------------------

EXTRACTION_SYSTEM = """You are a vehicle data extraction assistant. Your job is to extract
vehicle features from the user's natural language description.

The required features are:
{features_schema}

From the user's message, extract any features you can identify. Also note any EXTRA
information the user mentions that doesn't map to the features above (like driving
conditions, recent symptoms, modifications, climate, driver behavior, etc).

Respond in this EXACT JSON format (no markdown, no explanation):
{{
    "extracted": {{
        "feature_name": value,
        ...
    }},
    "extra_info": ["any extra info mentioned that doesn't map to a feature"],
    "confidence": "high/medium/low"
}}

Rules:
- For categorical features, map to the closest valid option.
- For numeric features, extract the number.
- **Negation / absence handling:** if the user explicitly says they don't have something
  or it is none/zero/never, extract 0 for these numeric features:
    * "no insurance" / "don't have insurance" / "uninsured" -> Insurance_Premium: 0
    * "no accidents" / "never had an accident" / "accident-free" -> Accident_History: 0
    * "no issues" / "no problems reported" -> Reported_Issues: 0
    * "never serviced" / "no service history" -> Service_History: 0
    * "warranty expired" without a date -> leave missing (ask user)
- If the user says something vague like "old truck", infer Vehicle_Model=Truck but don't
  guess the exact age - leave it as missing.
- "extra_info" captures things like: "I drive on rough roads", "car makes grinding noise",
  "I do a lot of highway driving", "it was in a flood", "check engine light is on", etc.
- Only extract what you're confident about. Don't guess.
- Return ONLY the JSON, nothing else."""


def _build_features_schema() -> str:
    lines = []
    for name, info in REQUIRED_FEATURES.items():
        if info["type"] == "categorical":
            lines.append(f"- {name}: one of {info['options']}")
        else:
            lines.append(f"- {name}: number ({info['range']})")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Conversational agent functions
# ---------------------------------------------------------------------------

def extract_features(messages: list[dict], api_key: str | None,
                     existing_features: dict) -> tuple[dict, list[str]]:
    """Use LLM to extract features from the latest user message."""
    llm = _get_llm(api_key)
    if llm is None:
        return {}, []

    schema = _build_features_schema()
    system = EXTRACTION_SYSTEM.format(features_schema=schema)

    # Include conversation context
    conversation = ""
    for msg in messages[-6:]:  # Last 6 messages for context
        role = msg["role"]
        conversation += f"{role}: {msg['content']}\n"

    if existing_features:
        conversation += f"\nAlready collected features: {json.dumps(existing_features)}\n"

    conversation += "\nExtract features from the user's latest message above."

    try:
        response = llm.invoke([
            SystemMessage(content=system),
            HumanMessage(content=conversation),
        ])

        text = response.content.strip()
        # Clean markdown code blocks if present
        if text.startswith("```"):
            text = text.split("\n", 1)[1]
            if text.endswith("```"):
                text = text[:-3]
            text = text.strip()

        result = json.loads(text)
        extracted = result.get("extracted", {})
        extra = result.get("extra_info", [])

        # Validate extracted features
        validated = {}
        for key, value in extracted.items():
            if key in REQUIRED_FEATURES:
                info = REQUIRED_FEATURES[key]
                if info["type"] == "categorical":
                    if str(value) in [str(o) for o in info["options"]]:
                        validated[key] = value
                else:
                    try:
                        validated[key] = float(value) if "." in str(value) else int(value)
                    except (ValueError, TypeError):
                        pass

        return validated, extra
    except Exception:
        return {}, []


NEGATION_PATTERNS = {
    "Insurance_Premium": [
        r"\bno\s+insurance\b",
        r"\bdon'?t\s+have\s+(any\s+)?insurance\b",
        r"\buninsured\b",
        r"\bwithout\s+insurance\b",
        r"\bno\s+insurance\s+(premium|policy)\b",
    ],
    "Accident_History": [
        r"\bno\s+accidents?\b",
        r"\bnever\s+(had|been\s+in)\s+an?\s+accident\b",
        r"\baccident[- ]free\b",
        r"\bzero\s+accidents?\b",
    ],
    "Reported_Issues": [
        r"\bno\s+(reported\s+)?issues?\b",
        r"\bno\s+problems?\b",
        r"\bno\s+complaints?\b",
    ],
    "Service_History": [
        r"\bnever\s+serviced\b",
        r"\bno\s+service\s+history\b",
    ],
}


def apply_negation_heuristics(user_message: str, collected: dict) -> dict:
    """Detect explicit negations ('no insurance', 'no accidents') and set to 0.

    Runs as a deterministic fallback in case the LLM misses the negation.
    Only fills features that aren't already collected.
    """
    text = user_message.lower()
    new = {}
    for feature, patterns in NEGATION_PATTERNS.items():
        if feature in collected:
            continue
        for pat in patterns:
            if re.search(pat, text):
                new[feature] = 0
                break
    return new


def get_missing_features(collected: dict) -> list[str]:
    """Return list of feature names not yet collected."""
    return [k for k in REQUIRED_FEATURES if k not in collected]


def build_followup_question(missing: list[str], collected: dict) -> str:
    """Build a natural follow-up question asking for missing features."""
    if not missing:
        return ""

    # Group related features for natural questions
    groups = {
        "basic": ["Vehicle_Model", "Fuel_Type", "Transmission_Type", "Engine_Size", "Owner_Type"],
        "usage": ["Mileage", "Vehicle_Age", "Odometer_Reading", "Fuel_Efficiency"],
        "condition": ["Tire_Condition", "Brake_Condition", "Battery_Status", "Maintenance_History"],
        "history": ["Reported_Issues", "Service_History", "Accident_History"],
        "service": ["Last_Service_Date_days", "Warranty_Expiry_Date_days", "Insurance_Premium"],
    }

    # Find which groups have missing features
    questions = []
    asked = set()
    for group_name, group_features in groups.items():
        group_missing = [f for f in group_features if f in missing and f not in asked]
        if group_missing:
            if len(group_missing) <= 3:
                for f in group_missing:
                    questions.append(REQUIRED_FEATURES[f]["question"])
                    asked.add(f)
            else:
                # Ask a batch question
                labels = [f.replace("_", " ").lower() for f in group_missing]
                questions.append(f"Could you tell me about the vehicle's {', '.join(labels)}?")
                asked.update(group_missing)

        if len(questions) >= 3:
            break

    n_remaining = len(missing) - len(asked)
    summary = "\n".join(f"- {q}" for q in questions)

    if n_remaining > 0:
        summary += f"\n\n(I still need {n_remaining} more details after these)"

    return summary


INTENT_SYSTEM = """You are an intent classifier for a vehicle maintenance assistant.

Classify the user's latest message into ONE of:
- "question": user is asking a question (what options exist, what a field means, what values are valid, clarification, help, examples, "I don't know X", "what should I put", etc.)
- "info": user is providing vehicle details/information
- "mixed": user is providing some info AND asking a question
- "offtopic": user is asking something unrelated to vehicles/fleet maintenance

Respond in EXACT JSON (no markdown):
{"intent": "question|info|mixed|offtopic", "topic": "<short phrase of what they are asking about, or empty>"}
"""


def classify_intent(user_message: str, api_key: str | None) -> dict:
    """Classify whether the user is asking a question or providing info."""
    llm = _get_llm(api_key)
    if llm is None:
        return {"intent": "info", "topic": ""}
    try:
        response = llm.invoke([
            SystemMessage(content=INTENT_SYSTEM),
            HumanMessage(content=user_message),
        ])
        text = response.content.strip()
        if text.startswith("```"):
            text = text.split("\n", 1)[1]
            if text.endswith("```"):
                text = text[:-3]
            text = text.strip()
        return json.loads(text)
    except Exception:
        return {"intent": "info", "topic": ""}


ANSWER_SYSTEM = """You are a helpful vehicle maintenance assistant. The user is asking
a question about vehicle features they need to provide. Answer their question concisely
using the feature schema below. If they don't know a value, help them pick a reasonable
option by explaining what each choice means.

## Feature schema (valid options and ranges)
{features_schema}

Rules:
- Answer ONLY about vehicle features / maintenance context. If the question is off-topic,
  politely say you can only help with vehicle maintenance topics.
- Be concise (3-6 lines). Use bullet points for options.
- If they're asking "what are the options for X", list the valid options with a brief
  one-line explanation for each.
- If they don't know a value, suggest they pick the closest/most common option or provide
  a typical default (e.g., for Engine Size: Sedan ~1500-2000cc, SUV ~2000-2500cc, Truck ~2500-3500cc).
- End with a short nudge back to collecting info, e.g., "Which one fits your vehicle?"
"""


def answer_question(user_message: str, missing: list[str], collected: dict,
                    api_key: str | None) -> str:
    """Answer the user's question about features / options."""
    llm = _get_llm(api_key)
    schema = _build_features_schema()

    if llm is None:
        # Fallback — just dump the schema for missing features
        lines = ["Here are the valid options:\n"]
        for f in missing[:5]:
            info = REQUIRED_FEATURES[f]
            if info["type"] == "categorical":
                lines.append(f"- **{f.replace('_', ' ')}**: {', '.join(str(o) for o in info['options'])}")
            else:
                lines.append(f"- **{f.replace('_', ' ')}**: number in range {info['range']}")
        return "\n".join(lines)

    try:
        response = llm.invoke([
            SystemMessage(content=ANSWER_SYSTEM.format(features_schema=schema)),
            HumanMessage(content=f"User question: {user_message}\n\n"
                         f"Still missing features: {missing}\n"
                         f"Already collected: {list(collected.keys())}"),
        ])
        return response.content.strip()
    except Exception:
        return ("The valid options depend on the feature. For example, Owner Type is one of: "
                "First, Second, Third. Feel free to ask about any specific field.")


def retrieve_risk_modifiers(extra_info: list[str]) -> list[str]:
    """Retrieve risk modifier docs based on extra info the user provided."""
    if not extra_info:
        return []

    query = ". ".join(extra_info)
    try:
        modifier_rag = get_modifier_rag()
        return modifier_rag.retrieve(query, k=3)
    except Exception:
        return []


def apply_risk_modifiers(prediction: dict, modifiers: list[str],
                         extra_info: list[str], api_key: str | None) -> dict:
    """Use LLM to adjust risk based on extra info and modifier guidelines."""
    llm = _get_llm(api_key)
    if llm is None or not extra_info:
        return prediction

    prompt = f"""Based on the ML model prediction and the additional information provided
by the user, determine if the risk should be adjusted.

## ML Prediction
- Probability: {prediction.get('probability', 0)}
- Risk Level: {prediction.get('risk_level', 'UNKNOWN')}

## Additional User Information
{json.dumps(extra_info, indent=2)}

## Risk Modifier Guidelines
{chr(10).join(modifiers)}

## Task
Determine a risk adjustment factor. Respond in EXACT JSON format:
{{
    "adjustment_factor": <float between 0.8 and 2.0>,
    "reason": "<one line explanation>",
    "adjusted_probability": <float between 0 and 1>,
    "adjusted_risk_level": "<CRITICAL/HIGH/MODERATE/LOW>"
}}

Rules:
- adjustment_factor > 1.0 means HIGHER risk
- adjustment_factor < 1.0 means LOWER risk (only if extra info is reassuring)
- Cap adjusted_probability at 1.0
- Return ONLY JSON, no markdown."""

    try:
        response = llm.invoke([
            SystemMessage(content="You are a vehicle risk assessment specialist. Be precise with numbers."),
            HumanMessage(content=prompt),
        ])
        text = response.content.strip()
        if text.startswith("```"):
            text = text.split("\n", 1)[1]
            if text.endswith("```"):
                text = text[:-3]
            text = text.strip()

        adj = json.loads(text)
        prediction = {**prediction}
        prediction["original_probability"] = prediction["probability"]
        prediction["original_risk_level"] = prediction["risk_level"]
        prediction["probability"] = min(1.0, round(adj.get("adjusted_probability", prediction["probability"]), 4))
        prediction["risk_level"] = adj.get("adjusted_risk_level", prediction["risk_level"])
        prediction["adjustment_factor"] = adj.get("adjustment_factor", 1.0)
        prediction["adjustment_reason"] = adj.get("reason", "")
        return prediction
    except Exception:
        return prediction


# ---------------------------------------------------------------------------
# Report generation (same as before but with extra info context)
# ---------------------------------------------------------------------------

def generate_report(vehicle_data: dict, prediction: dict, guidelines: list[str],
                    extra_info: list[str], risk_modifiers: list[str],
                    api_key: str | None) -> str:
    """Generate structured fleet management report."""
    llm = _get_llm(api_key)

    vehicle_summary = "\n".join(
        f"- {k.replace('_', ' ').title()}: {v}"
        for k, v in vehicle_data.items()
    )
    guidelines_text = "\n\n---\n\n".join(guidelines) if guidelines else "No guidelines retrieved."
    extra_text = "\n".join(f"- {e}" for e in extra_info) if extra_info else "None"
    risk = prediction.get("risk_level", "UNKNOWN")
    proba = prediction.get("probability", 0)

    if llm is None:
        return _generate_fallback_report(vehicle_data, prediction, guidelines, extra_info)

    # Check for adjustment
    adj_note = ""
    if "original_probability" in prediction:
        adj_note = f"""
**Risk Adjustment Applied:**
- Original ML Probability: {prediction['original_probability']:.1%}
- Adjusted Probability: {prediction['probability']:.1%}
- Adjustment Factor: {prediction.get('adjustment_factor', 1.0):.2f}x
- Reason: {prediction.get('adjustment_reason', 'N/A')}
"""

    prompt = f"""You are an AI Fleet Management Assistant. Based on the vehicle data,
ML prediction results, additional context from the user, and retrieved maintenance
guidelines, generate a structured fleet management report.

## Vehicle Data
{vehicle_summary}

## Additional Context from User
{extra_text}

## ML Prediction Results
- Needs Maintenance: {"YES" if prediction.get("needs_maintenance") else "NO"}
- Maintenance Probability: {proba:.1%}
- Risk Level: {risk}
- Top Contributing Factors: {prediction.get("top_features", [])}
{adj_note}

## Retrieved Maintenance Guidelines
{guidelines_text}

---

Generate a report with EXACTLY these sections:

### HEALTH SUMMARY
Provide a clear assessment of the vehicle's current health status and risk level.
Explain what the ML model found and why. If risk was adjusted based on additional
context, explain the adjustment.

### ACTION PLAN
List specific maintenance actions needed, ordered by priority (critical first).
Include estimated timelines for each action.

### MAINTENANCE SCHEDULE
Provide a recommended maintenance schedule going forward (immediate, 30 days,
90 days, 6 months).

### COST ESTIMATE
Provide rough cost ranges for recommended maintenance actions.

### SOURCES
List which maintenance guidelines informed this report.

### DISCLAIMER
Include an operational safety disclaimer that this is an AI-assisted recommendation
and should be verified by a certified mechanic before acting on critical maintenance.

Keep the report professional, actionable, and concise."""

    try:
        response = llm.invoke([
            SystemMessage(content="You are a professional fleet management AI assistant. "
                          "Provide accurate, actionable maintenance recommendations. "
                          "Always prioritize safety. Be specific with timelines and actions."),
            HumanMessage(content=prompt),
        ])
        return response.content
    except Exception as e:
        report = _generate_fallback_report(vehicle_data, prediction, guidelines, extra_info)
        report += f"\n\n> *Note: LLM generation failed ({e}). Showing rule-based report.*"
        return report


def _generate_fallback_report(vehicle_data: dict, prediction: dict,
                              guidelines: list[str], extra_info: list[str]) -> str:
    """Rule-based report when LLM is unavailable."""
    risk = prediction.get("risk_level", "UNKNOWN")
    proba = prediction.get("probability", 0)
    needs = prediction.get("needs_maintenance", 0)
    v = vehicle_data

    actions = []
    if v.get("Tire_Condition") == "Worn Out":
        actions.append("**[CRITICAL]** Replace worn-out tires immediately")
    if v.get("Brake_Condition") == "Worn Out":
        actions.append("**[CRITICAL]** Inspect and replace brake pads/rotors")
    if v.get("Battery_Status") == "Weak":
        actions.append("**[HIGH]** Test battery voltage and replace if below 12.4V")
    if v.get("Maintenance_History") == "Poor":
        actions.append("**[HIGH]** Schedule comprehensive service — poor maintenance history")
    if v.get("Last_Service_Date_days", 0) > 300:
        actions.append("**[HIGH]** Overdue for service — last service was over 300 days ago")
    if v.get("Mileage", 0) > 100000:
        actions.append("**[MODERATE]** High-mileage inspection recommended")
    if v.get("Accident_History", 0) > 1:
        actions.append("**[MODERATE]** Structural inspection due to accident history")
    if not actions:
        actions.append("No critical actions needed — continue regular maintenance schedule")

    extra_section = ""
    if extra_info:
        extra_section = "\n### ADDITIONAL CONTEXT CONSIDERED\n"
        extra_section += "\n".join(f"- {e}" for e in extra_info)
        extra_section += "\n"

    adj_section = ""
    if "original_probability" in prediction:
        adj_section = f"""
> **Risk Adjusted:** Original {prediction['original_probability']:.1%} → Adjusted {proba:.1%}
> Reason: {prediction.get('adjustment_reason', 'Additional context factors')}
"""

    return f"""### HEALTH SUMMARY
- **Risk Level:** {risk}
- **Maintenance Needed:** {"YES" if needs else "NO"}
- **Probability:** {proba:.1%}
- **Vehicle:** {v.get("Vehicle_Model", "N/A")} | Age: {v.get("Vehicle_Age", "N/A")} years | Mileage: {v.get("Mileage", "N/A"):,} km
{adj_section}
{extra_section}
### ACTION PLAN
{chr(10).join(f"{i+1}. {a}" for i, a in enumerate(actions))}

### MAINTENANCE SCHEDULE
- **Immediate:** Address all CRITICAL items before next trip
- **30 days:** Complete HIGH priority items
- **90 days:** Address MODERATE items and schedule routine inspection
- **6 months:** Full comprehensive service

### SOURCES
Retrieved {len(guidelines)} relevant maintenance guidelines from the knowledge base.

### DISCLAIMER
This is an AI-generated maintenance recommendation based on predictive analytics.
All critical maintenance decisions should be verified by a certified automotive
technician. Do not rely solely on this report for safety-critical decisions.
"""


# ---------------------------------------------------------------------------
# Retrieve maintenance guidelines
# ---------------------------------------------------------------------------

def retrieve_guidelines(vehicle_data: dict, risk_level: str) -> list[str]:
    """RAG retrieval of relevant maintenance guidelines."""
    v = vehicle_data
    query_parts = [f"vehicle maintenance for {v.get('Vehicle_Model', 'vehicle')}"]
    query_parts.append(f"risk level {risk_level}")

    tire = v.get("Tire_Condition", "")
    brake = v.get("Brake_Condition", "")
    battery = v.get("Battery_Status", "")

    if tire == "Worn Out":
        query_parts.append("tire replacement worn out")
    if brake == "Worn Out":
        query_parts.append("brake maintenance worn brakes")
    if battery == "Weak":
        query_parts.append("battery replacement weak battery")
    if v.get("Mileage", 0) > 100000:
        query_parts.append("high mileage vehicle maintenance")
    if v.get("Last_Service_Date_days", 0) > 300:
        query_parts.append("overdue service maintenance scheduling")

    query = ". ".join(query_parts)

    try:
        rag = get_rag()
        return rag.retrieve(query, k=4)
    except Exception:
        return []


# ---------------------------------------------------------------------------
# LangGraph StateGraph — the agentic workflow
# ---------------------------------------------------------------------------

def node_classify_intent(state: FleetState) -> FleetState:
    """Node: classify user intent (question / info / mixed / offtopic)."""
    result = classify_intent(state["user_message"], state.get("api_key"))
    return {"intent": result.get("intent", "info")}


def node_extract_features(state: FleetState) -> FleetState:
    """Node: extract vehicle features + apply negation heuristics."""
    messages = state["messages"]
    api_key = state.get("api_key")
    collected = state["collected_features"]

    new_features, new_extra = extract_features(messages, api_key, collected)
    negation_features = apply_negation_heuristics(state["user_message"], collected)
    for k, v in negation_features.items():
        new_features.setdefault(k, v)

    return {
        "new_features": new_features,
        "collected_features": {**collected, **new_features},
        "extra_info": state["extra_info"] + [
            e for e in new_extra if e and e not in state["extra_info"]
        ],
    }


def node_answer_question(state: FleetState) -> FleetState:
    """Node: answer the user's question about feature options."""
    missing = get_missing_features(state["collected_features"])
    answer = answer_question(
        state["user_message"], missing, state["collected_features"], state.get("api_key")
    )
    n_collected = len(REQUIRED_FEATURES) - len(missing)
    header = f"**[{n_collected}/{len(REQUIRED_FEATURES)} features collected]**\n\n"
    new_features = state.get("new_features", {})
    if new_features:
        summary = ", ".join(f"{k.replace('_', ' ')}: {v}" for k, v in new_features.items())
        header += f"Got it, I've noted: {summary}\n\n"
    return {"response": header + answer, "phase": "collecting"}


def node_offtopic(state: FleetState) -> FleetState:
    """Node: politely redirect off-topic questions."""
    missing = get_missing_features(state["collected_features"])
    reply = ("I can only help with vehicle maintenance and fleet management questions. "
             "Could we get back to your vehicle? " +
             (build_followup_question(missing, state["collected_features"]) if missing else ""))
    return {"response": reply, "phase": "collecting" if missing else "done"}


def node_ask_followup(state: FleetState) -> FleetState:
    """Node: ask for missing features."""
    missing = get_missing_features(state["collected_features"])
    n_collected = len(REQUIRED_FEATURES) - len(missing)
    progress = f"**[{n_collected}/{len(REQUIRED_FEATURES)} features collected]**\n\n"
    new_features = state.get("new_features", {})
    if new_features:
        summary = ", ".join(f"{k.replace('_', ' ')}: {v}" for k, v in new_features.items())
        progress += f"Got it, I've noted: {summary}\n\n"
    extra_info = state["extra_info"]
    if extra_info and len(extra_info) > len(state.get("_prev_extra_len", [])):
        new_extras = extra_info[-(len(extra_info) - len(state.get("_prev_extra_len", []))):]
        if new_extras:
            progress += f"I've also noted the additional context: {', '.join(new_extras)}\n\n"
    followup = build_followup_question(missing, state["collected_features"])
    return {"response": progress + "I still need a few more details:\n\n" + followup,
            "phase": "collecting"}


def node_predict(state: FleetState) -> FleetState:
    """Node: run ML prediction on collected features."""
    prediction = predict(state["collected_features"])
    return {"prediction": prediction, "risk_level": prediction["risk_level"]}


def node_retrieve_modifiers(state: FleetState) -> FleetState:
    """Node: RAG retrieval of risk modifier docs based on extra info."""
    mods = retrieve_risk_modifiers(state["extra_info"])
    return {"risk_modifiers": mods}


def node_apply_modifiers(state: FleetState) -> FleetState:
    """Node: LLM-driven risk adjustment from extra info."""
    adjusted = apply_risk_modifiers(
        state["prediction"], state["risk_modifiers"],
        state["extra_info"], state.get("api_key")
    )
    return {"prediction": adjusted, "risk_level": adjusted["risk_level"]}


def node_retrieve_guidelines(state: FleetState) -> FleetState:
    """Node: RAG retrieval of maintenance guidelines."""
    guidelines = retrieve_guidelines(state["collected_features"], state["risk_level"])
    return {"guidelines": guidelines}


def node_generate_report(state: FleetState) -> FleetState:
    """Node: LLM structured report generation with trace."""
    report = generate_report(
        state["collected_features"], state["prediction"], state["guidelines"],
        state["extra_info"], state["risk_modifiers"], state.get("api_key")
    )
    new_features = state.get("new_features", {})
    if new_features:
        summary = ", ".join(f"{k.replace('_', ' ')}: {v}" for k, v in new_features.items())
        intro = f"Got it — {summary}.\n\nAll vehicle details collected. Running analysis now...\n\n"
    else:
        intro = "All vehicle details collected. Running analysis now...\n\n"

    trace = f"""---

**Agent Workflow Trace (LangGraph):**
1. **classify_intent** → info
2. **extract_features** → Collected {len(state['collected_features'])} features
3. **predict_maintenance** → risk={state['prediction']['risk_level']}, probability={state['prediction']['probability']:.1%}"""
    if state["extra_info"]:
        trace += f"\n4. **retrieve_risk_modifiers** → {len(state['risk_modifiers'])} modifier docs"
        trace += f"\n5. **apply_risk_modifiers** → adjusted risk"
        trace += f"\n6. **retrieve_guidelines** → {len(state['guidelines'])} guideline docs"
        trace += f"\n7. **generate_report** → {'LLM' if state.get('api_key') else 'rule-based'}"
    else:
        trace += f"\n4. **retrieve_guidelines** → {len(state['guidelines'])} guideline docs"
        trace += f"\n5. **generate_report** → {'LLM' if state.get('api_key') else 'rule-based'}"

    return {"response": intro + report + "\n\n" + trace, "report": report, "phase": "done"}


# ---------------------------------------------------------------------------
# Routing functions (conditional edges)
# ---------------------------------------------------------------------------

def route_after_intent(state: FleetState) -> str:
    """After classifying intent, decide next node."""
    intent = state.get("intent", "info")
    if intent == "offtopic":
        return "offtopic"
    # question/mixed/info all extract first (mixed has info in it too)
    return "extract"


def route_after_extract(state: FleetState) -> str:
    """After extracting features, decide next node."""
    intent = state.get("intent", "info")
    missing = get_missing_features(state["collected_features"])
    if intent in ("question", "mixed") and missing:
        return "answer"
    if missing:
        return "ask_followup"
    return "predict"


# ---------------------------------------------------------------------------
# Build the graph (compiled once)
# ---------------------------------------------------------------------------

def _build_graph():
    g = StateGraph(FleetState)
    g.add_node("classify", node_classify_intent)
    g.add_node("extract", node_extract_features)
    g.add_node("answer", node_answer_question)
    g.add_node("offtopic", node_offtopic)
    g.add_node("ask_followup", node_ask_followup)
    g.add_node("predict", node_predict)
    g.add_node("retrieve_modifiers", node_retrieve_modifiers)
    g.add_node("apply_modifiers", node_apply_modifiers)
    g.add_node("retrieve_guidelines", node_retrieve_guidelines)
    g.add_node("generate_report", node_generate_report)

    g.set_entry_point("classify")
    g.add_conditional_edges(
        "classify", route_after_intent,
        {"offtopic": "offtopic", "extract": "extract"},
    )
    g.add_conditional_edges(
        "extract", route_after_extract,
        {"answer": "answer", "ask_followup": "ask_followup", "predict": "predict"},
    )
    g.add_edge("predict", "retrieve_modifiers")
    g.add_edge("retrieve_modifiers", "apply_modifiers")
    g.add_edge("apply_modifiers", "retrieve_guidelines")
    g.add_edge("retrieve_guidelines", "generate_report")
    g.add_edge("generate_report", END)
    g.add_edge("answer", END)
    g.add_edge("offtopic", END)
    g.add_edge("ask_followup", END)
    return g.compile()


_COMPILED_GRAPH = None


def get_graph():
    """Lazily compile the LangGraph StateGraph."""
    global _COMPILED_GRAPH
    if _COMPILED_GRAPH is None:
        _COMPILED_GRAPH = _build_graph()
    return _COMPILED_GRAPH


# ---------------------------------------------------------------------------
# Main conversational handler (invokes the compiled graph)
# ---------------------------------------------------------------------------

def handle_message(user_message: str, chat_history: list[dict],
                   collected_features: dict, extra_info: list[str],
                   api_key: str | None) -> tuple[str, dict, list[str], str]:
    """Invoke the LangGraph agent and return (response, features, extra_info, phase)."""
    initial_state: FleetState = {
        "user_message": user_message,
        "messages": chat_history + [{"role": "user", "content": user_message}],
        "collected_features": collected_features,
        "new_features": {},
        "extra_info": extra_info,
        "intent": "info",
        "risk_modifiers": [],
        "prediction": {},
        "risk_level": "",
        "guidelines": [],
        "report": "",
        "response": "",
        "phase": "collecting",
        "api_key": api_key or "",
        "error": "",
    }
    final_state = get_graph().invoke(initial_state)
    return (
        final_state["response"],
        final_state["collected_features"],
        final_state["extra_info"],
        final_state["phase"],
    )


# ---------------------------------------------------------------------------
# Legacy procedural handler (kept for reference / tests)
# ---------------------------------------------------------------------------

def _handle_message_procedural(user_message: str, chat_history: list[dict],
                               collected_features: dict, extra_info: list[str],
                               api_key: str | None) -> tuple[str, dict, list[str], str]:
    """Procedural fallback — mirrors the graph flow without LangGraph."""
    messages = chat_history + [{"role": "user", "content": user_message}]
    intent_result = classify_intent(user_message, api_key)
    intent = intent_result.get("intent", "info")

    if intent in ("info", "mixed", "question"):
        new_features, new_extra = extract_features(messages, api_key, collected_features)
        negation_features = apply_negation_heuristics(user_message, collected_features)
        for k, v in negation_features.items():
            new_features.setdefault(k, v)
        collected_features = {**collected_features, **new_features}
        extra_info = extra_info + [e for e in new_extra if e and e not in extra_info]
    else:
        new_features, new_extra = {}, []

    missing = get_missing_features(collected_features)

    if intent in ("question", "mixed") and missing:
        answer = answer_question(user_message, missing, collected_features, api_key)
        n_collected = len(REQUIRED_FEATURES) - len(missing)
        header = f"**[{n_collected}/{len(REQUIRED_FEATURES)} features collected]**\n\n"
        if new_features:
            summary = ", ".join(f"{k.replace('_', ' ')}: {v}" for k, v in new_features.items())
            header += f"Got it, I've noted: {summary}\n\n"
        return header + answer, collected_features, extra_info, "collecting"

    if intent == "offtopic":
        reply = ("I can only help with vehicle maintenance and fleet management questions. "
                 "Could we get back to your vehicle? " +
                 (build_followup_question(missing, collected_features) if missing else ""))
        return reply, collected_features, extra_info, "collecting" if missing else "done"

    if missing:
        # Still need more info
        n_collected = len(REQUIRED_FEATURES) - len(missing)
        n_total = len(REQUIRED_FEATURES)
        progress = f"**[{n_collected}/{n_total} features collected]**\n\n"

        if new_features:
            extracted_summary = ", ".join(
                f"{k.replace('_', ' ')}: {v}" for k, v in new_features.items()
            )
            progress += f"Got it, I've noted: {extracted_summary}\n\n"

        if new_extra:
            progress += f"I've also noted the additional context: {', '.join(new_extra)}\n\n"

        followup = build_followup_question(missing, collected_features)
        response = progress + "I still need a few more details:\n\n" + followup

        return response, collected_features, extra_info, "collecting"

    else:
        # All features collected — run analysis
        if new_features:
            extracted_summary = ", ".join(
                f"{k.replace('_', ' ')}: {v}" for k, v in new_features.items()
            )
            intro = f"Got it — {extracted_summary}.\n\n"
        else:
            intro = ""

        intro += "All vehicle details collected. Running analysis now...\n\n"

        # Run prediction
        prediction = predict(collected_features)
        risk_level = prediction["risk_level"]

        # Retrieve risk modifiers for extra info
        risk_mods = retrieve_risk_modifiers(extra_info)

        # Adjust prediction based on extra info
        adjusted_prediction = apply_risk_modifiers(
            prediction, risk_mods, extra_info, api_key
        )

        # Retrieve maintenance guidelines
        guidelines = retrieve_guidelines(collected_features, adjusted_prediction["risk_level"])

        # Generate report
        report = generate_report(
            collected_features, adjusted_prediction, guidelines,
            extra_info, risk_mods, api_key
        )

        # Build trace
        trace = f"""---

**Agent Workflow Trace:**
1. **extract_features** — Collected {len(collected_features)} features via conversation
2. **predict_maintenance** — ML model: risk={adjusted_prediction['risk_level']}, probability={adjusted_prediction['probability']:.1%}"""

        if extra_info:
            trace += f"\n3. **retrieve_risk_modifiers** — Found {len(risk_mods)} risk modifier guidelines for: {', '.join(extra_info[:3])}"
            trace += f"\n4. **apply_risk_modifiers** — Adjusted risk based on additional context"
            trace += f"\n5. **retrieve_guidelines** — Retrieved {len(guidelines)} maintenance guidelines via RAG"
            trace += f"\n6. **generate_report** — {'LLM-generated' if api_key else 'Rule-based'} structured report"
        else:
            trace += f"\n3. **retrieve_guidelines** — Retrieved {len(guidelines)} maintenance guidelines via RAG"
            trace += f"\n4. **generate_report** — {'LLM-generated' if api_key else 'Rule-based'} structured report"

        response = intro + report + "\n\n" + trace

        return response, collected_features, extra_info, "done"
