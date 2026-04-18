"""Gradio UI — Conversational Fleet Management Assistant (Milestone 2)."""

import warnings
warnings.filterwarnings("ignore")

import os
import gradio as gr
from agent import handle_message, REQUIRED_FEATURES
from model_utils import SAMPLE_VEHICLES


# ── CSS (matching mid-sem style) ────────────────────────────────
CSS = """
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@300;400;500;600&display=swap');

* { font-family: 'JetBrains Mono', monospace !important; }

.gradio-container {
    max-width: 1200px !important;
    margin: 0 auto !important;
    padding: 2rem !important;
}

#header {
    border-bottom: 1px solid #333;
    padding-bottom: 1rem;
    margin-bottom: 1.5rem;
}

#header h1 {
    font-size: 1.4rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.08em !important;
    text-transform: uppercase !important;
    margin: 0 0 0.25rem 0 !important;
}

#header p { font-size: 0.8rem !important; opacity: 0.55; margin: 0 !important; }

.section-label {
    font-size: 0.65rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.15em !important;
    text-transform: uppercase !important;
    opacity: 0.45;
    margin-bottom: 0.75rem !important;
    border-bottom: 1px solid #2a2a2a;
    padding-bottom: 0.4rem;
}

label span { font-size: 0.7rem !important; letter-spacing: 0.05em !important; opacity: 0.7; }
"""


WELCOME_MESSAGE = """Welcome to the **Fleet Management AI Assistant**.

Describe your vehicle and I'll analyze its maintenance needs. You can say something like:

> *"I have a 5-year-old diesel truck with 80,000 km on it. The brakes are worn out and
> the battery is weak. It hasn't been serviced in about 8 months. I mostly drive on
> rough unpaved roads for construction work."*

I'll extract what I can and ask follow-up questions for anything I still need. Any extra
context you share (driving conditions, symptoms, modifications, etc.) will be factored
into the risk assessment.

**Or select a sample vehicle below to auto-fill.**"""


# ── Chat handler ────────────────────────────────────────────────
def respond(user_message, chat_history, collected_features, extra_info, api_key):
    """Handle a chat message."""
    if not user_message.strip():
        return chat_history, collected_features, extra_info

    key = api_key or os.environ.get("GROQ_API_KEY", "")

    if not key:
        chat_history = chat_history + [
            {"role": "user", "content": user_message},
            {"role": "assistant", "content": "Please provide a Groq API key to use "
             "the conversational interface. The LLM is needed to extract vehicle "
             "features from your description."},
        ]
        return chat_history, collected_features, extra_info

    # Build message list for context
    messages_for_agent = []
    for msg in chat_history:
        messages_for_agent.append({"role": msg["role"], "content": msg["content"]})

    response, updated_features, updated_extra, phase = handle_message(
        user_message, messages_for_agent, collected_features, extra_info, key
    )

    chat_history = chat_history + [
        {"role": "user", "content": user_message},
        {"role": "assistant", "content": response},
    ]

    return chat_history, updated_features, updated_extra


def load_sample(sample_name, chat_history, collected_features, extra_info, api_key):
    """Load a sample vehicle into the chat as if the user described it."""
    if sample_name not in SAMPLE_VEHICLES:
        return chat_history, collected_features, extra_info

    v = SAMPLE_VEHICLES[sample_name]

    # Build a natural language description from the sample
    desc_parts = [
        f"I have a {v['Vehicle_Age']}-year-old {v['Fuel_Type'].lower()} {v['Vehicle_Model'].lower()}",
        f"with {v['Mileage']:,} km mileage",
        f"and {v['Odometer_Reading']:,} km on the odometer.",
        f"It's a {v['Transmission_Type'].lower()} transmission, {v['Engine_Size']}cc engine.",
        f"I'm the {v['Owner_Type'].lower()} owner.",
        f"Maintenance history is {v['Maintenance_History'].lower()}.",
        f"Tires are {v['Tire_Condition'].lower()},",
        f"brakes are {v['Brake_Condition'].lower()},",
        f"battery is {v['Battery_Status'].lower()}.",
        f"It has {v['Reported_Issues']} reported issues,",
        f"{v['Service_History']} services done,",
        f"and {v['Accident_History']} accident(s).",
        f"Fuel efficiency is {v['Fuel_Efficiency']} km/l.",
        f"Insurance premium is {v['Insurance_Premium']}.",
        f"Last serviced {v['Last_Service_Date_days']} days ago.",
    ]
    if v["Warranty_Expiry_Date_days"] < 0:
        desc_parts.append(f"Warranty expired {abs(v['Warranty_Expiry_Date_days'])} days ago.")
    else:
        desc_parts.append(f"Warranty expires in {v['Warranty_Expiry_Date_days']} days.")

    description = " ".join(desc_parts)

    return respond(description, chat_history, collected_features, extra_info, api_key)


def reset_chat():
    """Reset the conversation."""
    return (
        [{"role": "assistant", "content": WELCOME_MESSAGE}],
        {},
        [],
    )


# ── Gradio UI ───────────────────────────────────────────────────
def build_app():
    """Build the Gradio interface."""
    with gr.Blocks(
        title="Fleet Management AI Assistant",
        theme=gr.themes.Monochrome(),
        css=CSS,
    ) as demo:

        # State
        collected_features = gr.State({})
        extra_info = gr.State([])

        with gr.Column(elem_id="header"):
            gr.Markdown("# Fleet Management AI Assistant")
            gr.Markdown("Conversational vehicle maintenance prediction & recommendations")

        with gr.Row():
            api_key = gr.Textbox(
                label="Groq API Key",
                placeholder="gsk_... (required for conversation)",
                type="password",
                scale=3,
            )
            sample_dropdown = gr.Dropdown(
                choices=[""] + list(SAMPLE_VEHICLES.keys()),
                label="Load Sample Vehicle",
                value="",
                scale=2,
            )

        gr.Markdown("Conversation", elem_classes="section-label")

        chatbot = gr.Chatbot(
            value=[{"role": "assistant", "content": WELCOME_MESSAGE}],
            height=550,
            type="messages",
            show_copy_button=True,
        )

        with gr.Row():
            msg_input = gr.Textbox(
                placeholder="Describe your vehicle... (e.g., 'I have a 10-year-old diesel truck with worn brakes')",
                label="Your Message",
                scale=5,
                lines=2,
            )
            send_btn = gr.Button("Send", variant="primary", scale=1)

        with gr.Row():
            clear_btn = gr.Button("New Conversation", variant="secondary")

        # ── Wire up events ──
        send_btn.click(
            fn=respond,
            inputs=[msg_input, chatbot, collected_features, extra_info, api_key],
            outputs=[chatbot, collected_features, extra_info],
        ).then(lambda: "", outputs=msg_input)

        msg_input.submit(
            fn=respond,
            inputs=[msg_input, chatbot, collected_features, extra_info, api_key],
            outputs=[chatbot, collected_features, extra_info],
        ).then(lambda: "", outputs=msg_input)

        sample_dropdown.change(
            fn=load_sample,
            inputs=[sample_dropdown, chatbot, collected_features, extra_info, api_key],
            outputs=[chatbot, collected_features, extra_info],
        )

        clear_btn.click(
            fn=reset_chat,
            outputs=[chatbot, collected_features, extra_info],
        )

    return demo


if __name__ == "__main__":
    app = build_app()
    app.launch(server_name="0.0.0.0", ssr_mode=False)
