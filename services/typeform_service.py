import uuid
import requests
from typing import List
from config import TYPEFORM_API_KEY


def create_typeform_form(role_title: str, jd_markdown: str, fields: List[dict]) -> str:
    """Creates a live Typeform via the Typeform Create API and returns the public URL."""
    if not TYPEFORM_API_KEY or TYPEFORM_API_KEY.startswith("tfp_your"):
        # Fallback mock URL so local testing never crashes without a Typeform token
        mock_slug = uuid.uuid4().hex[:8]
        return f"https://form.typeform.com/to/mock-{mock_slug}?role={role_title.replace(' ', '+')}"

    tf_fields = []
    for idx, f in enumerate(fields):
        ftype = f.get("field_type", "short_text")
        # Ensure valid Typeform field type
        if ftype not in {"short_text", "long_text", "multiple_choice", "number", "email", "website", "yes_no"}:
            ftype = "short_text"

        field_payload = {
            "title": f.get("label", f"Question {idx + 1}"),
            "type": ftype,
            "validations": {"required": bool(f.get("required", True))}
        }
        if ftype == "multiple_choice":
            opts = f.get("options") or ["Yes", "No"]
            field_payload["properties"] = {
                "choices": [{"label": str(o)} for o in opts],
                "allow_multiple_selection": False
            }
        tf_fields.append(field_payload)

    payload = {
        "title": f"Application: {role_title}",
        "welcome_screens": [
            {
                "title": f"Apply for {role_title}",
                "properties": {
                    "show_button": True,
                    "button_text": "Start Application",
                    "description": "Please complete the short screening questions below."
                }
            }
        ],
        "fields": tf_fields
    }

    resp = requests.post(
        "https://api.typeform.com/forms",
        headers={
            "Authorization": f"Bearer {TYPEFORM_API_KEY}",
            "Content-Type": "application/json"
        },
        json=payload,
        timeout=20
    )
    if resp.status_code not in (200, 201):
        raise RuntimeError(f"Typeform API error ({resp.status_code}): {resp.text}")

    data = resp.json()
    return data.get("_links", {}).get("display", f"https://form.typeform.com/to/{data.get('id')}")
