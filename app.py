from flask import Flask, request, jsonify
from flask_cors import CORS
from datetime import datetime, timedelta
from pymongo import MongoClient
from twilio.rest import Client as TwilioClient
import joblib
import os
import numpy as np

app = Flask(__name__)
CORS(app)

# ===============================
# MongoDB
# ===============================
mongo_client       = MongoClient("mongodb://localhost:27017/")
db                 = mongo_client["mhews"]
collection         = db["sensor_data"]
alerts_collection  = db["alerts"]
sms_log_collection = db["sms_log"]      # tracks sent SMS to avoid spam

# ===============================
# Twilio Configuration
# ===============================
TWILIO_ACCOUNT_SID = os.environ.get("TWILIO_ACCOUNT_SID", "YOUR_ACCOUNT_SID")
TWILIO_AUTH_TOKEN  = os.environ.get("TWILIO_AUTH_TOKEN",  "YOUR_AUTH_TOKEN")
TWILIO_FROM_NUMBER = os.environ.get("TWILIO_FROM",        "+1234567890")

# Add all recipient numbers here (with country code)
ALERT_RECIPIENTS = [
    os.environ.get("ALERT_PHONE_1", "+91XXXXXXXXXX"),
    # os.environ.get("ALERT_PHONE_2", "+91XXXXXXXXXX"),  # add more as needed
]

twilio_client = TwilioClient(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)

# ===============================
# SMS Cooldown Config
# ===============================
# Minimum minutes between same-type SMS to prevent spam
SMS_COOLDOWN_MINUTES = {
    "FLOOD":     30,
    "FIRE":      20,
    "AIR":       60,
    "RAIN":      30,
    "RISK_HIGH": 15,
    "MANUAL":     0,   # manual alerts always go through
}

# ===============================
# Sensor Alert Thresholds
# ===============================
THRESHOLDS = {
    "distance":     {"warn": 100, "critical": 50},   # cm — lower = more water
    "rain_level":   {"warn": 70,  "critical": 90},   # 0-100 scale
    "air_quality":  {"warn": 150, "critical": 250},  # PPM
    "temperature":  {"warn": 40,  "critical": 45},   # Celsius
    "soil_moisture":{"warn": 80,  "critical": 95},   # percent
}


# ===============================
# Load ML Model
# ===============================
model = joblib.load("rf_multi_hazard_model.pkl")
EXPECTED_FEATURES = model.n_features_in_
print("Model expects features:", EXPECTED_FEATURES)


# ===============================
# Feature Engineering
# ===============================
def build_feature_vector(sensor_data):
    temperature = sensor_data.get("temperature", 0)
    humidity    = sensor_data.get("humidity", 0)
    soil        = sensor_data.get("soil_moisture", 0)
    rain        = sensor_data.get("rain_level", 0)
    air         = sensor_data.get("air_quality", 0)
    distance    = sensor_data.get("distance", 0)

    water_level    = (soil + rain) / 2
    moisture_index = (soil * 0.6 + rain * 0.4)
    dryness_index  = 100 - moisture_index
    heat_index     = temperature + (humidity * 0.1)

    features = [
        temperature, humidity, soil, rain, air, distance,
        water_level, moisture_index, dryness_index, heat_index
    ]
    while len(features) < EXPECTED_FEATURES:
        features.append(0)
    return features


# ===============================
# Risk Evaluation
# ===============================
def evaluate_risk(sensor_data):
    try:
        features_array = np.array([build_feature_vector(sensor_data)])
        prediction     = model.predict(features_array)[0]
        return {0: "LOW", 1: "MEDIUM", 2: "HIGH"}.get(prediction, "UNKNOWN")
    except Exception as e:
        print("Prediction Error:", e)
        return "UNKNOWN"


# ===============================
# SMS Core Helpers
# ===============================
def _is_on_cooldown(alert_type: str) -> bool:
    """True if same alert_type was sent within the cooldown window."""
    cooldown_mins = SMS_COOLDOWN_MINUTES.get(alert_type, 30)
    if cooldown_mins == 0:
        return False
    cutoff = (datetime.utcnow() - timedelta(minutes=cooldown_mins)).isoformat()
    return sms_log_collection.find_one({
        "alert_type": alert_type,
        "sent_at":    {"$gte": cutoff}
    }) is not None


def _log_sms(alert_type: str, message: str, recipients: list, sid: str):
    """Persist SMS send record for cooldown tracking and audit."""
    sms_log_collection.insert_one({
        "alert_type": alert_type,
        "message":    message,
        "recipients": recipients,
        "twilio_sid": sid,
        "sent_at":    datetime.utcnow().isoformat(),
    })


def send_sms_alert(alert_type: str, message: str, recipients: list = None) -> dict:
    """
    Send an SMS via Twilio to all recipients.
    Respects per-type cooldown window. Returns a result dict.
    """
    if recipients is None:
        recipients = ALERT_RECIPIENTS

    if _is_on_cooldown(alert_type):
        print(f"[SMS] Cooldown active for '{alert_type}' — skipping.")
        return {"sent": False, "reason": "cooldown"}

    results  = []
    last_sid = None

    try:
        for number in recipients:
            if not number or "XXXXXXXXXX" in number:
                print(f"[SMS] Skipping placeholder number: {number}")
                continue
            msg = twilio_client.messages.create(
                body=message,
                from_=TWILIO_FROM_NUMBER,
                to=number
            )
            last_sid = msg.sid
            results.append({"to": number, "sid": msg.sid, "status": msg.status})
            print(f"[SMS] ✅ Sent to {number} — SID: {msg.sid}")

        if last_sid:
            _log_sms(alert_type, message, recipients, last_sid)
            return {"sent": True, "count": len(results), "results": results}

        return {"sent": False, "reason": "no valid recipients"}

    except Exception as e:
        print(f"[SMS] ❌ Twilio error: {e}")
        return {"sent": False, "reason": str(e)}


# ===============================
# Auto SMS Trigger (from sensors)
# ===============================
def check_and_send_sensor_sms(sensor_data: dict, risk: str):
    """
    Evaluates every incoming sensor reading against thresholds.
    Fires targeted SMS alerts for each hazard type independently.
    """
    distance    = sensor_data.get("distance",      999)
    rain        = sensor_data.get("rain_level",    0)
    air         = sensor_data.get("air_quality",   0)
    temperature = sensor_data.get("temperature",   0)
    ts          = sensor_data.get("timestamp", datetime.utcnow().isoformat())
    ts_fmt      = ts[:19].replace("T", " ") + " UTC"

    # ── Flood / Water Level ────────────────────────────────────────────────────
    if distance <= THRESHOLDS["distance"]["critical"]:
        send_sms_alert(
            "FLOOD",
            f"MHEWS CRITICAL FLOOD ALERT\n"
            f"Water sensor: {distance}cm (DANGER LEVEL)\n"
            f"Immediate evacuation may be required.\n"
            f"Time: {ts_fmt}"
        )
    elif distance <= THRESHOLDS["distance"]["warn"]:
        send_sms_alert(
            "FLOOD",
            f"MHEWS Flood Warning\n"
            f"Water rising. Distance: {distance}cm.\n"
            f"Monitor closely and prepare to evacuate.\n"
            f"Time: {ts_fmt}"
        )

    # ── Heavy Rain ─────────────────────────────────────────────────────────────
    if rain >= THRESHOLDS["rain_level"]["critical"]:
        send_sms_alert(
            "RAIN",
            f"MHEWS Heavy Rain CRITICAL\n"
            f"Rain sensor: {rain}/100\n"
            f"Flash flood risk. Move to higher ground.\n"
            f"Time: {ts_fmt}"
        )
    elif rain >= THRESHOLDS["rain_level"]["warn"]:
        send_sms_alert(
            "RAIN",
            f"MHEWS Rain Advisory\n"
            f"Rain intensity: {rain}/100 (HIGH)\n"
            f"Avoid low-lying areas.\n"
            f"Time: {ts_fmt}"
        )

    # ── Air Quality ────────────────────────────────────────────────────────────
    if air >= THRESHOLDS["air_quality"]["critical"]:
        send_sms_alert(
            "AIR",
            f"MHEWS Air Quality CRITICAL\n"
            f"AQI: {air} PPM — Hazardous.\n"
            f"Stay indoors. Wear masks immediately.\n"
            f"Time: {ts_fmt}"
        )
    elif air >= THRESHOLDS["air_quality"]["warn"]:
        send_sms_alert(
            "AIR",
            f"MHEWS Air Quality Warning\n"
            f"AQI: {air} PPM — Unhealthy.\n"
            f"Limit outdoor exposure.\n"
            f"Time: {ts_fmt}"
        )

    # ── Temperature / Fire Risk ────────────────────────────────────────────────
    if temperature >= THRESHOLDS["temperature"]["critical"]:
        send_sms_alert(
            "FIRE",
            f"MHEWS Extreme Heat / Fire Risk\n"
            f"Temperature: {temperature}C (CRITICAL)\n"
            f"Extreme fire danger conditions.\n"
            f"Time: {ts_fmt}"
        )
    elif temperature >= THRESHOLDS["temperature"]["warn"]:
        send_sms_alert(
            "FIRE",
            f"MHEWS Heat Warning\n"
            f"Temperature: {temperature}C (HIGH)\n"
            f"High fire risk. Stay alert.\n"
            f"Time: {ts_fmt}"
        )

    # ── ML Model HIGH Risk (catch-all) ─────────────────────────────────────────
    if risk == "HIGH":
        send_sms_alert(
            "RISK_HIGH",
            f"MHEWS SYSTEM ALERT — HIGH RISK\n"
            f"ML model predicts HIGH hazard risk.\n"
            f"Temp:{temperature}C Rain:{rain} Water:{distance}cm AQI:{air}\n"
            f"Time: {ts_fmt}"
        )


# ===============================
# Routes
# ===============================

@app.route("/")
def home():
    return "MHEWS Backend Running"


@app.route("/sensor-data", methods=["POST"])
def receive_sensor_data():
    try:
        data = request.json
        if not data:
            return {"error": "No JSON received"}, 400

        data["timestamp"] = datetime.utcnow().isoformat()
        risk              = evaluate_risk(data)
        data["risk"]      = risk

        collection.insert_one(data)

        # Trigger SMS checks on every reading
        check_and_send_sensor_sms(data, risk)

        return jsonify({"status": "success", "risk": risk})

    except Exception as e:
        print("Server Error:", e)
        return {"error": "Internal Server Error"}, 500


@app.route("/latest-data", methods=["GET"])
def latest_data():
    try:
        latest = collection.find().sort("_id", -1).limit(1)
        for doc in latest:
            doc["_id"] = str(doc["_id"])
            return doc
        return {}
    except Exception as e:
        return {"error": "Could not fetch latest data"}


# ── Manual SMS broadcast (called from frontend dashboard) ─────────────────────
@app.route("/send-sms", methods=["POST"])
def manual_send_sms():
    """
    Operator-triggered SMS blast from the dashboard.
    Body: { "message": "...", "recipients": ["+91..."] (optional) }
    """
    try:
        data       = request.json or {}
        message    = data.get("message", "").strip()
        if not message:
            return jsonify({"error": "message is required"}), 400

        recipients = data.get("recipients") or ALERT_RECIPIENTS
        result     = send_sms_alert("MANUAL", message, recipients)
        return jsonify(result)

    except Exception as e:
        print("Manual SMS Error:", e)
        return jsonify({"error": str(e)}), 500


# ── SMS audit log viewer ───────────────────────────────────────────────────────
@app.route("/sms-log", methods=["GET"])
def get_sms_log():
    """Returns last 50 SMS entries for audit/dashboard display."""
    try:
        logs = list(sms_log_collection.find().sort("_id", -1).limit(50))
        for l in logs:
            l["_id"] = str(l["_id"])
        return jsonify(logs)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ===============================
# Weekly Risk Route
# ===============================
RISK_SCORE_MAP = {"LOW": 1, "MEDIUM": 2, "HIGH": 3, "UNKNOWN": 0}
RISK_LABEL_MAP = {1: "LOW", 2: "MEDIUM", 3: "HIGH", 0: "UNKNOWN"}

@app.route("/weekly-risk", methods=["GET"])
def weekly_risk():
    try:
        today      = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
        week_start = today - timedelta(days=6)

        day_buckets = {}
        for i in range(7):
            day = week_start + timedelta(days=i)
            day_buckets[day.strftime("%Y-%m-%d")] = {
                "name": day.strftime("%a"), "total_score": 0, "count": 0
            }

        for doc in collection.find({"timestamp": {"$gte": week_start.isoformat()}}):
            try:
                day_key = datetime.fromisoformat(doc["timestamp"]).strftime("%Y-%m-%d")
                score   = RISK_SCORE_MAP.get(doc.get("risk", "UNKNOWN"), 0)
                if day_key in day_buckets:
                    day_buckets[day_key]["total_score"] += score
                    day_buckets[day_key]["count"]       += 1
            except Exception:
                continue

        result = []
        for i in range(7):
            day_key = (week_start + timedelta(days=i)).strftime("%Y-%m-%d")
            bucket  = day_buckets[day_key]
            count   = bucket["count"]
            avg     = (bucket["total_score"] / count) if count > 0 else 0
            result.append({
                "name":       bucket["name"],
                "risk":       round(avg, 2),
                "risk_label": RISK_LABEL_MAP.get(round(avg), "NONE") if count > 0 else "NONE",
                "count":      count
            })
        return jsonify(result)

    except Exception as e:
        print("Weekly Risk Error:", e)
        return jsonify({"error": "Could not fetch weekly risk data"}), 500


# ===============================
# Alerts Routes
# ===============================

def severity_order(s):
    return {"Critical": 0, "High": 1, "Moderate": 2, "Low": 3}.get(s, 4)


@app.route("/alerts", methods=["GET"])
def get_alerts():
    try:
        docs = list(alerts_collection.find().sort("_id", -1).limit(100))
        for doc in docs:
            doc["_id"] = str(doc["_id"])
        docs.sort(key=lambda d: (
            0 if d.get("status") == "Active" else 1,
            severity_order(d.get("severity", ""))
        ))
        return jsonify(docs)
    except Exception as e:
        print("Alerts GET Error:", e)
        return jsonify({"error": "Could not fetch alerts"}), 500


@app.route("/alerts", methods=["POST"])
def create_alert():
    try:
        data = request.json
        if not data:
            return jsonify({"error": "No JSON received"}), 400

        for field in ["type", "severity", "location", "message"]:
            if not data.get(field):
                return jsonify({"error": f"Missing field: {field}"}), 400

        count    = alerts_collection.count_documents({})
        year     = datetime.utcnow().year
        alert_id = f"AL-{year}-{str(count + 1).zfill(3)}"

        alert = {
            "id":        alert_id,
            "type":      data["type"],
            "severity":  data["severity"],
            "location":  data["location"],
            "message":   data["message"],
            "channels":  data.get("channels", []),
            "status":    "Active",
            "timestamp": datetime.utcnow().isoformat(),
        }
        alerts_collection.insert_one(alert)
        alert["_id"] = str(alert["_id"])

        # Auto-send SMS if operator selected SMS channel in the dashboard
        if "SMS" in alert.get("channels", []):
            sms_body = (
                f"MHEWS ALERT [{alert['severity'].upper()}]\n"
                f"Type: {alert['type']}\n"
                f"Location: {alert['location']}\n"
                f"{alert['message'][:100]}\n"
                f"Ref: {alert_id}"
            )
            alert["sms_result"] = send_sms_alert("MANUAL", sms_body)

        return jsonify(alert), 201

    except Exception as e:
        print("Alerts POST Error:", e)
        return jsonify({"error": "Could not create alert"}), 500


@app.route("/alerts/<alert_id>/resolve", methods=["PATCH"])
def resolve_alert(alert_id):
    try:
        result = alerts_collection.update_one(
            {"id": alert_id},
            {"$set": {"status": "Resolved", "resolved_at": datetime.utcnow().isoformat()}}
        )
        if result.matched_count == 0:
            return jsonify({"error": "Alert not found"}), 404
        return jsonify({"success": True, "id": alert_id})
    except Exception as e:
        return jsonify({"error": "Could not resolve alert"}), 500


@app.route("/alerts/<alert_id>", methods=["DELETE"])
def delete_alert(alert_id):
    try:
        result = alerts_collection.delete_one({"id": alert_id})
        if result.deleted_count == 0:
            return jsonify({"error": "Alert not found"}), 404
        return jsonify({"success": True})
    except Exception as e:
        return jsonify({"error": "Could not delete alert"}), 500


# ===============================
# Run Server
# ===============================
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)