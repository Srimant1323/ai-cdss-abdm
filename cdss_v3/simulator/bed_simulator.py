"""
bed_simulator.py
================
Simulates an ABDM-compliant Hospital Management Information System (HMIS)
publishing real-time bed occupancy events via MQTT.

In a production deployment, this publisher would be replaced by:
  - The hospital's ADT (Admission-Discharge-Transfer) module
  - HL7 ADT^A01 (admit), ADT^A03 (discharge), ADT^A02 (transfer) events
  - IoT bed pressure sensors publishing via MQTT
  - ABDM HFR API endpoints for facility resource data

Architecture:
  Publisher (this script / hospital HMIS)
      │
      │ MQTT  topic: abdm/facility/{hospital_id}/beds/{ward_type}
      │       payload: JSON {beds_occupied, beds_total, timestamp, ...}
      ▼
  MQTT Broker (mosquitto / HiveMQ / AWS IoT Core)
      │
      ▼
  Subscriber (main Streamlit app)
      └── Updates BedOccupancy.csv + triggers UI refresh

References:
  - HL7 ADT messaging: https://www.hl7.org/implement/standards/
  - ABDM Health Information Exchange: https://sandbox.abdm.gov.in/
  - IoMT MQTT standard: JMIR 2025, e54470
"""

import paho.mqtt.client as mqtt
import json
import time
import random
import pandas as pd
from datetime import datetime
import os
import threading


BROKER_HOST  = "broker.hivemq.com"   # Free public broker — replace with mosquitto in production
BROKER_PORT  = 1883
TOPIC_BASE   = "abdm/facility"
DATA_PATH    = os.path.join(os.path.dirname(__file__), "..", "data", "BedOccupancy.csv")
UPDATE_INTERVAL = 45    # seconds between bed status updates


def load_bed_data():
    return pd.read_csv(DATA_PATH)


def simulate_admission_discharge_event(row: dict) -> dict:
    """
    Simulate a realistic ADT event:
    - 60% chance: discharge → beds_occupied decreases by 1–3
    - 40% chance: admission → beds_occupied increases by 1–3
    - Constrained by 0 ≤ occupied ≤ total
    - Higher occupancy → higher discharge probability (realistic)
    """
    total     = int(row["beds_total"])
    occupied  = int(row["beds_occupied"])
    occ_rate  = occupied / max(total, 1)

    # Higher occupancy → higher pressure to discharge
    p_discharge = 0.45 + (occ_rate * 0.30)
    p_discharge = min(p_discharge, 0.85)

    if random.random() < p_discharge:
        change = -random.randint(1, min(3, max(1, occupied)))
    else:
        available = total - occupied
        change = random.randint(1, min(3, max(1, available))) if available > 0 else 0

    new_occupied  = max(0, min(total, occupied + change))
    new_available = total - new_occupied
    new_pct       = round(new_occupied / max(total, 1) * 100, 1)

    event_type = "DISCHARGE" if change < 0 else ("ADMISSION" if change > 0 else "NO_CHANGE")

    return {
        "hospital_id":     row["hospital_id"],
        "hospital_name":   row["hospital_name"],
        "ward_type":       row["ward_type"],
        "beds_total":      total,
        "beds_occupied":   new_occupied,
        "beds_available":  new_available,
        "occupancy_pct":   new_pct,
        "event_type":      event_type,
        "delta":           change,
        "last_updated":    datetime.now().isoformat(),
        "hl7_event":       f"ADT^A{'03' if change < 0 else '01'}",  # HL7 standard codes
        "source_system":   "HMIS_SIMULATOR_v1.0",
    }


def update_local_csv(updated_rows: list):
    """Persist updated bed states to local CSV (simulates database write)."""
    df = load_bed_data()
    for row in updated_rows:
        mask = ((df["hospital_id"] == row["hospital_id"]) &
                (df["ward_type"] == row["ward_type"]))
        df.loc[mask, "beds_occupied"]  = row["beds_occupied"]
        df.loc[mask, "beds_available"] = row["beds_available"]
        df.loc[mask, "occupancy_pct"]  = row["occupancy_pct"]
        df.loc[mask, "last_updated"]   = row["last_updated"]
    df.to_csv(DATA_PATH, index=False)


class BedOccupancySimulator:
    """
    Simulates an HMIS publishing bed status to an MQTT broker.
    In production: replace with actual HMIS ADT event listener.
    """

    def __init__(self, use_mqtt: bool = False):
        self.use_mqtt   = use_mqtt
        self.running    = False
        self.client     = None
        self.update_log = []

        if use_mqtt:
            self._setup_mqtt()

    def _setup_mqtt(self):
        self.client = mqtt.Client(
            client_id=f"hmis_simulator_{random.randint(1000,9999)}",
            protocol=mqtt.MQTTv5,
        )
        self.client.on_connect    = self._on_connect
        self.client.on_disconnect = self._on_disconnect
        try:
            self.client.connect(BROKER_HOST, BROKER_PORT, keepalive=60)
            self.client.loop_start()
        except Exception as e:
            print(f"[MQTT] Connection failed: {e}. Running in local-only mode.")
            self.use_mqtt = False

    def _on_connect(self, client, userdata, flags, rc, properties=None):
        print(f"[MQTT] Connected to broker: {BROKER_HOST}:{BROKER_PORT}")

    def _on_disconnect(self, client, userdata, rc, properties=None, reasonCode=None):
        print(f"[MQTT] Disconnected (rc={rc})")

    def _publish(self, hospital_id: str, ward_type: str, payload: dict):
        if self.use_mqtt and self.client:
            topic = f"{TOPIC_BASE}/{hospital_id}/beds/{ward_type.lower()}"
            self.client.publish(topic, json.dumps(payload), qos=1)

    def run_one_cycle(self) -> list:
        """Run one update cycle — simulate ADT events for all wards."""
        df = load_bed_data()
        updated = []

        # Randomly select 30–50% of wards to update each cycle (realistic)
        sample_size = max(1, int(len(df) * random.uniform(0.3, 0.5)))
        rows_to_update = df.sample(n=sample_size, random_state=None)

        for _, row in rows_to_update.iterrows():
            updated_row = simulate_admission_discharge_event(row.to_dict())
            updated.append(updated_row)
            self._publish(updated_row["hospital_id"], updated_row["ward_type"], updated_row)

        update_local_csv(updated)
        return updated

    def run_continuous(self, interval: int = UPDATE_INTERVAL):
        """Run simulator continuously in background thread."""
        self.running = True
        print(f"[Simulator] Starting. Update interval: {interval}s")
        while self.running:
            try:
                updated = self.run_one_cycle()
                ts = datetime.now().strftime("%H:%M:%S")
                print(f"[{ts}] Updated {len(updated)} ward(s)")
            except Exception as e:
                print(f"[Simulator] Error: {e}")
            time.sleep(interval)

    def start_background(self, interval: int = UPDATE_INTERVAL):
        """Start simulator in a daemon thread."""
        t = threading.Thread(
            target=self.run_continuous,
            args=(interval,),
            daemon=True,
            name="BedOccupancySimulator",
        )
        t.start()
        return t

    def stop(self):
        self.running = False
        if self.client:
            self.client.loop_stop()
            self.client.disconnect()


# ─────────────────────────────────────────────────────────────────────────────
# Run standalone for testing
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Starting ABDM Bed Occupancy Simulator (standalone)")
    print(f"Publishing to MQTT broker: {BROKER_HOST}:{BROKER_PORT}")
    print(f"Topic pattern: {TOPIC_BASE}/{{hospital_id}}/beds/{{ward}}")
    print(f"Update interval: {UPDATE_INTERVAL}s")
    print("Press Ctrl+C to stop.\n")

    sim = BedOccupancySimulator(use_mqtt=True)
    try:
        while True:
            updates = sim.run_one_cycle()
            for u in updates:
                print(f"  [{u['event_type']}] {u['hospital_name']} / {u['ward_type']} "
                      f"→ {u['beds_occupied']}/{u['beds_total']} occupied "
                      f"({u['occupancy_pct']}%) | HL7: {u['hl7_event']}")
            print()
            time.sleep(UPDATE_INTERVAL)
    except KeyboardInterrupt:
        sim.stop()
        print("\nSimulator stopped.")
