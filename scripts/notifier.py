import os
import requests
import logging

# Configuration
WEBHOOK_URL = os.getenv("ALERT_WEBHOOK_URL", "") # User should provide this
PLATFORM = os.getenv("ALERT_PLATFORM", "DISCORD") # DISCORD or TELEGRAM

logger = logging.getLogger("Notifier")

def send_alert(item_name: str, target_price: float, confidence: float, predicted_profit: float):
    """
    Sends a high-priority alert via Webhook.
    
    Format: 🚨 ALPHA SIGNAL: {ItemName} | Target: {Price} ISK | Confidence: {65%}
    """
    if not WEBHOOK_URL:
        # If no webhook set, just log it.
        logger.warning(f"🚨 ALPHA SIGNAL (No Webhook): {item_name} | Target: {target_price:,.2f} | Conf: {confidence:.2%}")
        return

    confidence_pct = f"{confidence:.2%}"
    price_fmt = f"{target_price:,.2f}"
    profit_fmt = f"{predicted_profit:,.2f}"

    message = f"🚨 **ALPHA SIGNAL**: {item_name}\n" \
              f"🎯 **Target**: {price_fmt} ISK\n" \
              f"🧠 **Confidence**: {confidence_pct}\n" \
              f"💰 **Proj. Profit**: {profit_fmt} ISK"

    try:
        if PLATFORM.upper() == "DISCORD":
            payload = {"content": message}
            response = requests.post(WEBHOOK_URL, json=payload, timeout=5)
        elif PLATFORM.upper() == "TELEGRAM":
            # Assuming WEBHOOK_URL is https://api.telegram.org/bot<token>/sendMessage?chat_id=<id>
            # OR user provided raw params. Let's assume standard webhook structure.
            payload = {"text": message}
            response = requests.post(WEBHOOK_URL, json=payload, timeout=5)
            
        if response.status_code not in [200, 204]:
             logger.error(f"Failed to send alert: {response.status_code} - {response.text}")
        else:
             logger.info(f"Alert sent for {item_name}")

    except Exception as e:
        logger.error(f"Alert Transport Error: {e}")

if __name__ == "__main__":
    # Test
    logging.basicConfig(level=logging.INFO)
    send_alert("TestItem", 100.0, 0.95, 50000000.0)
