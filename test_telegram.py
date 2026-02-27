cat > test_telegram.py << 'EOF'
"""
🧪 Quick Telegram Test
"""

import requests
import os
from dotenv import load_dotenv

load_dotenv()

TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

def test_telegram():
    """Test sending a message to Telegram"""
    
    url = f"https://api.telegram.org/bot{TOKEN}/sendMessage"
    
    test_message = """🚀 <b>Тестовое сообщение</b>

Это тест Telegram бота для торгового ассистента.

📊 Пример сигнала:
• BTCUSDT: $52341
• RSI: 72.5 (перекуплен)
• Решение: ПРОДАЖА

<i>Если вы это видите - бот работает!</i>"""
    
    payload = {
        "chat_id": CHAT_ID,
        "text": test_message,
        "parse_mode": "HTML",
        "disable_web_page_preview": True
    }
    
    try:
        response = requests.post(url, json=payload, timeout=10)
        response.raise_for_status()
        print("✅ Тестовое сообщение отправлено!")
        print(f"📱 Проверьте Telegram: https://t.me/MantraTrada561Bot")
        return True
    except Exception as e:
        print(f"❌ Ошибка: {e}")
        return False

if __name__ == "__main__":
    test_telegram()
EOF
