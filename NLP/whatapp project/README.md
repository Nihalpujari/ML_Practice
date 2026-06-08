# 💬 WhatsApp Chat Analysis Project

> Turn a WhatsApp chat export into a data-science playground — who talks the most, when, and about what?

---

## 📂 Contents

| File | Purpose |
|------|---------|
| `whatsapp.ipynb` | Parse, clean and analyze an exported WhatsApp chat |

---

## 📥 How To Get a WhatsApp Export

1. Open the chat in WhatsApp
2. Tap **⋮** → **More** → **Export chat** → **Without media**
3. You'll get a `.txt` file like:
   ```
   12/05/24, 9:31 PM - Alice: hey
   12/05/24, 9:32 PM - Bob: 👋
   12/05/24, 9:32 PM - Bob: free tonight?
   ```

---

## 🎯 What the Notebook Does

1. **Parse** the raw `.txt` into a structured DataFrame:
   `date | time | user | message`
2. **Clean** messages (drop `<Media omitted>`, system messages, emojis if needed)
3. **Stats per user** — message count, words per message, most active hours
4. **Time analysis** — messages over time, busiest day / hour
5. **Word clouds** per user
6. **Sentiment analysis** (optional)
7. **Emoji frequency** (optional)

---

## 🛠️ Requirements

```bash
pip install pandas numpy matplotlib seaborn wordcloud nltk emoji
```

---

## ▶️ Run

```bash
jupyter notebook whatsapp.ipynb
```

---

## 🔒 Privacy

⚠️ Don't commit real chat exports to a public repo. Use a **sample/anonymized** file or `.gitignore` your input `.txt`.

---

## 💡 Extensions

- Compare two chats (groups vs DMs)
- Detect language switches (English ↔ Hindi, etc.)
- Train a per-user "style classifier" — predict who sent a message
