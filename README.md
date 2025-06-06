
# ✈️ Flight Booking Chatbot using Machine Learning

This is a **terminal-based chatbot** that simulates flight booking, searching, viewing, and cancellation. It uses **Natural Language Processing (NLP)** and **Machine Learning (ML)** to detect user intent and extract useful information from human queries.

---

## 🔍 Project Highlights

- 🤖 Built with **Python**, **scikit-learn**, **spaCy**, and **pandas**
- 🧠 Uses ML (Naive Bayes Classifier) to understand **user intent**
- 📅 Extracts city names and travel dates using NLP
- 📁 Saves bookings to a CSV file
- ❌ Supports booking cancellation and viewing history

---

## 🎯 Problem Statement

Create a conversational chatbot that allows users to:
- Book domestic flights
- Search for flights
- Cancel or view bookings
- Understand natural human queries like:
  - “Book a flight from Delhi to Mumbai on June 10”
  - “Cancel my flight to Srinagar”
  - “Show my bookings on June 25”

---

## 🧠 Machine Learning Used

### ✅ Intent Detection with scikit-learn

**Model**: `MultinomialNB`  
**Vectorizer**: `CountVectorizer`

> Input: `"Book a ticket to Chennai"`  
> Output: `"book_flight"` (intent)

**Training Examples (Supervised Learning):**
```python
("Book a flight from Delhi to Mumbai", "book_flight")
("Cancel my flight", "cancel_flight")
("Search flights from Pune to Bangalore", "search_flight")
("Show my bookings", "view_booking")
```

ML Pipeline:
1. Vectorize input text using bag-of-words
2. Train classifier on labeled text-intent pairs
3. Predict intent of new inputs

---

## 🧠 NLP Used

### ✅ spaCy for Entity Extraction
Used to extract:
- **Cities**: using spaCy’s `GPE` entity type
- **Dates**: using `dateparser`

```python
input: "Book a flight from Mumbai to Srinagar on June 25"
output: {
  "from": "Mumbai",
  "to": "Srinagar",
  "date": "2025-06-25"
}
```

---

## 🗃️ Dataset

### 1. **`flights.csv`** – Dummy Air India flight schedule  
Fixed daily flights between Indian cities

| flight_no | from     | to        | departure | price |
|-----------|----------|-----------|-----------|-------|
| AI101     | Delhi    | Mumbai    | 06:00     | 4500  |
| AI202     | Mumbai   | Srinagar  | 09:00     | 5500  |

### 2. **`bookings.csv`** – Logged bookings made via chatbot

---

## ✅ Features

| Feature                 | Description                                   |
|------------------------|-----------------------------------------------|
| ✈ Book Flights         | Match intent, find flights, save to CSV       |
| 🔍 Search Flights       | Filter based on city/date                     |
| 📅 View Bookings        | See past bookings (by name/date)              |
| ❌ Cancel Bookings      | Remove matching booking from file             |
| 🧠 Understand Text      | Intent classification using ML                |

---

## 🚀 How to Run

```bash
python chatbot_combined.py
```

Then type queries like:

```
You: Book a flight from Mumbai to Delhi on June 10
You: Show my bookings
You: Cancel my flight to Delhi
```

---

## 📚 Tech Stack

| Tool           | Purpose                            |
|----------------|------------------------------------|
| Python         | Core programming                   |
| scikit-learn   | Intent classification              |
| spaCy          | Named Entity Recognition           |
| dateparser     | Parsing natural language dates     |
| pandas         | Dataset filtering and CSV handling |

---

## 🔍 Possible Extensions

- Add real airline data from an API
- Use BERT or LSTM instead of Naive Bayes
- Deploy as web app (Flask/Streamlit)
- Add PDF ticket generation
- Add login and auth system

---

## 🧠 Why It’s a Machine Learning Project

This chatbot uses:
- **Supervised learning** to detect intent
- **Text classification pipeline**
- Integration of **ML + rule-based NLP**
- Trains and improves based on labeled queries

> ✨ This makes it a complete beginner-friendly **ML + NLP project** with real-world application.
