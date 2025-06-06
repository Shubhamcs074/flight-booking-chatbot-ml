import spacy
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import pandas as pd
import dateparser
import os


flights_df = pd.read_csv("flights.csv")


if not os.path.exists("bookings.csv"):
    with open("bookings.csv", "w") as f:
        f.write("name,flight_no,from,to,date,departure,price\n")



training_sentences = [
    "Book a flight from Delhi to Mumbai",
    "I want to book a ticket to Chennai",
    "Cancel my flight",
    "I want to cancel",
    "Search flights from Pune to Bangalore",
    "Hello",
    "Show my bookings",
    "View booking details",
    "Can I see my bookings?",
    "Display my flight reservations"
]

training_labels = [
   "book_flight",
   "book_flight",
   "cancel_flight",
   "cancel_flight",
   "search_flight",
   "greeting",
   "view_booking",
   "view_booking",
   "view_booking",
   "view_booking"
]


vectorizer = CountVectorizer()
X = vectorizer.fit_transform(training_sentences)


model = MultinomialNB()
model.fit(X, training_labels)


nlp = spacy.load("en_core_web_sm")


def search_flights(source, destination, travel_date=None):
    filtered = flights_df[
        (flights_df['from'].str.lower() == source.lower()) &
        (flights_df['to'].str.lower() == destination.lower())
    ]

    if travel_date:
        travel_date = dateparser.parse(travel_date)
        if travel_date:
            travel_date_str = travel_date.strftime("%Y-%m-%d")
            filtered.loc[:, "date"] = travel_date_str 

    return filtered


def book_flight(name, source, destination, travel_date=None):
    flights = search_flights(source, destination, travel_date)
    if flights.empty:
        return None

    selected = flights.iloc[0]

    parsed_date = dateparser.parse(travel_date) if travel_date else None
    formatted_date = parsed_date.strftime("%Y-%m-%d") if parsed_date else "unknown"

    with open("bookings.csv", "a") as f:
        f.write(f"{name},{selected['flight_no']},{source},{destination},{formatted_date},{selected['departure']},{selected['price']}\n")

    return selected

def view_bookings(name=None, travel_date=None):
    if not os.path.exists("bookings.csv"):
        print("No bookings found.")
        return

    bookings = pd.read_csv("bookings.csv")
    
    if name:
        bookings = bookings[bookings['name'].str.lower() == name.lower()]
    
    if travel_date:
        parsed_date = dateparser.parse(travel_date)
        if parsed_date:
            date_str = parsed_date.strftime("%Y-%m-%d")
            bookings = bookings[bookings['date'] == date_str]

    if bookings.empty:
        print("🔍 No matching bookings found.")
    else:
        print("📋 Your Bookings:")
        for _, row in bookings.iterrows():
            print(f"→ {row['name']} | {row['flight_no']} | {row['from']} → {row['to']} | {row['date']} at {row['departure']} | ₹{row['price']}")


def cancel_booking(name, destination=None):
    if not os.path.exists("bookings.csv"):
        print("No bookings to cancel.")
        return

    bookings = pd.read_csv("bookings.csv")
    original_len = len(bookings)

    if destination:
        bookings = bookings[~((bookings['name'].str.lower() == name.lower()) & 
                              (bookings['to'].str.lower() == destination.lower()))]
    else:
        bookings = bookings[bookings['name'].str.lower() != name.lower()]

    if len(bookings) < original_len:
        bookings.to_csv("bookings.csv", index=False)
        print("✅ Booking canceled successfully.")
    else:
        print("❌ No matching booking found to cancel.")


user_input = input("You: ")


X_test = vectorizer.transform([user_input])
intent = model.predict(X_test)[0]

doc = nlp(user_input)
cities = [ ent.text for ent in doc.ents if ent.label_ == "GPE"]
dates = [ent.text for ent in doc.ents if ent.label_ == "DATE"]

print("\n Chatbot Response: ")
print(f"-> Intent: {intent}")

if intent in ["book_flight", "search_flight"]:
    if len(cities) >= 2:
        source = cities[0]
        destination = cities[1]
        print(f"→ From: {source}")
        print(f"→ To: {destination}")

        travel_date = dates[0] if dates else None

        name = input("→ Please enter your name to confirm booking: ")
        booked = book_flight(name, source, destination, travel_date)

        if booked is not None:
            print("✅ Booking Confirmed!")
            print(f"  ✈ Flight: {booked['flight_no']} | Departure: {booked['departure']} | ₹{booked['price']}")
        else:
            print("❌ No available flights to book.")
    else:
        print("❌ Please provide both source and destination.")

    if dates:
        print(f"→ Date: {dates[0]}")
    else:
        print("→ Date not found.")

    
elif intent == "cancel_flight":
        print("→ Intent: cancel_flight")
        name = input("→ Enter your name: ")
        dest = cities[0] if cities else None
        cancel_booking(name, dest)

elif intent == "greeting":
    print("-> Hello! How can I assist you today?")
    
elif intent == "view_booking":
    name = input("→ Enter your name to view bookings: ")
    travel_date = input("→ Optional: Enter date (or press Enter to skip): ")
    travel_date = travel_date if travel_date.strip() else None
    view_bookings(name, travel_date)

else:
    print("-> Sorry, I didn't understand that.")

